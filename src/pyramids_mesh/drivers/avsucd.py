"""AVS-UCD (``.inp`` / ``.avs``) unstructured-cell-data reader.

An AVS-UCD ASCII file opens with a five-integer header
``n_nodes n_cells n_node_data n_cell_data model_data``; then ``n_nodes`` node rows
``node_id x y z``; then ``n_cells`` cell rows ``cell_id material_id cell_type v1 ...``
whose ``cell_type`` is a keyword (``pt``, ``line``, ``tri``, ``quad``, ``tet``,
``pyr``, ``prism``, ``hex``). Each cell's node list is mapped to a fill-padded face
row -- the ``exodus`` precedent -- so a solid mesh of ``tet`` cells reads as four-node
faces exactly as ``exodus`` maps ``TETRA``; ``pt`` and ``line`` cells are dropped.

The ``material_id`` and ``cell_type`` columns are discarded, and the optional node-data
and cell-data sections that may follow the cells are skipped. Node and cell labels may
be gapped or out of order, so connectivity is resolved through
:func:`~pyramids_mesh._text.native_id_map`, never by assuming ``id == index + 1``. See
``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records, native_id_map

# The line-comment marker; `iter_records` skips blank and comment lines alike.
_COMMENT = "#"

# The five-integer header, and the index of the cell count within its fields (the
# keyword of the record is the node count).
_HEADER_FIELDS = 5

# Cell-type keywords whose node list becomes a face. `pt` and `line` are 0-D and 1-D
# and are dropped -- a point or an edge is not a face.
_FACE_CELL_TYPES = frozenset({"tri", "quad", "tet", "pyr", "prism", "hex"})
_SKIP_CELL_TYPES = frozenset({"pt", "line"})

# Short face rows are padded to the widest cell with -1: the vertex ids are already
# 0-based indices (resolved through the id map) by the time they are stored, so 0 is a
# real index and only -1 can mark padding.
_PAD = -1


class AvsUcdDriver(BaseDriver):
    """Reader for the AVS-UCD (``.inp`` / ``.avs``) unstructured mesh format.

    The mesh comes from the node coordinates and the cell node lists. Node labels are
    resolved through :func:`~pyramids_mesh._text.native_id_map`, so a gapped or
    reordered file is wired correctly; the connectivity arrives 0-based with the ``1``
    base recorded for a writer. Elevation becomes a ``node_z`` variable only when the
    ``z`` column is not all zero, matching the other readers.

    Documented limitations:
        - **A volumetric cell becomes a face.** ``tet``, ``pyr``, ``prism`` and ``hex``
          have their node list read straight into a face row (the ``exodus``
          precedent); a tetrahedron is not expanded into its bounding triangles, so its
          winding is judged in the ``xy`` plane like any other face.
        - **``pt`` and ``line`` cells are dropped**, being 0-D and 1-D. A file whose
          only cells are of those types builds no faces and is refused.
        - **The node-data and cell-data sections are skipped.** Only the mesh is built;
          the per-node and per-cell value fields an AVS-UCD file may carry after the
          cells are out of scope for this reader, as are the ``material_id`` and
          ``cell_type`` columns.
        - There is no CRS in the format, so ``crs`` is ``None``.

    Example:
        Read an AVS-UCD mesh (shown, not run as a doctest -- it needs a real file),
        reached through ``read_file``::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("mesh.avs")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "avsucd"
    long_name = "AVS-UCD unstructured cell data"
    suffixes = (".inp", ".avs")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first significant line is exactly five integers.

        An AVS-UCD file opens with the ``n_nodes n_cells n_node_data n_cell_data
        model_data`` header. This is deliberately the inverse of
        :meth:`~pyramids_mesh.drivers.abaqus.AbaqusDriver._sniff`, whose first
        significant line begins a ``*`` keyword, so the two ``.inp`` drivers never both
        accept one file.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first significant line is five integer tokens.
        """
        for keyword, fields in iter_records(path, comment=_COMMENT):
            tokens = [keyword, *fields]
            if len(tokens) != _HEADER_FIELDS:
                return False
            return all(_is_int(token) for token in tokens)
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an AVS-UCD file into a :class:`ParsedMesh`.

        Args:
            path: The ``.inp``/``.avs`` file to read.
            lazy: Unused -- the file is plain text and read eagerly; accepted to match
                the driver protocol.

        Returns:
            ParsedMesh: The surface mesh and, for a 3D file, ``node_z``.

        Raises:
            FileFormatNotSupportedError: If the header is missing or malformed, a
                record is short, or the file carries no face-forming cell.
            ValueError: If a cell references an undefined node label, or two nodes
                share a label.
        """
        return _AvsUcdReader(path).build()


class _AvsUcdReader:
    """One AVS-UCD file: read the header, the nodes, then the cells in order."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.records = iter_records(path, comment=_COMMENT)

    def build(self) -> ParsedMesh:
        """Read the counted sections and assemble the parsed mesh.

        Returns:
            ParsedMesh: Nodes, fill-padded faces, and ``node_z`` for a 3D file.

        Raises:
            FileFormatNotSupportedError: On a missing or malformed header, a short
                record, or a file with no face-forming cell.
            ValueError: If a cell references an undefined node label, or two nodes
                share a label.
        """
        n_nodes, n_cells = self._read_header()
        node_ids, node_x, node_y, node_z = self._read_nodes(n_nodes)
        cell_vertices = self._read_cells(n_cells)
        if not cell_vertices:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable AVS-UCD mesh: it has no tri/quad/tet/"
                f"pyr/prism/hex cell to build a face from"
            )
        node_map = native_id_map(node_ids)
        faces = self._fill_pad(cell_vertices, node_map)
        z = np.array(node_z, dtype=np.float64)
        return ParsedMesh(
            node_x=np.array(node_x, dtype=np.float64),
            node_y=np.array(node_y, dtype=np.float64),
            faces=faces,
            fill_value=_PAD,
            # The connectivity is already 0-based (resolved through the id map); the
            # format labels from 1, which is recorded for a writer.
            start_index=0,
            original_start_index=1,
            node_z=z if np.any(z != 0.0) else None,
            crs=None,
            variables={},
            global_attributes={"source": "AVS-UCD"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_header(self) -> tuple[int, int]:
        """Read the five-integer header, returning the node and cell counts."""
        try:
            keyword, fields = next(self.records)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is empty: an AVS-UCD file starts with a five-integer "
                f"header"
            ) from None
        tokens = [keyword, *fields]
        if len(tokens) != _HEADER_FIELDS or not all(_is_int(t) for t in tokens):
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an AVS-UCD file: the header line {tokens!r} is "
                f"not five integers"
            )
        return int(tokens[0]), int(tokens[1])

    def _read_nodes(
        self, n_nodes: int
    ) -> tuple[list[int], list[float], list[float], list[float]]:
        """Read ``n_nodes`` ``node_id x y z`` rows."""
        ids: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for _ in range(n_nodes):
            keyword, fields = self._next_record("a node")
            if len(fields) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short node record {[keyword, *fields]!r}: "
                    f"expected id, x, y and z"
                )
            ids.append(_as_int(keyword, "node", self.path))
            xs.append(_as_float(fields[0], "node", self.path))
            ys.append(_as_float(fields[1], "node", self.path))
            zs.append(_as_float(fields[2], "node", self.path))
        return ids, xs, ys, zs

    def _read_cells(self, n_cells: int) -> list[list[int]]:
        """Read ``n_cells`` cell rows, keeping the node list of each face-forming one.

        A ``cell_id material_id cell_type v1 ...`` row contributes its node list when
        ``cell_type`` makes a face; ``pt`` and ``line`` cells are dropped.

        Raises:
            FileFormatNotSupportedError: If a cell row is short of a type and one node,
                or names a cell type this reader does not recognise.
        """
        vertices: list[list[int]] = []
        for _ in range(n_cells):
            keyword, fields = self._next_record("a cell")
            # keyword is the cell id; fields[0] material id, fields[1] cell type.
            if len(fields) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short cell record {[keyword, *fields]!r}: "
                    f"expected id, material, type and at least one node"
                )
            cell_type = fields[1].lower()
            if cell_type in _SKIP_CELL_TYPES:
                continue
            if cell_type not in _FACE_CELL_TYPES:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has an unknown AVS-UCD cell type {cell_type!r}"
                )
            vertices.append([_as_int(value, "cell", self.path) for value in fields[2:]])
        return vertices

    def _fill_pad(
        self, cell_vertices: list[list[int]], node_map: dict[int, int]
    ) -> np.ndarray:
        """Map cell node labels to indices and pad short rows to the widest cell."""
        max_nodes = max(len(vertices) for vertices in cell_vertices)
        faces = np.full((len(cell_vertices), max_nodes), _PAD, dtype=np.int64)
        for row, vertices in enumerate(cell_vertices):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, self.path)
        return faces

    def _next_record(self, what: str) -> tuple[str, list[str]]:
        """Pull the next significant record, or report the truncation."""
        try:
            return next(self.records)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends before {what} the header promised"
            ) from None


def _is_int(text: str) -> bool:
    """Report whether ``text`` parses as an integer, without raising."""
    try:
        int(text)
    except ValueError:
        return False
    return True


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Map one native node label to its 0-based index, or report the dangling id."""
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} references node {native}, which no node record defines"
        ) from None


def _as_int(text: str, what: str, path: str) -> int:
    """Parse an integer field, naming the record kind and file on failure."""
    try:
        return int(text)
    except ValueError:
        raise FileFormatNotSupportedError(
            f"{path!r} has {what} record with a non-integer value {text!r}"
        ) from None


def _as_float(text: str, what: str, path: str) -> float:
    """Parse a float field, naming the record kind and file on failure."""
    try:
        return float(text)
    except ValueError:
        raise FileFormatNotSupportedError(
            f"{path!r} has {what} record with a non-numeric value {text!r}"
        ) from None
