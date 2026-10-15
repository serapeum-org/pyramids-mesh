"""SMS-family drivers: 2DM (mesh frame) and DAT (results-only).

2DM is the reference driver every later one copies (task D1). DAT carries values but
no topology, so it declares ``needs_mesh`` and implements ``read_data`` rather than
``read`` (task D2). See ``planning/design.md`` section 20.1.

The 2DM structure here was verified against MDAL's ``mdal_2dm.cpp``: a plain-text file
whose first significant line is ``MESH2D``, then one record per line.

``topology_info`` is ``None`` for this driver -- a 2DM file has no NetCDF variables to
name, and nothing is synthesised that ``to_file`` would not write.
"""

from __future__ import annotations

import warnings

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._text import iter_records, native_id_map

_HEADER = "MESH2D"

# The node count comes from the keyword's second character (MDAL reads it as
# ``line[1]`` and asserts 3, 4 or 6), but the full keyword is matched instead: E2L and
# E3L carry a digit too and are line elements, not faces.
_FACE_NODE_COUNTS = {"E3T": 3, "E4Q": 4, "E6T": 6}

# MDAL refuses the whole file on these (``Err_UnsupportedElement``) rather than guess a
# mapping, and so does this driver. E3L is a 1D element MDAL also rejects.
_UNSUPPORTED_ELEMENTS = ("E3L", "E8Q", "E9Q")

_LINE_ELEMENT = "E2L"
_NODE = "ND"
_MATERIALS_TAG = "NUM_MATERIALS_PER_ELEM"

# Kept in step with ``iter_records``' default: the header sniff and the parser must
# skip the same lines, or one accepts a file the other refuses.
_COMMENT = "#"

# ``can_open`` must not read the whole file, so the header sniff looks at a bounded
# prefix. A binary file decodes to replacement characters rather than raising.
_SNIFF_BYTES = 4096

# The smallest valid 2DM: the unit square split into two counter-clockwise triangles.
_MINIMAL_FIXTURE = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
E3T 1 1 2 3 1
E3T 2 1 3 4 1
"""


class TwoDmDriver(BaseDriver):
    """Reader for the SMS/TUFLOW 2DM mesh-frame format.

    Reads nodes, triangular/quadrilateral/six-node faces and 1D line elements. Node
    elevation becomes a ``node_z`` variable and the material id a face-located
    ``material`` variable, since the pyramids mesh model is planar and carries no
    per-node elevation coordinate.

    Documented limitations:
        - ``E3L``, ``E8Q`` and ``E9Q`` are refused outright, as MDAL refuses them;
          treating a mid-side or centre node as a corner would silently distort the
          element.
        - ``E6T`` is read as a real six-node face, not demoted to its corner triangle.
        - A file holding both faces and ``E2L`` line elements returns the faces only,
          with a warning: a mixed-dimension file is reached through ``Network1d``.
        - 2DM carries no CRS, so ``crs`` is always ``None``; it is never guessed from
          the coordinate magnitudes.
    """

    name = "2dm"
    long_name = "SMS/TUFLOW 2DM mesh"
    suffixes = (".2dm",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first significant line is the ``MESH2D`` header.

        Reads a bounded prefix and decodes it leniently, so a binary or empty file
        answers ``False`` instead of raising.

        Args:
            path: The file to inspect.

        Blank and comment lines are skipped, exactly as
        :func:`~pyramids_mesh._text.iter_records` skips them while parsing. The two must
        agree: a sniff stricter than the parser would decline a file this driver can
        read, and dispatch would then hand it to nobody.

        Returns:
            bool: ``True`` when the first significant line starts with ``MESH2D``.
        """
        with open(path, "rb") as handle:
            head = handle.read(_SNIFF_BYTES)
        for line in head.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(_COMMENT):
                return stripped.upper().startswith(_HEADER)
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a 2DM file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Node and element ids are collected as the file gives them and resolved to
        contiguous indices through :func:`~pyramids_mesh._text.native_id_map` in a
        second pass, exactly as MDAL does: 2DM ids may have gaps and may be out of
        order, so ``id == index + 1`` would mis-wire every face of a renumbered file.

        Args:
            path: The 2DM file to read.
            lazy: Ignored -- a 2DM file is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, faces or edges, and the material/elevation variables.

        Raises:
            FileFormatNotSupportedError: If the ``MESH2D`` header is missing, an
                unsupported element type appears, or the file declares no elements.
            ValueError: If an element references a node id the file never defined.
        """
        scan = _Scan()
        scan.run(path)
        return scan.build(path)


def build_minimal_fixture(path: str) -> None:
    """Write the smallest valid 2DM file, for the cross-driver invariant harness.

    Args:
        path: Where to write the file.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_MINIMAL_FIXTURE)


class _Scan:
    """One 2DM file's records, collected in file order then resolved to indices."""

    def __init__(self) -> None:
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.face_ids: list[int] = []
        self.face_vertices: list[list[int]] = []
        self.face_extras: list[list[str]] = []
        self.edge_vertices: list[list[int]] = []
        self.materials_per_element = 1

    def run(self, path: str) -> None:
        """Stream the file once, collecting records without resolving any ids.

        Args:
            path: The 2DM file to read.

        Raises:
            FileFormatNotSupportedError: On a missing header or an unsupported element.
        """
        header_seen = False
        for keyword, fields in iter_records(path, comment=_COMMENT):
            upper = keyword.upper()
            if not header_seen:
                if upper != _HEADER:
                    raise FileFormatNotSupportedError(
                        f"{path!r} is not a 2DM file: expected a {_HEADER} header, "
                        f"found {keyword!r}"
                    )
                header_seen = True
                continue
            self._consume(upper, fields, path)
        if not header_seen:
            raise FileFormatNotSupportedError(
                f"{path!r} is empty: a 2DM file starts with a {_HEADER} header"
            )

    def _consume(self, upper: str, fields: list[str], path: str) -> None:
        """Dispatch one record onto the matching collector."""
        if upper in _UNSUPPORTED_ELEMENTS:
            raise FileFormatNotSupportedError(
                f"{path!r} uses the {upper} element, which is not supported; MDAL "
                f"refuses such a file rather than guessing a node mapping"
            )
        if upper == _MATERIALS_TAG:
            _require_fields(upper, fields, 1, path)
            self.materials_per_element = max(1, int(fields[0]))
        elif upper == _NODE:
            self._add_node(fields, path)
        elif upper in _FACE_NODE_COUNTS:
            self._add_face(upper, fields, path)
        elif upper == _LINE_ELEMENT:
            _require_fields(upper, fields, 3, path)
            self.edge_vertices.append([int(value) for value in fields[1:3]])

    def _add_node(self, fields: list[str], path: str) -> None:
        """Collect one ``ND <id> <x> <y> <z>`` record.

        Args:
            fields: The record's fields, without the keyword.
            path: The source path, for the error message.

        Raises:
            FileFormatNotSupportedError: If the record is short of an id and both
                coordinates.
        """
        _require_fields(_NODE, fields, 3, path)
        self.node_ids.append(int(fields[0]))
        self.node_x.append(float(fields[1]))
        self.node_y.append(float(fields[2]))
        self.node_z.append(float(fields[3]) if len(fields) > 3 else 0.0)

    def _add_face(self, upper: str, fields: list[str], path: str) -> None:
        """Collect one face record, keeping its trailing columns unsliced.

        The material and aux columns are stored verbatim because
        ``NUM_MATERIALS_PER_ELEM`` may appear after the elements it applies to; the
        split happens once the whole file has been read. Fields are only ever indexed
        from the start, since the column count varies with the aux columns.

        Args:
            upper: The element keyword, upper-cased.
            fields: The record's fields, without the keyword.
            path: The source path, for the error message.

        Raises:
            FileFormatNotSupportedError: If the record carries fewer vertex ids than
                the element type declares.
        """
        count = _FACE_NODE_COUNTS[upper]
        _require_fields(upper, fields, 1 + count, path)
        self.face_ids.append(int(fields[0]))
        self.face_vertices.append([int(value) for value in fields[1 : 1 + count]])
        self.face_extras.append(fields[1 + count :])

    def build(self, path: str) -> ParsedMesh:
        """Resolve collected ids to indices and assemble the parsed mesh.

        Args:
            path: The source path, for error messages.

        Returns:
            ParsedMesh: The assembled mesh or 1D network.

        Raises:
            FileFormatNotSupportedError: If the file declares no elements at all.
            ValueError: If an element references an undefined node id.
        """
        if not self.face_vertices and not self.edge_vertices:
            raise FileFormatNotSupportedError(
                f"{path!r} declares no elements: no E3T/E4Q/E6T face and no E2L line"
            )
        node_map = native_id_map(self.node_ids)
        native_id_map(self.face_ids)  # rejects duplicate element ids
        faces = self._resolve_faces(node_map, path)
        edges = self._resolve_edges(node_map, path)
        if faces is not None and edges is not None:
            warnings.warn(
                f"{path!r} holds both faces and E2L line elements; only the faces "
                f"were returned. Read the 1D elements with Network1d.read_file",
                stacklevel=2,
            )
            edges = None
        return self._assemble(path, faces, edges)

    def _resolve_faces(self, node_map: dict[int, int], path: str) -> np.ndarray | None:
        """Map face vertex ids to 0-based indices, padding short rows with ``-1``."""
        if not self.face_vertices:
            return None
        max_nodes = max(len(vertices) for vertices in self.face_vertices)
        faces = np.full((len(self.face_vertices), max_nodes), -1, dtype=np.int64)
        for row, vertices in enumerate(self.face_vertices):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, path)
        return faces

    def _resolve_edges(self, node_map: dict[int, int], path: str) -> np.ndarray | None:
        """Map E2L vertex ids to 0-based indices."""
        if not self.edge_vertices:
            return None
        edges = np.empty((len(self.edge_vertices), 2), dtype=np.int64)
        for row, vertices in enumerate(self.edge_vertices):
            for column, native in enumerate(vertices):
                edges[row, column] = _index_of(native, node_map, path)
        return edges

    def _assemble(
        self, path: str, faces: np.ndarray | None, edges: np.ndarray | None
    ) -> ParsedMesh:
        """Build the :class:`ParsedMesh` from resolved connectivity."""
        variables: dict[str, ParsedVariable] = {}
        if faces is not None:
            variables.update(self._material_variables(len(faces)))
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            edges=edges,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map, so
            # there is nothing for the validator to subtract. The file's own 1-based
            # numbering is recorded separately so a writer can restore it.
            start_index=0,
            original_start_index=1,
            node_z=np.array(self.node_z, dtype=np.float64) if self.node_z else None,
            crs=None,
            variables=variables,
            global_attributes={"Conventions": "UGRID-1.0", "source": "SMS 2DM"},
            topology_dimension=1 if faces is None else 2,
            mesh_name="mesh2d",
        )

    def _material_variables(self, n_face: int) -> dict[str, ParsedVariable]:
        """Split the trailing columns into one face variable per material column."""
        variables: dict[str, ParsedVariable] = {}
        for index in range(self.materials_per_element):
            column = [
                float(extras[index]) if len(extras) > index else 0.0
                for extras in self.face_extras
            ]
            name = "material" if index == 0 else f"material_{index + 1}"
            variables[name] = ParsedVariable(
                location="face",
                shape=(n_face,),
                values=np.array(column, dtype=np.float64),
            )
        return variables


def _require_fields(keyword: str, fields: list[str], needed: int, path: str) -> None:
    """Check a record carries at least the fields its keyword requires.

    Without this a truncated line fails with a bare ``IndexError`` deep in the parser,
    which says nothing about which record of which file is malformed.

    Args:
        keyword: The record's keyword, for the error message.
        fields: The record's fields, without the keyword.
        needed: The minimum number of fields the keyword requires.
        path: The source path, for the error message.

    Raises:
        FileFormatNotSupportedError: If fewer than ``needed`` fields are present.
    """
    if len(fields) < needed:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected at least {needed} "
            f"fields after the keyword, found {len(fields)}"
        )


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Return the 0-based index of a native node id.

    Args:
        native: The node id as the file wrote it.
        node_map: The id-to-index mapping built from the ``ND`` records.
        path: The source path, for the error message.

    Returns:
        int: The node's 0-based index.

    Raises:
        ValueError: If the file never defined that node id.
    """
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} references node id {native}, which no ND record defines"
        ) from None
