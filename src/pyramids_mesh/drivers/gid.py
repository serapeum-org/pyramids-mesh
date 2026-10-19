"""GiD / CIMNE ASCII mesh reader -- the format the Iber 2D model exports.

GiD (CIMNE) writes a plain-text mesh as one or more blocks, each opening with a
``MESH`` header that names the spatial dimension, the element type and the nodes per
element::

    MESH [name] dimension {2|3} ElemType {Triangle|Quadrilateral|Tetrahedra|...} Nnode N
    Coordinates
        node_id x y [z]
        ...
    end coordinates
    Elements
        el_id n1 n2 ... nN [material]
        ...
    end elements

Keywords are case-insensitive and ``#`` starts a comment. Iber -- the free 2D
hydraulic model built on GiD/CIMNE -- exports its mesh in exactly this layout, which is
why the reader exists (issue #68).

Node ids are labels that may be gapped or out of file order, so connectivity is resolved
through :func:`~pyramids_mesh._text.native_id_map` rather than by assuming
``id == index + 1``, exactly as the 2DM reference reader does. Surface element types
(``Triangle``/``Quadrilateral``) become faces; a higher-order element lists its corners
first and its mid-edge nodes after, and only the corners form the face polygon, mirroring
the CGNS and MED readers. ``Line``/``Point`` scaffolding is dropped; a volumetric
``Tetrahedra``/``Hexahedra`` block maps its node list to a face row verbatim, as the
Exodus reader does, rather than returning an empty mesh.

The ``.msh`` suffix is shared with the Gmsh reader (:mod:`pyramids_mesh.drivers.gmsh`),
which opens with ``$MeshFormat``; ``can_open`` requires the GiD signature (a first
significant line starting with ``MESH`` and carrying ``dimension`` and ``ElemType``), so
the two never both claim a file.

GiD carries engineering coordinates with no datum, so ``crs`` is always ``None`` and is
never guessed. See ``planning/design.md``; this driver follows the 2DM reference shape.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

from typing import Iterator

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records, native_id_map

# GiD uses '#' for comments, the default iter_records marker; named so the sniff and the
# parser skip the same lines.
_COMMENT = "#"

# Block and header keywords, matched lower-cased because GiD keywords are case-insensitive.
_MESH = "mesh"
_COORDINATES = "coordinates"
_ELEMENTS = "elements"
_END = "end"
_DIMENSION = "dimension"
_ELEMTYPE = "elemtype"
_NNODE = "nnode"

# Surface element types and their corner-node counts. A higher-order element (a six-node
# quadratic Triangle, an eight-node Quadrilateral) lists its corners first then its
# mid-edge nodes; only the corners form the face polygon, so only they are kept -- the
# same stance the CGNS and MED readers take on TRI_6/QU8.
_SURFACE_CORNERS = {"triangle": 3, "quadrilateral": 4}

# Volumetric element types. A genuine 3D mesh is not modelled yet (design section 21.1),
# so a cell's node list is read verbatim as a face row rather than expanded into its
# bounding faces, exactly as the Exodus reader does, rather than returning an empty mesh.
_VOLUME_TYPES = frozenset({"tetrahedra", "hexahedra", "prism", "pyramid"})


def _as_int(value: str, context: str, path: str) -> int:
    """Parse one integer field, naming the file and record on failure.

    Args:
        value: The raw token to convert.
        context: What the token was expected to be, for the error message.
        path: The source path, for the error message.

    Returns:
        int: The parsed integer.

    Raises:
        FileFormatNotSupportedError: If the token is not an integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {context}: expected an integer, found {value!r}"
        ) from None


def _as_float(value: str, context: str, path: str) -> float:
    """Parse one float field, naming the file and record on failure.

    Args:
        value: The raw token to convert.
        context: What the token was expected to be, for the error message.
        path: The source path, for the error message.

    Returns:
        float: The parsed float.

    Raises:
        FileFormatNotSupportedError: If the token is not a number.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {context}: expected a number, found {value!r}"
        ) from None


class GidDriver(BaseDriver):
    """Reader for the GiD / CIMNE ASCII mesh format (Iber 2D export).

    Reads node coordinates and the surface connectivity from one or more ``MESH`` blocks,
    mapping ``Triangle`` and ``Quadrilateral`` elements to ``face_node_connectivity``; a
    file mixing the two pads its triangles to the quad width with ``-1``. Node ids are
    labels resolved through an id map, so a gapped or out-of-order file wires correctly,
    and the connectivity arrives 0-based with the GiD 1-based base recorded for a writer.
    Node elevation becomes a ``node_z`` variable only when some node carries a non-zero z,
    since a planar mesh gains nothing from an all-zero column.

    Documented limitations:
        - **Higher-order surface elements are reduced to their corners.** A six-node
          quadratic ``Triangle`` (``Nnode 6``) or an eight-node ``Quadrilateral`` lists
          its corners first and its mid-edge nodes after; only the corners (3 for a
          triangle, 4 for a quadrilateral) form the face polygon, so the mid-edge nodes
          are dropped from the connectivity -- the same stance the CGNS and MED readers
          take. The mid-edge nodes remain in the node table, simply unreferenced.
        - **``Line`` and ``Point`` elements are dropped.** They are lower-dimensional
          scaffolding, not surface faces; folding them into the face table would mislabel
          them.
        - **A volumetric ``Tetrahedra``/``Hexahedra`` block becomes faces verbatim.** Its
          node list is read as a polygon rather than expanded into bounding faces -- a
          genuine 3D mesh is not modelled yet (design section 21.1) -- exactly as the
          Exodus reader does, and winding is judged in the ``xy`` plane like any other
          face. A file whose only elements are ``Line``/``Point`` (or an unrecognised
          type) is refused for having no surface.
        - **Node ids may be gapped or out of order.** They are resolved through a
          label->index map, never by assuming ``id == index + 1``. A node id repeated
          across blocks -- GiD may re-list the coordinate block per ``MESH`` -- keeps its
          first definition.
        - **Only the mesh is read.** Any per-element material column trailing an element
          record is ignored; GiD result (``.res``) data is a separate format.
        - The ``.msh`` suffix is shared with the Gmsh reader. ``can_open`` requires the
          GiD signature (a first significant line starting with ``MESH`` carrying
          ``dimension`` and ``ElemType``), so a Gmsh ``$MeshFormat`` file is declined and
          the two never both claim a file.
        - GiD carries engineering coordinates with no datum, so ``crs`` is always
          ``None``; it is never guessed from the coordinate magnitudes.

    Examples:
        - A tiny single-triangle mesh reads to one face and three nodes, with the 1-based
          node ids arriving 0-based:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.gid import GidDriver
            >>> text = '''MESH dimension 2 ElemType Triangle Nnode 3
            ... Coordinates
            ... 1 0.0 0.0
            ... 2 1.0 0.0
            ... 3 0.0 1.0
            ... end coordinates
            ... Elements
            ... 1 1 2 3
            ... end elements
            ... '''
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "mesh.msh")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write(text)
            ...     dataset = GidDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (3, 1)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2]]

        - Dispatch checks the suffix and sniffs the ``MESH`` header, so a Gmsh file under
          the shared ``.msh`` suffix is declined:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.gid import GidDriver
            >>> gmsh_text = '''$MeshFormat
            ... 2.2 0 8
            ... $EndMeshFormat
            ... '''
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "gmsh.msh")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write(gmsh_text)
            ...     GidDriver().can_open(path)
            False
    """

    name = "gid"
    long_name = "GiD / CIMNE ASCII mesh"
    suffixes = (".msh",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens with a GiD ``MESH`` header, never raising.

        The first significant line must start with ``MESH`` (case-insensitive) and carry
        both a ``dimension`` and an ``ElemType`` keyword. This is the GiD signature, and
        it is what declines the other ``.msh`` format: a Gmsh file opens with
        ``$MeshFormat``, so its first keyword is never ``MESH`` and it is declined here,
        just as this file's ``MESH`` header makes the Gmsh sniff decline it. A binary or
        empty file answers ``False`` rather than raising, so ``can_open`` stays within its
        never-raise contract.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first significant line is a GiD ``MESH`` header.
        """
        records = iter_records(path, comment=_COMMENT)
        try:
            keyword, fields = next(records)
        except StopIteration:
            return False
        if keyword.lower() != _MESH:
            return False
        lowered = [field.lower() for field in fields]
        return _DIMENSION in lowered and _ELEMTYPE in lowered

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a GiD file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The ``.msh`` file to read.
            lazy: Ignored -- a GiD ASCII file is plain text and read eagerly; accepted to
                match the driver protocol.

        Returns:
            ParsedMesh: Nodes, the surface face connectivity and, when some node carries a
                non-zero z, a ``node_z`` elevation variable.

        Raises:
            FileFormatNotSupportedError: If the file carries no ``MESH`` header, a header
                omits ``ElemType``/``Nnode``, a record is short or non-numeric, a
                ``Coordinates``/``Elements`` block is never closed, or the file declares
                no surface or volumetric element.
            ValueError: If an element references an undefined node id, or two nodes share
                an id within one coordinate block.
        """
        return _GidReader(path).build()


class _GidReader:
    """One GiD file: stream its records block by block and assemble the mesh."""

    def __init__(self, path: str) -> None:
        """Prepare to read ``path``, collecting records before resolving any id.

        Args:
            path: The ``.msh`` file to read.
        """
        self.path = path
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self._seen_nodes: set[int] = set()
        self.face_rows: list[list[int]] = []
        self.header_seen = False
        # The node ids to keep from each element of the current block: a corner count for
        # a surface type, the full Nnode for a volume type, or None to drop the block.
        self._keep: int | None = None

    def build(self) -> ParsedMesh:
        """Read the whole file and resolve it into a parsed mesh.

        Returns:
            ParsedMesh: The assembled surface mesh.

        Raises:
            FileFormatNotSupportedError: On a missing ``MESH`` header, a malformed header
                or record, an unclosed block, or a file with no surface or volumetric
                element.
            ValueError: If an element references an undefined node id, or two nodes share
                an id.
        """
        records = iter_records(self.path, comment=_COMMENT)
        for keyword, fields in records:
            lower = keyword.lower()
            if lower == _MESH:
                self._begin_mesh(fields)
            elif lower == _COORDINATES:
                self._read_coordinates(records)
            elif lower == _ELEMENTS:
                self._read_elements(records)
            # Any other top-level keyword (a stray metadata line) is ignored.
        return self._assemble()

    def _begin_mesh(self, fields: list[str]) -> None:
        """Parse a ``MESH ... dimension D ElemType T Nnode N`` header line.

        The spatial dimension, the element type and the nodes-per-element are found by
        keyword rather than by position, so an optional (possibly quoted) mesh name
        between ``MESH`` and ``dimension`` does not throw the reader off. The element type
        sets how many node ids to keep from each element of the block that follows.

        Args:
            fields: The header line's tokens after the ``MESH`` keyword.

        Raises:
            FileFormatNotSupportedError: If the header omits ``ElemType`` or ``Nnode``, or
                ``Nnode`` is not an integer.
        """
        lowered = [field.lower() for field in fields]
        elem_type = self._value_after(fields, lowered, _ELEMTYPE).lower()
        nnode = _as_int(self._value_after(fields, lowered, _NNODE), "Nnode", self.path)
        self.header_seen = True
        if elem_type in _SURFACE_CORNERS:
            self._keep = _SURFACE_CORNERS[elem_type]
        elif elem_type in _VOLUME_TYPES:
            self._keep = nnode
        else:
            # Line, Point or any other lower-dimensional type: dropped, so the block's
            # elements never reach the face table.
            self._keep = None

    def _value_after(self, fields: list[str], lowered: list[str], key: str) -> str:
        """Return the token following a header keyword, matched case-insensitively.

        Args:
            fields: The header tokens, in original case.
            lowered: The same tokens lower-cased, so the keyword match is case-insensitive.
            key: The lower-cased keyword to find (e.g. ``"elemtype"``).

        Returns:
            str: The token immediately after ``key``.

        Raises:
            FileFormatNotSupportedError: If the keyword is absent or has no value after it.
        """
        try:
            index = lowered.index(key)
        except ValueError:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a MESH header with no {key!r}: {fields!r}"
            ) from None
        if index + 1 >= len(fields):
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a MESH header whose {key!r} has no value: {fields!r}"
            )
        return fields[index + 1]

    def _read_coordinates(self, records: Iterator[tuple[str, list[str]]]) -> None:
        """Read a ``Coordinates`` block up to its ``end coordinates`` line.

        Each record is ``node_id x y [z]``; the id is the record keyword and the
        coordinates its fields. The block must be closed, or the file is refused rather
        than silently reading a truncated mesh.

        Args:
            records: The shared record iterator, consumed up to and including the ``end``.

        Raises:
            FileFormatNotSupportedError: If a record is short or non-numeric, or the block
                runs to end of file without an ``end coordinates`` line.
        """
        for keyword, fields in records:
            if keyword.lower() == _END:
                return
            self._add_node(keyword, fields)
        raise FileFormatNotSupportedError(
            f"{self.path!r} ends inside a Coordinates block with no 'end coordinates'"
        )

    def _add_node(self, id_token: str, fields: list[str]) -> None:
        """Collect one ``node_id x y [z]`` coordinate record.

        A node id already seen is skipped rather than re-added: GiD may re-list the whole
        coordinate block in each ``MESH``, and the first definition is kept.

        Args:
            id_token: The record keyword, the node id.
            fields: The coordinate fields after the id.

        Raises:
            FileFormatNotSupportedError: If the record carries fewer than an id and two
                coordinates, or a coordinate is not a number.
        """
        node_id = _as_int(id_token, "node id", self.path)
        if node_id in self._seen_nodes:
            return
        if len(fields) < 2:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short coordinate record for node {node_id}: "
                f"expected 'id x y [z]', found {fields!r}"
            )
        self._seen_nodes.add(node_id)
        self.node_ids.append(node_id)
        self.node_x.append(_as_float(fields[0], "node x", self.path))
        self.node_y.append(_as_float(fields[1], "node y", self.path))
        self.node_z.append(
            _as_float(fields[2], "node z", self.path) if len(fields) > 2 else 0.0
        )

    def _read_elements(self, records: Iterator[tuple[str, list[str]]]) -> None:
        """Read an ``Elements`` block up to its ``end elements`` line.

        Each record is ``el_id n1 n2 ... nN [material]``; the element id is the keyword
        and the node ids its leading fields. How many node ids are kept was fixed by the
        block's ``MESH`` header: a corner count for a surface type, the full node list for
        a volume type, or none for a dropped type (``Line``/``Point``). A trailing
        material column is ignored. The block must be closed.

        Args:
            records: The shared record iterator, consumed up to and including the ``end``.

        Raises:
            FileFormatNotSupportedError: If a record carries fewer node ids than the block
                keeps, a node id is non-numeric, or the block runs to end of file without
                an ``end elements`` line.
        """
        for keyword, fields in records:
            if keyword.lower() == _END:
                return
            self._add_element(keyword, fields)
        raise FileFormatNotSupportedError(
            f"{self.path!r} ends inside an Elements block with no 'end elements'"
        )

    def _add_element(self, el_token: str, fields: list[str]) -> None:
        """Keep one element's node ids, reduced to what the block's type keeps.

        Args:
            el_token: The record keyword, the element id (unused -- nothing references an
                element by id, so it is read only as the record's leading token).
            fields: The node ids, followed by an optional material column.

        Raises:
            FileFormatNotSupportedError: If the record carries fewer node ids than the
                block's element type requires, or a node id is non-numeric.
        """
        if self._keep is None:
            # A Line/Point (or unrecognised) block: its elements are not faces.
            return
        if len(fields) < self._keep:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short element record (id {el_token!r}): expected "
                f"{self._keep} node id(s), found {len(fields)}"
            )
        self.face_rows.append(
            [
                _as_int(fields[index], "element node id", self.path)
                for index in range(self._keep)
            ]
        )

    def _assemble(self) -> ParsedMesh:
        """Resolve node ids to indices and build the parsed surface mesh.

        Returns:
            ParsedMesh: The assembled mesh.

        Raises:
            FileFormatNotSupportedError: If the file carries no ``MESH`` header, or no
                surface or volumetric element to build a face table from.
            ValueError: If two nodes share an id, or an element references an undefined
                node id.
        """
        if not self.header_seen:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a GiD file: no MESH header"
            )
        if not self.face_rows:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no surface or volumetric element, so it has no "
                f"faces to build (only Line/Point elements, if any)"
            )
        node_map = native_id_map(self.node_ids)
        faces = self._resolve_faces(node_map)
        node_z = np.array(self.node_z, dtype=np.float64)
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map, so the
            # validator has nothing to subtract. GiD's 1-based ids are recorded for a
            # writer.
            start_index=0,
            original_start_index=1,
            node_z=node_z if np.any(node_z != 0.0) else None,
            crs=None,
            global_attributes={"source": "GiD / CIMNE"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _resolve_faces(self, node_map: dict[int, int]) -> np.ndarray:
        """Map face node ids to 0-based indices, padding short rows with ``-1``.

        Args:
            node_map: The node-id to 0-based-index map.

        Returns:
            np.ndarray: The ``(n_face, max_nodes)`` connectivity, fill-padded.

        Raises:
            ValueError: If a face references a node id no coordinate record defined.
        """
        max_nodes = max(len(row) for row in self.face_rows)
        faces = np.full((len(self.face_rows), max_nodes), -1, dtype=np.int64)
        for row, ids in enumerate(self.face_rows):
            for column, native in enumerate(ids):
                if native not in node_map:
                    raise ValueError(
                        f"{self.path!r} has an element referencing node id {native}, "
                        f"which no coordinate record defines"
                    )
                faces[row, column] = node_map[native]
        return faces
