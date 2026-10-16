"""Geometry interchange drivers: PLY and XMS TIN (task D5).

Not solver formats -- no time axis, elevation per node. pyramids' mesh model is planar,
so z becomes a node-located variable and a round trip is not geometrically faithful
(design section 6). These are hand-parsed: the ``meshio`` dependency the original plan
reached for was dropped, and PLY and the XMS TIN are simple enough to read directly with
the package's own helpers.

Esri TIN -- the third format this task named -- is an undocumented *directory* of binary
files, reverse-engineered by MDAL. Without a real ArcGIS sample to test against it cannot
be implemented honestly, so it is deliberately not here; see ``planning/design.md``
section 20 and the task brief.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import struct
from typing import Iterator

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._text import iter_records

# -- PLY --------------------------------------------------------------------------

_PLY_MAGIC = b"ply"
_PLY_END_HEADER = "end_header"
_PLY_FORMATS = {
    "ascii": None,
    "binary_little_endian": "<",
    "binary_big_endian": ">",
}

# PLY scalar type names, including the explicit-width aliases, to a numpy type code.
_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

# The vertex coordinate properties, by name; anything else on a vertex becomes a
# node-located variable rather than being dropped.
_PLY_COORDS = ("x", "y", "z")

# The integer type codes a face list's count may use; a float count type cannot size a
# record, so it is refused rather than reaching _struct_char as a KeyError.
_INTEGER_TYPES = frozenset({"i1", "u1", "i2", "u2", "i4", "u4"})

_PLY_MINIMAL = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 2.0
1.0 0.0 2.5
1.0 1.0 3.0
0.0 1.0 3.5
3 0 1 2
3 0 2 3
"""

# -- XMS TIN ----------------------------------------------------------------------

_TIN_HEADER = "TIN"
_TIN_BEGIN = "BEGT"
_TIN_END = "ENDT"
_TIN_VERTICES = "VERT"
_TIN_TRIANGLES = "TRI"
_TIN_NAME = "TNAM"

_TIN_MINIMAL = """TIN
BEGT
TNAM minimal
MAT 1
VERT 4
0.0 0.0 2.0 0
1.0 0.0 2.5 0
1.0 1.0 3.0 0
0.0 1.0 3.5 0
TRI 2
1 2 3
1 3 4
ENDT
"""


class PlyDriver(BaseDriver):
    """Reader for the Stanford PLY polygon format.

    Reads ASCII and both binary byte orders. A PLY is a geometry format with elevation
    per vertex and no time axis, so ``z`` becomes a ``node_z`` variable and any other
    vertex property (colours, normals) becomes its own node-located variable rather than
    being dropped.

    Documented limitations:
        - **A round trip is not geometrically faithful.** ``z`` survives as node data, but
          pyramids' mesh model is planar, so a consumer reading the geometry sees the
          ``(x, y)`` projection. This is inherent to carrying a 3D surface in a 2D model
          (design section 6).
        - **The whole file is read eagerly.** A PLY from a 3D scanner can be large; there
          is no record index to seek, so every vertex property and face is read at once.
          The driver does not declare ``lazy_data``.
        - Faces may be triangles, quads or mixed; short rows are padded to the widest with
          ``-1``. A PLY edge or point element (a face of fewer than three vertices) is
          refused, since it is not a surface.
        - PLY carries no CRS and no units, so ``crs`` is ``None``.
        - Face indices are 0-based in PLY, and stay 0-based.

    Examples:
        - Write a small ASCII PLY and read it back. The vertex z becomes ``node_z`` and
          the faces come through 0-based:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.interchange import (
            ...     PlyDriver,
            ...     build_minimal_fixture_ply,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "surface.ply")
            ...     build_minimal_fixture_ply(path)
            ...     dataset = PlyDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.get("node_z").data.tolist()
            [2.0, 2.5, 3.0, 3.5]
    """

    name = "ply"
    long_name = "Stanford PLY"
    suffixes = (".ply",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens with the ``ply`` magic line."""
        with open(path, "rb") as handle:
            first = handle.readline().strip()
        return first == _PLY_MAGIC

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a PLY file into a :class:`ParsedMesh`.

        Args:
            path: The PLY file to read.
            lazy: Ignored -- a PLY has no record index to seek, so it is read eagerly.

        Returns:
            ParsedMesh: Nodes, faces, ``node_z`` and any extra vertex-property variables.

        Raises:
            FileFormatNotSupportedError: On a missing or malformed header, an unknown
                property type, a face of fewer than three vertices, or a truncated body.
        """
        return _PlyReader(path).build()


def build_minimal_fixture_ply(path: str) -> None:
    """Write the smallest valid ASCII PLY, for the cross-driver invariant harness.

    The unit square split into two counter-clockwise triangles, each vertex carrying an
    elevation. Named per driver because this module holds more than one.

    Args:
        path: Where to write the file.

    Examples:
        - The fixture is four vertices and two triangular faces:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.interchange import build_minimal_fixture_ply
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.ply")
            ...     build_minimal_fixture_ply(path)
            ...     with open(path, encoding="utf-8") as handle:
            ...         lines = handle.read().splitlines()
            >>> lines[0]
            'ply'
            >>> lines[2]
            'element vertex 4'
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_PLY_MINIMAL)


class XmsTinDriver(BaseDriver):
    """Reader for the XMS / SMS TIN text format.

    A triangulated irregular network: a header, a ``VERT`` block of ``x y z`` vertices,
    and a ``TRI`` block of 1-based triangles. Like PLY it is a geometry format, so ``z``
    becomes a ``node_z`` variable.

    Documented limitations:
        - **A round trip is not geometrically faithful**, for the same reason as PLY: the
          elevation survives as data, but the mesh geometry is the planar projection
          (design section 6).
        - Triangle vertex indices are **1-based** and arrive 0-based, with the base
          recorded for a writer.
        - A vertex's optional lock flag is read for width but not carried; it is an
          editing-state marker the mesh model has no place for.
        - XMS TIN carries no CRS, so ``crs`` is ``None``.

    Examples:
        - Write a small TIN and read it back. The 1-based triangles arrive 0-based and
          the vertex z becomes ``node_z``:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.interchange import (
            ...     XmsTinDriver,
            ...     build_minimal_fixture_xmstin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "surface.tin")
            ...     build_minimal_fixture_xmstin(path)
            ...     dataset = XmsTinDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1
    """

    name = "xmstin"
    long_name = "XMS / SMS TIN"
    suffixes = (".tin",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first significant line is the ``TIN`` header."""
        try:
            for keyword, _fields in iter_records(path):
                return keyword.upper() == _TIN_HEADER
        except OSError:  # pragma: no cover - can_open checks isfile first, so the open
            return False  # here does not fail in practice; a defensive guard only
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an XMS TIN into a :class:`ParsedMesh`.

        Args:
            path: The TIN file to read.
            lazy: Ignored -- the format is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, triangle faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a missing header or count, a short record, or
                a non-numeric field.
        """
        return _TinReader(path).build()


def build_minimal_fixture_xmstin(path: str) -> None:
    """Write the smallest valid XMS TIN, for the cross-driver invariant harness.

    The unit square split into two counter-clockwise triangles, each vertex carrying an
    elevation. Named per driver because this module holds more than one.

    Args:
        path: Where to write the file.

    Examples:
        - The fixture is a ``VERT`` block of four and a ``TRI`` block of two:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.interchange import build_minimal_fixture_xmstin
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.tin")
            ...     build_minimal_fixture_xmstin(path)
            ...     with open(path, encoding="utf-8") as handle:
            ...         text = handle.read()
            >>> text.splitlines()[0]
            'TIN'
            >>> "VERT 4" in text and "TRI 2" in text
            True
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_TIN_MINIMAL)


class _PlyProperty:
    """One declared PLY property: either a scalar or a variable-length list."""

    def __init__(self, name: str, scalar: str, count: str | None = None) -> None:
        self.name = name
        self.scalar = scalar
        self.count = count


class _PlyReader:
    """One PLY file: header parsed first, then the body for ASCII or either byte order."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.endian: str | None = None
        self.vertex_count = 0
        self.face_count = 0
        self.vertex_props: list[_PlyProperty] = []
        self.face_prop: _PlyProperty | None = None

    def build(self) -> ParsedMesh:
        """Parse the header, read the body, and assemble the mesh.

        Returns:
            ParsedMesh: Nodes, faces, ``node_z`` and any extra vertex variables.

        Raises:
            FileFormatNotSupportedError: On any malformed header or body.
        """
        header_bytes = self._read_header()
        if self.endian is None:
            vertices, faces = self._read_ascii(header_bytes)
        else:
            vertices, faces = self._read_binary(header_bytes)
        return self._assemble(vertices, faces)

    def _read_header(self) -> int:
        """Parse the header lines, returning the byte offset where the body begins.

        Raises:
            FileFormatNotSupportedError: On a missing magic line, format, or end marker,
                or an unknown property type.
        """
        with open(self.path, "rb") as handle:
            if handle.readline().strip() != _PLY_MAGIC:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} is not a PLY file: missing the 'ply' magic line"
                )
            current = None
            while True:
                raw = handle.readline()
                if not raw:
                    raise FileFormatNotSupportedError(
                        f"{self.path!r} header ends before '{_PLY_END_HEADER}'"
                    )
                tokens = raw.decode("ascii", errors="replace").split()
                if not tokens:
                    continue
                current = self._consume_header_line(tokens, current)
                if tokens[0] == _PLY_END_HEADER:
                    return handle.tell()

    def _consume_header_line(
        self, tokens: list[str], current: str | None
    ) -> str | None:
        """Handle one header line, tracking which element block is open."""
        keyword = tokens[0]
        if keyword == "format":
            self._set_format(tokens)
        elif keyword == "element":
            return self._set_element(tokens)
        elif keyword == "property":
            self._add_property(tokens, current)
        # `comment`, `obj_info` and the end marker need no handling here.
        return current

    def _set_format(self, tokens: list[str]) -> None:
        """Record the body encoding from the ``format`` line."""
        if len(tokens) < 2 or tokens[1] not in _PLY_FORMATS:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has an unsupported PLY format line: {tokens[1:]!r}"
            )
        self.endian = _PLY_FORMATS[tokens[1]]

    def _set_element(self, tokens: list[str]) -> str:
        """Open a ``vertex`` or ``face`` element block and record its count."""
        if len(tokens) < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has an element line without a name and count: "
                f"{tokens!r}"
            )
        name = tokens[1]
        count = _as_int(tokens[2], f"element {name}", self.path)
        if name == "vertex":
            self.vertex_count = count
        elif name == "face":
            self.face_count = count
        return name

    def _add_property(self, tokens: list[str], current: str | None) -> None:
        """Attach a property to the open element block.

        Raises:
            FileFormatNotSupportedError: On an unknown scalar type, a face list whose
                count type is not an integer (the binary reader cannot size it), or a
                scalar property on the face element -- a trailing per-face property the
                readers do not model, which would mis-align the face rows.
        """
        if tokens[1] == "list":
            # property list <count-type> <index-type> <name>
            count_type = tokens[2]
            if _PLY_TYPES.get(count_type) not in _INTEGER_TYPES:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} declares a face list with a non-integer count type "
                    f"{count_type!r}"
                )
            self.face_prop = _PlyProperty(tokens[4], tokens[3], count=count_type)
            return
        scalar = tokens[1]
        if scalar not in _PLY_TYPES:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares an unknown PLY property type {scalar!r}"
            )
        if current == "vertex":
            self.vertex_props.append(_PlyProperty(tokens[2], scalar))
        elif current == "face":
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares a per-face property {tokens[2]!r}; only the "
                f"vertex-index list is supported on a face, and a trailing scalar would "
                f"mis-align the face rows"
            )

    def _read_ascii(self, offset: int) -> tuple[dict[str, np.ndarray], list[list[int]]]:
        """Read the ASCII body: vertex rows then face rows."""
        with open(self.path, "r", encoding="ascii", errors="replace") as handle:
            handle.seek(offset)
            tokens = handle.read().split()
        cursor = 0
        columns = {prop.name: np.empty(self.vertex_count) for prop in self.vertex_props}
        for row in range(self.vertex_count):
            for prop in self.vertex_props:
                columns[prop.name][row] = self._ascii_number(tokens, cursor)
                cursor += 1
        faces: list[list[int]] = []
        for _ in range(self.face_count):
            count = int(self._ascii_number(tokens, cursor))
            cursor += 1
            faces.append(
                [int(self._ascii_number(tokens, cursor + i)) for i in range(count)]
            )
            cursor += count
        return columns, faces

    def _ascii_number(self, tokens: list[str], index: int) -> float:
        """Return token ``index`` as a float, refusing a truncated body."""
        if index >= len(tokens):
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends inside its data: the header promises more values "
                f"than the body holds"
            )
        try:
            return float(tokens[index])
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a non-numeric PLY value {tokens[index]!r}"
            ) from error

    def _read_binary(
        self, offset: int
    ) -> tuple[dict[str, np.ndarray], list[list[int]]]:
        """Read the binary body: a fixed vertex block then variable-length faces."""
        with open(self.path, "rb") as handle:
            handle.seek(offset)
            buffer = handle.read()
        if self.face_count and self.face_prop is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares faces but no 'property list' for them; the "
                f"face connectivity cannot be read"
            )
        dtype = np.dtype(
            [
                (prop.name, f"{self.endian}{_PLY_TYPES[prop.scalar]}")
                for prop in self.vertex_props
            ]
        )
        block = dtype.itemsize * self.vertex_count
        if len(buffer) < block:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends inside its vertex block: expected {block} bytes, "
                f"found {len(buffer)}"
            )
        records = np.frombuffer(buffer, dtype=dtype, count=self.vertex_count)
        columns = {
            prop.name: records[prop.name].astype(np.float64)
            for prop in self.vertex_props
        }
        faces = list(self._read_binary_faces(buffer[block:]))
        return columns, faces

    def _read_binary_faces(self, buffer: bytes) -> Iterator[list[int]]:
        """Yield each binary face's index list, honouring its own count and index types."""
        count_fmt = f"{self.endian}{_struct_char(_PLY_TYPES[self.face_prop.count])}"
        index_code = _PLY_TYPES[self.face_prop.scalar]
        index_size = np.dtype(index_code).itemsize
        count_size = struct.calcsize(count_fmt)
        cursor = 0
        for _ in range(self.face_count):
            if cursor + count_size > len(buffer):
                raise FileFormatNotSupportedError(
                    f"{self.path!r} ends inside its face block"
                )
            (count,) = struct.unpack_from(count_fmt, buffer, cursor)
            cursor += count_size
            end = cursor + count * index_size
            if end > len(buffer):
                raise FileFormatNotSupportedError(
                    f"{self.path!r} ends inside a face's index list"
                )
            indices = np.frombuffer(
                buffer[cursor:end], dtype=f"{self.endian}{index_code}", count=count
            )
            cursor = end
            yield indices.astype(np.int64).tolist()

    def _assemble(
        self, columns: dict[str, np.ndarray], face_rows: list[list[int]]
    ) -> ParsedMesh:
        """Build the :class:`ParsedMesh` from vertex columns and face rows."""
        for coord in ("x", "y"):
            if coord not in columns:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has no '{coord}' vertex property; it is not a "
                    f"readable surface"
                )
        if not face_rows:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no faces; a point cloud is not a mesh"
            )
        faces = _pad_faces(face_rows, self.path)
        node_z = columns.get("z")
        variables: dict[str, ParsedVariable] = {}
        n_node = len(columns["x"])
        for name, values in columns.items():
            if name in _PLY_COORDS:
                continue
            variables[name] = ParsedVariable(
                location="node", shape=(n_node,), values=values.astype(np.float64)
            )
        return ParsedMesh(
            node_x=columns["x"].astype(np.float64),
            node_y=columns["y"].astype(np.float64),
            faces=faces,
            fill_value=-1,
            start_index=0,
            original_start_index=0,
            node_z=None if node_z is None else node_z.astype(np.float64),
            crs=None,
            variables=variables,
            global_attributes={"source": "Stanford PLY"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )


class _TinReader:
    """One XMS TIN file, read with the text record helper."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Read the ``VERT`` and ``TRI`` blocks and assemble the mesh.

        Returns:
            ParsedMesh: Nodes, triangle faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a missing header or block, a short record, or
                a non-numeric field.
        """
        records = iter_records(self.path)
        try:
            keyword, _ = next(records)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is empty: an XMS TIN starts with a 'TIN' header"
            ) from None
        if keyword.upper() != _TIN_HEADER:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an XMS TIN: expected a 'TIN' header, found "
                f"{keyword!r}"
            )
        node_x, node_y, node_z, faces = self._read_blocks(records)
        return ParsedMesh(
            node_x=np.array(node_x, dtype=np.float64),
            node_y=np.array(node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            start_index=1,
            original_start_index=1,
            node_z=np.array(node_z, dtype=np.float64),
            crs=None,
            variables={},
            global_attributes={"source": "XMS TIN"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_blocks(self, records: Iterator[tuple[str, list[str]]]):
        """Walk the records, filling the vertex and triangle blocks in order."""
        node_x: list[float] = []
        node_y: list[float] = []
        node_z: list[float] = []
        faces: list[list[int]] = []
        for keyword, fields in records:
            upper = keyword.upper()
            if upper == _TIN_VERTICES:
                count = self._block_count(_TIN_VERTICES, fields)
                self._read_vertices(records, count, node_x, node_y, node_z)
            elif upper == _TIN_TRIANGLES:
                count = self._block_count(_TIN_TRIANGLES, fields)
                self._read_triangles(records, count, faces)
        if not node_x:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no {_TIN_VERTICES} block"
            )
        if not faces:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no {_TIN_TRIANGLES} block"
            )
        return node_x, node_y, node_z, _pad_faces(faces, self.path)

    def _block_count(self, card: str, fields: list[str]) -> int:
        """Return a VERT/TRI block's declared count, naming a card that omits it."""
        if not fields:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a {card} card without a count"
            )
        return _as_int(fields[0], card, self.path)

    def _read_vertices(self, records, count, node_x, node_y, node_z) -> None:
        """Read ``count`` ``x y z [lock]`` vertex lines."""
        for _ in range(count):
            keyword, fields = self._next(records, _TIN_VERTICES)
            row = [keyword, *fields]
            if len(row) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short vertex record: expected x y z, found "
                    f"{len(row)} field(s)"
                )
            node_x.append(_as_float(row[0], _TIN_VERTICES, self.path))
            node_y.append(_as_float(row[1], _TIN_VERTICES, self.path))
            node_z.append(_as_float(row[2], _TIN_VERTICES, self.path))

    def _read_triangles(self, records, count, faces) -> None:
        """Read ``count`` 1-based triangle lines."""
        for _ in range(count):
            keyword, fields = self._next(records, _TIN_TRIANGLES)
            row = [keyword, *fields]
            if len(row) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short triangle record: expected three vertex "
                    f"ids, found {len(row)}"
                )
            faces.append(
                [_as_int(value, _TIN_TRIANGLES, self.path) for value in row[:3]]
            )

    def _next(self, records, what: str) -> tuple[str, list[str]]:
        """Return the next record, refusing a block that ends early."""
        try:
            return next(records)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends inside its {what} block: the count promises more "
                f"records than the file holds"
            ) from None


def _pad_faces(rows: list[list[int]], path: str) -> np.ndarray:
    """Pad variable-width face rows to a rectangular ``-1``-filled array.

    Raises:
        FileFormatNotSupportedError: If any face has fewer than three vertices, which is
            an edge or a point rather than a surface element.
    """
    widths = [len(row) for row in rows]
    if min(widths) < 3:
        raise FileFormatNotSupportedError(
            f"{path!r} has a face with fewer than three vertices, which is not a "
            f"surface element"
        )
    faces = np.full((len(rows), max(widths)), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        faces[index, : len(row)] = row
    return faces


def _struct_char(code: str) -> str:
    """Return the :mod:`struct` format character for a numpy integer type code."""
    return {"i1": "b", "u1": "B", "i2": "h", "u2": "H", "i4": "i", "u4": "I"}[code]


def _as_float(value: str, what: str, path: str) -> float:
    """Parse one field as a float, naming the record and file if it is not one."""
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {what} record: expected a number, found "
            f"{value!r}"
        ) from error


def _as_int(value: str, what: str, path: str) -> int:
    """Parse one field as an integer, naming the record and file if it is not one."""
    try:
        return int(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {what} record: expected an integer, found "
            f"{value!r}"
        ) from error
