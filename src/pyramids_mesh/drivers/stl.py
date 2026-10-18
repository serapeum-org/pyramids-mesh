"""STL (stereolithography) surface-mesh reader.

STL is a triangle-soup geometry format with two encodings: ASCII
(``solid``/``facet``/``outer loop``/``vertex``/``endloop``/``endfacet``/``endsolid``) and
binary (an 80-byte header, a ``uint32`` triangle count, then 50 bytes per triangle -- a
normal and three vertices as 12 ``float32`` values, plus a 2-byte attribute). It is not a
solver format: there is no time axis, and it carries no CRS or georeferencing of any kind,
so ``crs`` is ``None`` (design section 6).

STL has **no vertex indexing** -- every triangle stores its three corner coordinates
inline -- so the reader welds identical vertices into a shared node list and rebuilds the
triangle connectivity against it. As with the other geometry readers, ``z`` becomes a
``node_z`` node variable, since pyramids' mesh model is planar.

Hand-parsed; no third-party geometry library is pulled in.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import struct

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The binary layout, in bytes. A binary STL is exactly ``_HEADER + _COUNT + _TRIANGLE *
# count`` long, and that size identity -- not the leading word -- is what tells binary
# from ASCII: a binary STL's 80-byte header very often *also* begins with "solid".
_STL_HEADER = 80
_STL_COUNT = 4
_STL_TRIANGLE = 50  # 12 float32 (normal + three vertices) + a 2-byte attribute count


class StlDriver(BaseDriver):
    """Reader for the STL (stereolithography) format, ASCII and binary.

    STL stores a flat list of triangles, each carrying its three corner coordinates
    inline with no shared vertex table. The reader welds byte-identical vertices into one
    node list and rebuilds the triangle face connectivity against it, so a closed solid
    comes back as a proper indexed mesh rather than a triangle soup.

    Documented limitations:
        - **A round trip is not geometrically faithful.** ``z`` survives as a ``node_z``
          variable, but pyramids' mesh model is planar, so a consumer reading the
          geometry sees the ``(x, y)`` projection -- and a 3D solid's vertical faces
          collapse to zero-area (degenerate) faces in that projection (design section 6).
        - **STL carries no CRS or georeferencing**, so ``crs`` is ``None``. This makes it
          a low-value format for a GIS stack; it is provided for interchange only.
        - **The whole file is read eagerly**, so the driver does not declare ``lazy_data``.
        - The per-triangle normal and the 2-byte attribute word are read for width but
          not carried; the mesh model has no place for either.
        - Welded vertices are 0-based, as pyramids expects, and stay so.

    Examples:
        - Write a tiny two-triangle ASCII STL and read it back. The shared corner is
          welded, so the unit square comes back as four nodes and two faces:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.stl import StlDriver
            >>> lines = (
            ...     "solid square",
            ...     "facet normal 0 0 1",
            ...     " outer loop",
            ...     "  vertex 0 0 0",
            ...     "  vertex 1 0 0",
            ...     "  vertex 1 1 0",
            ...     " endloop",
            ...     " endfacet",
            ...     "facet normal 0 0 1",
            ...     " outer loop",
            ...     "  vertex 0 0 0",
            ...     "  vertex 1 1 0",
            ...     "  vertex 0 1 0",
            ...     " endloop",
            ...     " endfacet",
            ...     "endsolid square",
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "square.stl")
            ...     with open(path, "w", encoding="ascii") as handle:
            ...         for line in lines:
            ...             print(line, file=handle)
            ...     dataset = StlDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.crs is None
            True
    """

    name = "stl"
    long_name = "STL (stereolithography)"
    suffixes = (".stl",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a size-consistent binary or an ASCII STL.

        Binary is tested first and by size, not by the leading word: a binary STL's
        header often starts with ``solid`` too, so only the ``80 + 4 + 50 * count`` size
        identity tells the two apart reliably. Never raises -- an unreadable file is
        reported as not an STL.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` for a size-consistent binary STL or an ASCII ``solid``/``facet``
                file.
        """
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError:  # pragma: no cover - can_open checks isfile first
            return False
        if _binary_triangle_count(raw) is not None:
            return True
        return _looks_ascii(raw)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an STL file into a :class:`ParsedMesh`.

        Args:
            path: The STL file to read.
            lazy: Ignored -- the whole file is read eagerly.

        Returns:
            ParsedMesh: Welded nodes, triangle faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a file that is neither a size-consistent
                binary STL nor an ASCII ``solid``/``facet`` file, a facet without exactly
                three vertices, or a non-numeric coordinate.
        """
        with open(path, "rb") as handle:
            raw = handle.read()
        count = _binary_triangle_count(raw)
        if count is not None:
            if count == 0:
                raise FileFormatNotSupportedError(
                    f"{path!r} is an empty binary STL (zero triangles); it carries no "
                    f"surface to build a mesh from"
                )
            triangles = _read_binary_triangles(raw, count)
        elif _looks_ascii(raw):
            triangles = _read_ascii_triangles(raw, path)
        else:
            raise FileFormatNotSupportedError(
                f"{path!r} is not a readable STL: it is neither a size-consistent "
                f"binary STL (an 80-byte header, a uint32 triangle count and 50 bytes "
                f"per triangle) nor an ASCII 'solid'/'facet' file"
            )
        return _weld(triangles, path)


def _binary_triangle_count(raw: bytes) -> int | None:
    """Return the triangle count iff ``raw`` is a size-consistent binary STL, else ``None``.

    Args:
        raw: The whole file.

    Returns:
        int | None: The declared triangle count when the file length is exactly
            ``80 + 4 + 50 * count``; ``None`` otherwise, so an ASCII file (or a truncated
            binary one) falls through to the ASCII path.
    """
    if len(raw) < _STL_HEADER + _STL_COUNT:
        return None
    (count,) = struct.unpack_from("<I", raw, _STL_HEADER)
    if len(raw) == _STL_HEADER + _STL_COUNT + _STL_TRIANGLE * count:
        return count
    return None


def _looks_ascii(raw: bytes) -> bool:
    """Report whether ``raw`` decodes as an ASCII STL carrying ``solid`` and ``facet``."""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return False
    lowered = text.lower()
    return lowered.lstrip().startswith("solid") and "facet" in lowered


def _read_binary_triangles(raw: bytes, count: int) -> np.ndarray:
    """Read ``count`` binary triangles into an ``(count, 3, 3)`` array of vertex coords.

    Each 50-byte record is 12 ``float32`` values -- a normal then three vertices -- and a
    2-byte attribute. The normal and the attribute are read for width and discarded; only
    the nine vertex coordinates are kept.

    Args:
        raw: The whole file.
        count: The triangle count, already size-checked by :func:`_binary_triangle_count`.

    Returns:
        np.ndarray: The three vertex ``(x, y, z)`` triples of every triangle.
    """
    offset = _STL_HEADER + _STL_COUNT
    block = raw[offset : offset + _STL_TRIANGLE * count]
    dtype = np.dtype([("floats", "<f4", 12), ("attr", "<u2")])
    records = np.frombuffer(block, dtype=dtype, count=count)
    floats = records["floats"].astype(np.float64)
    # Columns 0..2 are the facet normal; 3..11 are the three vertex xyz triples.
    return floats[:, 3:12].reshape(count, 3, 3)


def _read_ascii_triangles(raw: bytes, path: str) -> np.ndarray:
    """Read an ASCII STL's ``vertex`` lines into an ``(n_tri, 3, 3)`` array.

    Args:
        raw: The whole file.
        path: The source path, for error messages.

    Returns:
        np.ndarray: The three vertex ``(x, y, z)`` triples of every facet.

    Raises:
        FileFormatNotSupportedError: On a facet without exactly three vertices, a short
            ``vertex`` line, a non-numeric coordinate, or a file with no facets at all.
    """
    text = raw.decode("ascii", errors="replace")
    triangles: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == "vertex":
            current.append(_ascii_vertex(tokens, path))
        elif head == "endfacet":
            if len(current) != 3:
                raise FileFormatNotSupportedError(
                    f"{path!r} has a facet with {len(current)} vertices; an STL facet "
                    f"is a triangle and must have exactly three"
                )
            triangles.append(current)
            current = []
    if not triangles:
        raise FileFormatNotSupportedError(
            f"{path!r} is an ASCII STL with no complete facets; there is no surface to "
            f"read"
        )
    return np.array(triangles, dtype=np.float64)


def _ascii_vertex(tokens: list[str], path: str) -> list[float]:
    """Parse one ``vertex x y z`` line into three floats."""
    if len(tokens) < 4:
        raise FileFormatNotSupportedError(
            f"{path!r} has a short vertex line: expected 'vertex x y z', found "
            f"{len(tokens)} token(s)"
        )
    return [_as_float(tokens[index], path) for index in (1, 2, 3)]


def _weld(triangles: np.ndarray, path: str) -> ParsedMesh:
    """Weld byte-identical vertices into a node list and build triangle connectivity.

    Args:
        triangles: An ``(n_tri, 3, 3)`` array of per-triangle vertex coordinates.
        path: The source path, recorded on the mesh's global attributes.

    Returns:
        ParsedMesh: The welded nodes, 0-based triangle faces and the ``node_z`` elevation.
    """
    tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    nodes: list[tuple[float, float, float]] = []
    index_of: dict[tuple[float, float, float], int] = {}
    connectivity: list[list[int]] = []
    for triangle in tris:
        row: list[int] = []
        for vertex in triangle:
            key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            index = index_of.get(key)
            if index is None:
                index = len(nodes)
                index_of[key] = index
                nodes.append(key)
            row.append(index)
        connectivity.append(row)
    coordinates = np.array(nodes, dtype=np.float64)
    return ParsedMesh(
        node_x=coordinates[:, 0],
        node_y=coordinates[:, 1],
        faces=np.array(connectivity, dtype=np.int64),
        fill_value=-1,
        start_index=0,
        original_start_index=0,
        node_z=coordinates[:, 2],
        crs=None,
        variables={},
        global_attributes={"source": "STL"},
        topology_dimension=2,
        mesh_name="mesh2d",
    )


def _as_float(value: str, path: str) -> float:
    """Parse one field as a float, naming the file if it is not one."""
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed vertex record: expected a number, found "
            f"{value!r}"
        ) from error
