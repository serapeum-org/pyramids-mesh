"""OFF (Object File Format) surface-mesh reader.

OFF is a plain-text geometry interchange format: a magic ``OFF`` line, a
``nVerts nFaces nEdges`` count line, then ``nVerts`` vertex lines ``x y z`` and
``nFaces`` face lines ``k v0 v1 ... v{k-1}`` whose leading ``k`` gives the vertex count
and whose ``k`` ids reference the vertices **0-based**. It is not a solver format --
there is no time axis, and it carries no CRS or georeferencing of any kind, so ``crs`` is
``None`` (design section 6). As with the other geometry readers, ``z`` becomes a
``node_z`` node variable rather than a coordinate, since pyramids' mesh model is planar.

A ``#`` begins a comment that runs to the end of the line (whole-line or trailing), and
blank lines are ignored. A face line may carry per-face RGBA colour values after its
vertex ids; those trailing numbers are not modelled on a planar mesh and are ignored.

Hand-parsed line by line; no third-party geometry library is pulled in for a format this
simple.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

from typing import Iterator

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The magic keyword a readable OFF opens with. Case-folded before the comparison so an
# ``off`` written in lower case is still recognised; the coloured and higher-dimensional
# variants (COFF, 4OFF, nOFF) are deliberately out of scope and declined.
_OFF_MAGIC = "OFF"

# The line-comment marker. Unlike the record helper's whole-line comments, an OFF comment
# may also trail real data -- a vertex or the header -- so it is stripped per line from
# the first occurrence to the end, rather than only skipping lines that start with it.
_OFF_COMMENT = "#"


class OffDriver(BaseDriver):
    """Reader for the OFF (Object File Format) geometry format.

    Reads the ASCII form: an ``OFF`` magic line, a ``nVerts nFaces nEdges`` count line,
    then the vertex block of ``x y z`` lines and the face block of ``k v0 v1 ... v{k-1}``
    lines. The leading ``k`` on a face is its vertex count and the ``k`` ids that follow
    are 0-based; any further values on the line are per-face RGB/RGBA colour and are
    ignored. Triangles, quads and higher n-gons are all accepted and padded to the widest
    face with ``-1``.

    Documented limitations:
        - **A round trip is not geometrically faithful.** ``z`` survives as a ``node_z``
          variable, but pyramids' mesh model is planar, so a consumer reading the
          geometry sees the ``(x, y)`` projection (design section 6).
        - **OFF carries no CRS or georeferencing**, so ``crs`` is ``None``. This makes it
          a low-value format for a GIS stack; it is provided for interchange only.
        - **The whole file is read eagerly** -- a line-oriented text format has no record
          index to seek -- so the driver does not declare ``lazy_data``.
        - **Per-face colour is ignored.** Trailing RGB/RGBA values on a face line are read
          past; a planar mesh has no home for them.
        - The declared edge count (the third header field) is advisory and ignored; only
          the vertex and face blocks are read.
        - Face indices are 0-based in OFF and stay 0-based. The coloured and
          higher-dimensional magic variants (``COFF``, ``4OFF``, ``nOFF``) are not
          supported and are declined.

    Examples:
        - Read a small ASCII OFF. The vertex ``z`` becomes ``node_z`` and the 0-based
          faces stay 0-based:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.off import OffDriver
            >>> lines = (
            ...     "OFF",
            ...     "3 1 0",
            ...     "0.0 0.0 2.0",
            ...     "1.0 0.0 2.5",
            ...     "0.0 1.0 3.0",
            ...     "3 0 1 2",
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "surface.off")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         for line in lines:
            ...             print(line, file=handle)
            ...     dataset = OffDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (3, 1)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2]]
            >>> dataset.mesh.crs is None
            True
    """

    name = "off"
    long_name = "Object File Format"
    suffixes = (".off",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first significant line is the ``OFF`` magic.

        A significant line is one that is neither blank nor a comment once any trailing
        comment is stripped. Never raises: an unreadable file is reported as not an OFF.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` only when the first significant line's first token is ``OFF``.
        """
        try:
            for tokens in _iter_significant(path):
                # _iter_significant skips blank/comment lines, so the first yield is a
                # non-empty token list; the first token of the first one must be OFF.
                return tokens[0].upper() == _OFF_MAGIC
        except OSError:  # pragma: no cover - can_open checks isfile first
            return False
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an OFF file into a :class:`ParsedMesh`.

        Args:
            path: The OFF file to read.
            lazy: Ignored -- the format is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, polygon faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a missing ``OFF`` magic, a missing or
                malformed count line, a short vertex record, a face line shorter than its
                declared vertex count, a file declaring no faces, a face of fewer than
                three vertices, a non-numeric field, or a file truncated before its
                header promises.
        """
        return _OffReader(path).build()


def _iter_significant(path: str) -> Iterator[list[str]]:
    """Yield the whitespace tokens of each significant line of an OFF file.

    A trailing or whole-line ``#`` comment is stripped first, then blank lines are
    skipped, so a vertex with an inline comment still reads as its coordinates alone.
    Decoded as ``utf-8-sig`` so a leading byte-order mark is dropped rather than glued
    onto the magic keyword, and undecodable bytes are replaced rather than raising.

    Args:
        path: The OFF file to read.

    Yields:
        list[str]: The whitespace-separated tokens of one significant line.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            marker = line.find(_OFF_COMMENT)
            if marker != -1:
                line = line[:marker]
            stripped = line.strip()
            if not stripped:
                continue
            yield stripped.split()


class _OffReader:
    """One OFF file, read line by line with inline comments stripped."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Read the magic, counts, vertex block and face block, and assemble the mesh.

        Returns:
            ParsedMesh: Nodes, faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a malformed or truncated surface.
        """
        lines = _iter_significant(self.path)
        header = self._read_magic(lines)
        n_verts, n_faces = self._read_counts(lines, header)
        node_x, node_y, node_z = self._read_vertices(lines, n_verts)
        face_rows = self._read_faces(lines, n_faces)
        faces = _pad_faces(face_rows, self.path)
        return ParsedMesh(
            node_x=np.array(node_x, dtype=np.float64),
            node_y=np.array(node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            start_index=0,
            original_start_index=0,
            node_z=np.array(node_z, dtype=np.float64),
            crs=None,
            variables={},
            global_attributes={"source": "Object File Format"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_magic(self, lines: Iterator[list[str]]) -> list[str]:
        """Consume the magic line, returning it so appended counts can be recovered.

        Raises:
            FileFormatNotSupportedError: On an empty file or a first significant line
                whose first token is not ``OFF``.
        """
        try:
            tokens = next(lines)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is empty: an OFF file starts with an 'OFF' magic line"
            ) from None
        if not tokens or tokens[0].upper() != _OFF_MAGIC:
            found = tokens[0] if tokens else ""
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an OFF file: expected an 'OFF' magic line, found "
                f"{found!r}"
            )
        return tokens

    def _read_counts(
        self, lines: Iterator[list[str]], header: list[str]
    ) -> tuple[int, int]:
        """Return ``(n_verts, n_faces)`` from the count line.

        The counts may be appended to the magic line (``OFF 8 6 12``) or stand on the
        next significant line; either way the third field, the advisory edge count, is
        ignored.

        Raises:
            FileFormatNotSupportedError: If the count line is missing or carries fewer
                than two numbers, or a count is not an integer.
        """
        count_tokens = header[1:] if len(header) > 1 else self._next(lines, "header")
        if len(count_tokens) < 2:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a malformed OFF header: expected "
                f"'nVerts nFaces nEdges', found {count_tokens!r}"
            )
        n_verts = _as_int(count_tokens[0], "vertex count", self.path)
        n_faces = _as_int(count_tokens[1], "face count", self.path)
        if n_verts < 0 or n_faces < 0:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a negative OFF count in {count_tokens[:2]!r}"
            )
        return n_verts, n_faces

    def _read_vertices(
        self, lines: Iterator[list[str]], n_verts: int
    ) -> tuple[list[float], list[float], list[float]]:
        """Read ``n_verts`` ``x y z`` vertex lines, taking the first three coordinates."""
        node_x: list[float] = []
        node_y: list[float] = []
        node_z: list[float] = []
        for _ in range(n_verts):
            tokens = self._next(lines, "vertex block")
            if len(tokens) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short vertex record: expected x y z, found "
                    f"{len(tokens)} field(s)"
                )
            node_x.append(_as_float(tokens[0], "vertex", self.path))
            node_y.append(_as_float(tokens[1], "vertex", self.path))
            node_z.append(_as_float(tokens[2], "vertex", self.path))
        return node_x, node_y, node_z

    def _read_faces(self, lines: Iterator[list[str]], n_faces: int) -> list[list[int]]:
        """Read ``n_faces`` ``k v0 ... v{k-1}`` face lines into 0-based index lists.

        The leading ``k`` is the vertex count; exactly the next ``k`` tokens are the
        vertex ids, and any further tokens -- per-face RGB/RGBA colour -- are ignored.
        """
        face_rows: list[list[int]] = []
        for _ in range(n_faces):
            tokens = self._next(lines, "face block")
            count = _as_int(tokens[0], "face vertex count", self.path)
            if count < 0:
                # A negative count would slice from the tail (tokens[1:1+count]) and
                # silently build a wrong face; refuse it. A count of 0/1/2 is left to the
                # short-record and "fewer than three vertices" guards below/downstream.
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a face with a negative leading vertex count {count}"
                )
            if len(tokens) < 1 + count:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short face record: the leading count {count} "
                    f"promises more ids than the line holds ({len(tokens) - 1})"
                )
            face_rows.append(
                [_as_int(token, "face", self.path) for token in tokens[1 : 1 + count]]
            )
        return face_rows

    def _next(self, lines: Iterator[list[str]], what: str) -> list[str]:
        """Return the next significant line, refusing a block that ends early."""
        try:
            return next(lines)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends inside its {what}: the header promises more "
                f"records than the file holds"
            ) from None


def _pad_faces(rows: list[list[int]], path: str) -> np.ndarray:
    """Pad variable-width face rows to a rectangular ``-1``-filled array.

    Raises:
        FileFormatNotSupportedError: If there are no faces at all, or any face has fewer
            than three vertices, which is an edge or a point rather than a surface
            element.
    """
    if not rows:
        raise FileFormatNotSupportedError(
            f"{path!r} declares no faces; a point cloud is not a mesh"
        )
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
