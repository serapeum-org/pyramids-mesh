"""Wavefront OBJ surface-mesh reader.

OBJ is a plain-text geometry interchange format: ``v x y z`` vertex lines and ``f``
face lines whose indices reference those vertices **1-based**. It is not a solver
format -- there is no time axis, and it carries no CRS or georeferencing of any kind, so
``crs`` is ``None`` (design section 6). As with the other geometry readers, ``z`` becomes
a ``node_z`` node variable rather than a coordinate, since pyramids' mesh model is planar.

Hand-parsed through the package's own text-record helper; no third-party geometry
library is pulled in for a format this simple.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records

# The two record keywords the reader acts on. Everything else a line can start with --
# ``vn`` normals, ``vt`` texture coordinates, ``g``/``o``/``s`` grouping, material
# cards -- is parsed past rather than modelled: none of it has a home on a planar mesh.
_OBJ_VERTEX = "v"
_OBJ_FACE = "f"


class ObjDriver(BaseDriver):
    """Reader for the Wavefront OBJ geometry format.

    Reads the ASCII ``v``/``f`` subset: each ``v`` line is a vertex ``x y z`` and each
    ``f`` line is a polygon referencing vertices 1-based. A face token may be a bare
    index, or an ``index/texture/normal`` triple (``f v/vt/vn``); only the vertex index
    before the first ``/`` is used. Triangles, quads and higher n-gons are all accepted
    and padded to the widest face with ``-1``.

    Documented limitations:
        - **A round trip is not geometrically faithful.** ``z`` survives as a ``node_z``
          variable, but pyramids' mesh model is planar, so a consumer reading the
          geometry sees the ``(x, y)`` projection (design section 6).
        - **OBJ carries no CRS or georeferencing**, so ``crs`` is ``None``. This makes it
          a low-value format for a GIS stack; it is provided for interchange only.
        - **The whole file is read eagerly** -- a line-oriented text format has no record
          index to seek -- so the driver does not declare ``lazy_data``.
        - Texture coordinates, vertex normals, grouping and material cards are read past
          and not carried; only vertices and faces reach the mesh.
        - Face indices are 1-based in OBJ and arrive 0-based, with the base recorded for
          a writer. Negative (relative) indices are not supported and are refused as
          out-of-range rather than silently mis-wired.

    Examples:
        - Write a small ASCII OBJ and read it back. The vertex ``z`` becomes ``node_z``
          and the 1-based faces arrive 0-based:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.obj import ObjDriver
            >>> lines = (
            ...     "v 0.0 0.0 2.0",
            ...     "v 1.0 0.0 2.5",
            ...     "v 1.0 1.0 3.0",
            ...     "v 0.0 1.0 3.5",
            ...     "f 1 2 3",
            ...     "f 1 3 4",
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "surface.obj")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         for line in lines:
            ...             print(line, file=handle)
            ...     dataset = ObjDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1
    """

    name = "obj"
    long_name = "Wavefront OBJ"
    suffixes = (".obj",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file carries both a ``v`` and an ``f`` record.

        An OBJ is recognised by the presence of at least one vertex line and one face
        line; a point cloud (``v`` only) or a stray text file is declined. Never raises:
        an unreadable file is reported as not an OBJ.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` only when both a vertex and a face record are present.
        """
        try:
            found_vertex = found_face = False
            for keyword, _fields in iter_records(path):
                if keyword == _OBJ_VERTEX:
                    found_vertex = True
                elif keyword == _OBJ_FACE:
                    found_face = True
                if found_vertex and found_face:
                    return True
        except OSError:  # pragma: no cover - can_open checks isfile first
            return False
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an OBJ file into a :class:`ParsedMesh`.

        Args:
            path: The OBJ file to read.
            lazy: Ignored -- the format is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, polygon faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a file with no vertices or no faces, a short
                vertex line, a face of fewer than three vertices, or a non-numeric field.
        """
        return _ObjReader(path).build()


class _ObjReader:
    """One OBJ file, read through the text-record helper."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Walk the records, collecting vertices and faces, and assemble the mesh.

        Returns:
            ParsedMesh: Nodes, faces and the ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On a malformed or empty surface.
        """
        node_x: list[float] = []
        node_y: list[float] = []
        node_z: list[float] = []
        face_rows: list[list[int]] = []
        for keyword, fields in iter_records(self.path):
            if keyword == _OBJ_VERTEX:
                self._read_vertex(fields, node_x, node_y, node_z)
            elif keyword == _OBJ_FACE:
                face_rows.append(self._read_face(fields))
        if not node_x:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no 'v' vertex lines; it is not a readable OBJ surface"
            )
        if not face_rows:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no 'f' faces; a point cloud is not a mesh"
            )
        faces = _pad_faces(face_rows, self.path)
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
            global_attributes={"source": "Wavefront OBJ"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_vertex(
        self,
        fields: list[str],
        node_x: list[float],
        node_y: list[float],
        node_z: list[float],
    ) -> None:
        """Read one ``v x y z`` line, taking the first three coordinates."""
        if len(fields) < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short vertex record: expected x y z, found "
                f"{len(fields)} field(s)"
            )
        node_x.append(_as_float(fields[0], "v", self.path))
        node_y.append(_as_float(fields[1], "v", self.path))
        node_z.append(_as_float(fields[2], "v", self.path))

    def _read_face(self, fields: list[str]) -> list[int]:
        """Read one ``f`` line into a list of 1-based vertex indices.

        Each token may be ``v``, ``v/vt`` or ``v/vt/vn``; only the vertex index before
        the first ``/`` is taken.
        """
        if len(fields) < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a face with fewer than three vertices, which is not "
                f"a surface element"
            )
        return [self._vertex_index(token) for token in fields]

    def _vertex_index(self, token: str) -> int:
        """Return the 1-based vertex index from an ``f`` token, dropping any ``/`` part."""
        return _as_int(token.split("/")[0], "f", self.path)


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
