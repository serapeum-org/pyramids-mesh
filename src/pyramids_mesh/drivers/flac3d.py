"""FLAC3D grid reader.

FLAC3D is Itasca's explicit finite-difference geomechanics code, and its ``.f3grid`` is
a flat, line-keyed ASCII grid: ``*`` comment lines, ``G <id> <x> <y> <z>`` gridpoints,
and ``Z <zonetype> <id> <gp1>..<gpN>`` zones whose type code fixes the gridpoint count
(``T4`` tetrahedron 4, ``W6`` wedge 6, ``B8`` brick 8, ``P5`` pyramid 5). A file may also
carry ``ZGROUP``/``FGROUP`` groupings and an ``F`` face table; those are ignored here --
only the gridpoints and the zones build the mesh.

Each zone is a solid element, and -- following the Exodus precedent (``exodus.py``) -- its
gridpoint list is mapped straight to a fill-padded face row rather than expanded into its
bounding polygons: boundary extraction is pyramids' job, not a driver's (ADR-0005). A
brick's eight gridpoint ids become an eight-node face, winding then judged in the ``xy``
plane like any other face. Gridpoint ids may be gapped or out of order, so they are
resolved through :func:`~pyramids_mesh._text.native_id_map` rather than assumed
contiguous. The coordinates are engineering coordinates with no datum, so ``crs`` is
``None``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records, native_id_map

# FLAC3D comments open with `*` (e.g. `* GRIDPOINTS`, `* ZONES`), not `#`.
_COMMENT = "*"

# The line keywords that build the mesh. Everything else -- ZGROUP, FGROUP, the F face
# table, and the bare id-continuation rows under a group -- is ignored.
_GRIDPOINT = "G"
_ZONE = "Z"

# Zone type code -> gridpoint count. A brick carries eight gridpoints, a wedge six, a
# pyramid five and a tetrahedron four; the count fixes how many ids a zone row holds.
_ZONE_NODE_COUNTS = {"T4": 4, "W6": 6, "B8": 8, "P5": 5}

# How many bytes of the head the sniff decodes looking for a gridpoint and a zone line.
_SNIFF_LIMIT = 64 * 1024


class Flac3dDriver(BaseDriver):
    """Reader for the Itasca FLAC3D grid (``.f3grid``).

    Gridpoints (``G``) become ``node_x``/``node_y`` and, when any is non-zero,
    ``node_z``; zones (``Z``) become the fill-padded ``face_node_connectivity``, one face
    row per zone. A file mixing zone types (bricks among tetrahedra) pads the short rows
    to the widest with ``-1``.

    Documented limitations:
        - **Every zone's gridpoint list becomes a face row**, including a volume element
          such as a ``B8`` brick. A genuine volumetric mesh is not modelled yet, so a
          brick's eight gridpoint ids are read verbatim as an eight-node face rather than
          expanded into its bounding quadrilaterals (the Exodus precedent); winding is
          judged in the ``xy`` plane like any other face, so a projected-degenerate zone
          is reported degenerate rather than reversed.
        - **Gridpoint ids are resolved through an id map.** FLAC3D ids may be gapped or
          out of file order, so connectivity is resolved by position, never by assuming
          ``id == index + 1``; the ``1`` base FLAC3D writes is recorded for a writer.
        - **Only gridpoints and zones are read.** The ``ZGROUP``/``FGROUP`` groupings and
          the ``F`` face table a file may carry are ignored.
        - FLAC3D grids are engineering coordinates with no datum, so ``crs`` is ``None``.

    Example:
        Read a FLAC3D grid (shown, not run as a doctest -- it reads a committed data
        file), reached through ``read_file``::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("model.f3grid")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "flac3d"
    long_name = "FLAC3D grid"
    suffixes = (".f3grid",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file head carries a ``G`` gridpoint and a ``Z`` zone line.

        A bounded prefix is decoded leniently -- so a binary or empty file answers
        ``False`` rather than raising -- and scanned for a significant line whose first
        token is ``G`` and another whose first token is ``Z``. Requiring both keeps a
        Kratos ``.mdpa`` or an unrelated text file from qualifying.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when both a gridpoint and a zone line are found.
        """
        with open(path, "rb") as handle:
            prefix = handle.read(_SNIFF_LIMIT)
        has_gridpoint = False
        has_zone = False
        for line in prefix.decode("utf-8-sig", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(_COMMENT):
                continue
            token = stripped.split()[0]
            if token == _GRIDPOINT:
                has_gridpoint = True
            elif token == _ZONE:
                has_zone = True
            if has_gridpoint and has_zone:
                return True
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a FLAC3D grid into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The ``.f3grid`` file to read.
            lazy: Ignored -- a FLAC3D grid is plain text and read eagerly.

        Returns:
            ParsedMesh: Gridpoints as nodes, zones as the fill-padded face connectivity,
                and the per-zone type codes recorded as metadata.

        Raises:
            FileFormatNotSupportedError: If the file declares no gridpoints, no zones, a
                zone of an unknown type code, a short zone row, or a non-numeric field.
        """
        scan = _Scan(path)
        scan.run()
        return scan.build()


class _Scan:
    """One FLAC3D grid, streamed once into gridpoints and zones."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.gridpoint_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.zone_vertices: list[list[int]] = []
        self.zone_types: list[str] = []

    def run(self) -> None:
        """Stream the file, collecting ``G`` gridpoints and ``Z`` zones in order.

        Every other keyword -- the group tables and the face table -- is ignored, as are
        the bare id-continuation rows beneath a ``ZGROUP``/``FGROUP``.

        Raises:
            FileFormatNotSupportedError: On an unknown zone type, a short record, or a
                non-numeric coordinate or id.
        """
        for keyword, fields in iter_records(self.path, comment=_COMMENT):
            if keyword == _GRIDPOINT:
                self._add_gridpoint(fields)
            elif keyword == _ZONE:
                self._add_zone(fields)

    def _add_gridpoint(self, fields: list[str]) -> None:
        """Collect one ``G <id> <x> <y> <z>`` gridpoint record.

        Raises:
            FileFormatNotSupportedError: If the record is short of an id and three
                coordinates, or a field is not numeric.
        """
        if len(fields) < 4:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short {_GRIDPOINT} record: expected an id and "
                f"three coordinates, found {len(fields)} field(s)"
            )
        self.gridpoint_ids.append(_as_int(self.path, _GRIDPOINT, fields[0]))
        self.node_x.append(_as_float(self.path, _GRIDPOINT, fields[1]))
        self.node_y.append(_as_float(self.path, _GRIDPOINT, fields[2]))
        self.node_z.append(_as_float(self.path, _GRIDPOINT, fields[3]))

    def _add_zone(self, fields: list[str]) -> None:
        """Collect one ``Z <zonetype> <id> <gp1>..<gpN>`` zone record.

        The type code fixes the gridpoint count, so a row short of it is a format error
        rather than a silently truncated element.

        Raises:
            FileFormatNotSupportedError: If the type code is unknown, the row is short of
                the gridpoints its type declares, or a gridpoint id is not an integer.
        """
        if len(fields) < 2:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short {_ZONE} record: expected a type code and an "
                f"id, found {len(fields)} field(s)"
            )
        zone_type = fields[0].upper()
        count = _ZONE_NODE_COUNTS.get(zone_type)
        if count is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} uses the zone type {fields[0]!r}, which this reader does "
                f"not know; FLAC3D writes T4, W6, B8 and P5"
            )
        if len(fields) < 2 + count:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short {zone_type} zone: expected {count} "
                f"gridpoint ids, found {len(fields) - 2}"
            )
        self.zone_vertices.append(
            [_as_int(self.path, _ZONE, value) for value in fields[2 : 2 + count]]
        )
        self.zone_types.append(zone_type)

    def build(self) -> ParsedMesh:
        """Assemble the parsed mesh from the collected gridpoints and zones.

        Returns:
            ParsedMesh: The mesh, 0-based after id resolution with the ``1`` base
                recorded for a writer, and the per-zone type codes as metadata.

        Raises:
            FileFormatNotSupportedError: If the file declares no gridpoints or no zones.
            ValueError: If a zone references a gridpoint id the file never defined, or
                two gridpoints share an id.
        """
        if not self.gridpoint_ids:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no {_GRIDPOINT} gridpoints, so it is not a "
                f"readable FLAC3D grid"
            )
        if not self.zone_vertices:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no {_ZONE} zones, so there is no connectivity "
                f"to build a mesh from"
            )
        node_map = native_id_map(self.gridpoint_ids)
        faces = self._resolve_faces(node_map)
        node_z = np.array(self.node_z, dtype=np.float64)
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Gridpoint ids were resolved through the id map, so the connectivity is
            # already 0-based; FLAC3D numbers from 1, recorded for a writer to restore.
            start_index=0,
            original_start_index=1,
            node_z=node_z if np.any(node_z != 0.0) else None,
            crs=None,
            global_attributes={"source": "FLAC3D", "zone_types": list(self.zone_types)},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _resolve_faces(self, node_map: dict[int, int]) -> np.ndarray:
        """Map each zone's gridpoint ids to 0-based indices, padding short rows ``-1``.

        Args:
            node_map: The gridpoint-id -> 0-based-index mapping.

        Returns:
            np.ndarray: The ``(n_zone, max_nodes)`` fill-padded face connectivity.

        Raises:
            ValueError: If a zone references a gridpoint id the file never defined.
        """
        width = max(len(vertices) for vertices in self.zone_vertices)
        faces = np.full((len(self.zone_vertices), width), -1, dtype=np.int64)
        for row, vertices in enumerate(self.zone_vertices):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, self.path)
        return faces


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Return a gridpoint id's 0-based index, or raise naming the undefined id.

    Args:
        native: The gridpoint id a zone referenced.
        node_map: The gridpoint-id -> 0-based-index mapping.
        path: The source path, for the error message.

    Returns:
        int: The 0-based node index.

    Raises:
        ValueError: If the id was never defined by a ``G`` record.
    """
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} has a zone referencing gridpoint id {native}, which no "
            f"{_GRIDPOINT} record defines"
        ) from None


def _as_int(path: str, keyword: str, value: str) -> int:
    """Parse one field as an integer, naming the keyword and file if it is not one.

    Args:
        path: The source path, for the error message.
        keyword: The record keyword, for the error message.
        value: The field text.

    Returns:
        int: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not an integer.
    """
    try:
        return int(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} field: expected an integer, found "
            f"{value!r}"
        ) from error


def _as_float(path: str, keyword: str, value: str) -> float:
    """Parse one field as a float, naming the keyword and file if it is not one.

    Args:
        path: The source path, for the error message.
        keyword: The record keyword, for the error message.
        value: The field text.

    Returns:
        float: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not a number.
    """
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} field: expected a number, found "
            f"{value!r}"
        ) from error
