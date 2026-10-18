"""SCRIP remapping-grid reader.

SCRIP (the Spherical Coordinate Remapping and Interpolation Package) describes a grid
for remapping as NetCDF: every cell is given by its corner coordinates directly in
``grid_corner_lat``/``grid_corner_lon`` (shape ``grid_size`` x ``grid_corners``), with
``grid_center_lat``/``grid_center_lon`` the cell centres and ``grid_imask`` a cell mask.
There is no shared node list -- the same corner is repeated by every cell that touches
it -- so the work is deduplicating those corners into a node table and emitting
face-node connectivity into it.

Read through the NetCDF substrate the vendored GDAL provides, the same multidimensional
route the other NetCDF drivers use, with no declared ``netCDF4`` or ``h5py``. SCRIP grids
use ``.nc``; they are reached through ``read_file``'s NetCDF-suffix fallback
(``pyramids-gis`` 0.69.0), since a SCRIP grid is not UGRID.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The arrays that make a NetCDF file a SCRIP grid: the per-cell corner tables. They are
# unique to SCRIP, so requiring them declines every sibling ``.nc`` format (MPAS, ICON,
# ESMF, GEOS, 3Di, TUFLOW FV), none of which carries a ``grid_corner_*`` array.
_CORNER_LAT = "grid_corner_lat"
_CORNER_LON = "grid_corner_lon"
_CENTER_LAT = "grid_center_lat"
_CENTER_LON = "grid_center_lon"

# Corner coordinates are rounded to this many decimals before hashing: float equality on
# a shared corner is not reliable, and a rounded key makes the dedup stable.
_DEDUP_DECIMALS = 6


def _group_names(group) -> set[str]:
    """Return the set of root-group MD-array names.

    Args:
        group: An open GDAL multidimensional root group.

    Returns:
        set[str]: The array names in the group.
    """
    return set(group.GetMDArrayNames())


def _array_units(group, name: str) -> str:
    """Return an array's ``units`` attribute, or an empty string.

    An attribute whose value cannot be read is reported as absent rather than allowed to
    abort the parse.

    Args:
        group: The open root group.
        name: The array whose ``units`` attribute to read.

    Returns:
        str: The units string, or ``""`` when the attribute is missing or unreadable.
    """
    for attribute in group.OpenMDArray(name).GetAttributes():
        if attribute.GetName() == "units":
            try:
                return str(attribute.Read())
            except RuntimeError:
                return ""
    return ""


def _to_degrees(values: np.ndarray, units: str) -> np.ndarray:
    """Return ``values`` in degrees, converting from radians when ``units`` says so.

    Args:
        values: A latitude or longitude array in the file's native unit.
        units: The array's declared units.

    Returns:
        np.ndarray: The array in degrees -- converted when the units name radians, else
            returned unchanged.
    """
    if "rad" in units.lower():
        return np.degrees(values)
    return values


class ScripDriver(BaseDriver):
    """Reader for SCRIP remapping-grid NetCDF.

    SCRIP stores each cell's corner coordinates (``grid_corner_lat``/``grid_corner_lon``)
    rather than a node list and connectivity. The driver deduplicates the corners into a
    shared node table and builds face-node connectivity into it, so a SCRIP grid becomes
    an ordinary :class:`~pyramids.netcdf.ugrid.UgridDataset`. The cell centres
    (``grid_center_lat``/``grid_center_lon``) become the face centres.

    Documented limitations:
        - **Corner dedup uses a rounded key, not float equality.** Shared corners are
          matched after rounding to a fixed tolerance; raw float comparison would leave
          duplicate nodes and split edges. The unique nodes are sorted, so the same file
          gives the same node order every run.
        - **A repeated corner marks padding.** SCRIP fills a cell with fewer corners than
          the ``grid_corners`` width by repeating a corner; a corner that duplicates an
          earlier one in the same cell becomes ``-1`` fill, so a triangle in a quad-width
          table yields a fill-padded three-node face. SCRIP repeats a trailing corner, so
          the padding lands in the tail of the row as pyramids requires.
        - **Coordinates are carried in the file's unit, degrees by default.** A
          ``units`` attribute naming radians is honoured and the coordinates converted to
          degrees; otherwise they are left as the file gave them.
        - **No CRS.** The grid is lon/lat on a sphere with no EPSG authority, so ``crs``
          is ``None`` rather than guessed.
        - **``grid_imask`` is not used to drop cells.** The mask is read past but every
          cell is kept; this driver declares ``read_mesh`` only and ships no variables.

    Example:
        Read a SCRIP grid (shown, not run as a doctest -- it needs a real NetCDF file),
        reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("outCSne8.nc")   # corners -> shared nodes
            mesh.mesh.n_face                               # the SCRIP cells
    """

    name = "scrip"
    long_name = "SCRIP remapping grid"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a SCRIP grid, by its corner arrays.

        Keyed on ``grid_corner_lat`` and ``grid_corner_lon``, which are unique to SCRIP,
        so a plain NetCDF or another model's ``.nc`` -- MPAS, ICON, ESMF, GEOS, 3Di or
        TUFLOW FV -- is not claimed. Opening a non-NetCDF file raises inside GDAL and is
        reported as "not SCRIP" rather than propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when both corner arrays are present.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                names = _group_names(dataset.GetRootGroup())
            finally:
                dataset.Close()
        return {_CORNER_LAT, _CORNER_LON} <= names

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a SCRIP grid into a :class:`ParsedMesh`.

        Args:
            path: The SCRIP ``.nc`` file.
            lazy: Unused -- this driver reads only the topology, which is eager.

        Returns:
            ParsedMesh: Nodes deduplicated from the cell corners, the face connectivity
                and the cell centres.

        Raises:
            FileFormatNotSupportedError: If a defining corner array is missing.
        """
        return _ScripReader(path).build()


class _ScripReader:
    """One SCRIP file: dedup the cell corners and build the parsed mesh."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file, dedup corners to nodes, and assemble the mesh.

        Returns:
            ParsedMesh: The deduplicated surface mesh with its cell centres.

        Raises:
            FileFormatNotSupportedError: If a defining corner array is missing.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = _group_names(group)
                self._require(names)
                corner_units = _array_units(group, _CORNER_LAT)
                lon = _to_degrees(
                    np.asarray(
                        group.OpenMDArray(_CORNER_LON).ReadAsArray(), np.float64
                    ),
                    _array_units(group, _CORNER_LON),
                )
                lat = _to_degrees(
                    np.asarray(
                        group.OpenMDArray(_CORNER_LAT).ReadAsArray(), np.float64
                    ),
                    corner_units,
                )
                node_x, node_y, faces = self._dedup(lon, lat)
                face_x, face_y = self._face_centres(group, names)
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=0,
            original_start_index=0,
            face_x=face_x,
            face_y=face_y,
            # Lon/lat on a sphere with no EPSG authority -- see the class docs.
            crs=None,
            global_attributes={"source": "SCRIP remapping grid"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self, names: set[str]) -> None:
        """Refuse a file missing a defining corner array.

        Args:
            names: The array names present in the file.

        Raises:
            FileFormatNotSupportedError: If either corner array is absent.
        """
        missing = {_CORNER_LAT, _CORNER_LON} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a SCRIP grid: missing {sorted(missing)}"
            )

    def _dedup(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deduplicate cell corners into a shared node table and face connectivity.

        Corners are matched on a rounded key so shared corners collapse to one node, and
        the unique nodes are sorted so the node order is deterministic. A corner that
        duplicates an earlier one in the same cell is the format's padding and becomes
        ``-1`` fill.

        Args:
            lon: The ``(grid_size, grid_corners)`` corner longitudes, in degrees.
            lat: The ``(grid_size, grid_corners)`` corner latitudes, in degrees.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The unique node x- and
                y-coordinates and the ``(grid_size, grid_corners)`` fill-padded, 0-based
                face connectivity.
        """
        n_face, n_corner = lon.shape
        corners = np.stack([lon.ravel(), lat.ravel()], axis=-1)
        rounded = np.round(corners, _DEDUP_DECIMALS)
        _unique, first_index, inverse = np.unique(
            rounded, axis=0, return_index=True, return_inverse=True
        )
        inverse = np.ravel(np.asarray(inverse)).reshape(n_face, n_corner)
        faces = self._pad_repeated_corners(inverse)
        # The rounded array is used only for matching; store each node's first-occurrence
        # original (unrounded) corner so node coordinates keep full precision.
        node_x = corners[first_index, 0].astype(np.float64)
        node_y = corners[first_index, 1].astype(np.float64)
        return node_x, node_y, faces

    @staticmethod
    def _pad_repeated_corners(connectivity: np.ndarray) -> np.ndarray:
        """Mark a corner that repeats an earlier one in its cell as ``-1`` padding.

        SCRIP pads a short cell by repeating a corner; the first occurrence of each node
        is kept and any later repeat in the same row becomes fill, so a triangle in a
        quad-width table yields a three-node face padded in its tail.

        Args:
            connectivity: The ``(n_face, n_corner)`` node indices, one row per cell.

        Returns:
            np.ndarray: The connectivity with repeated corners replaced by ``-1``.
        """
        faces = connectivity.astype(np.int64).copy()
        for row in faces:
            seen: set[int] = set()
            for position, node in enumerate(row):
                value = int(node)
                if value in seen:
                    row[position] = -1
                else:
                    seen.add(value)
        return faces

    @staticmethod
    def _face_centres(group, names: set[str]):
        """Return the cell centres as face_x/face_y, when present.

        Both ``grid_center_lon`` and ``grid_center_lat`` must be present; one without the
        other yields no centres rather than a half-populated pair.

        Args:
            group: The open root group.
            names: The array names present in the file.

        Returns:
            tuple: ``(face_x, face_y)`` in degrees, or ``(None, None)`` when absent.
        """
        if _CENTER_LON not in names or _CENTER_LAT not in names:
            return None, None
        face_x = _to_degrees(
            np.asarray(group.OpenMDArray(_CENTER_LON).ReadAsArray(), np.float64),
            _array_units(group, _CENTER_LON),
        )
        face_y = _to_degrees(
            np.asarray(group.OpenMDArray(_CENTER_LAT).ReadAsArray(), np.float64),
            _array_units(group, _CENTER_LAT),
        )
        return face_x, face_y
