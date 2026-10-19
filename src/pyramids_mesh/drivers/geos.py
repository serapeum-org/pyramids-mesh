"""GEOS cubed-sphere reader.

GEOS (the NASA Global Modeling and Assimilation Office model) writes its native
cubed-sphere grid as NetCDF-4 with a logically structured corner grid per cube face:
``corner_lons``/``corner_lats`` have shape ``(6, ny, nx)`` -- six faces, each an
``ny`` x ``nx`` lattice of corner points in degrees -- and the optional
``lons``/``lats`` (``(6, ny-1, nx-1)``) are the cell centres. The driver ravels the
corner lattice into a node list and reconstructs the quad face-node connectivity by index
arithmetic, so the structured corner grid becomes an unstructured surface mesh.

Read through the NetCDF substrate the vendored GDAL provides, with no declared
``netCDF4`` or ``h5py``. GEOS native grids use ``.nc`` or ``.nc4``; they are reached
through ``read_file``'s NetCDF-suffix fallback (``pyramids-core`` 0.69.0), since a GEOS
grid is not UGRID.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The arrays that make a NetCDF file a GEOS cubed-sphere grid: the corner lattices.
_CORNER_LONS = "corner_lons"
_CORNER_LATS = "corner_lats"
_CENTRE_LONS = "lons"
_CENTRE_LATS = "lats"

# Signature arrays of the sibling ``.nc`` formats GEOS must decline even though they, too,
# describe cells by corner coordinates. GEOS keys on ``corner_lons``/``corner_lats`` --
# which neither SCRIP nor ESMF carries -- and additionally refuses a file holding either
# of those formats' own corner tables, so no file is ever claimed by two drivers.
_SCRIP_CORNERS = frozenset({"grid_corner_lat", "grid_corner_lon"})
_ESMF_ARRAYS = frozenset({"nodeCoords", "elementConn"})


def _group_names(group) -> set[str]:
    """Return the set of root-group MD-array names.

    Args:
        group: An open GDAL multidimensional root group.

    Returns:
        set[str]: The array names in the group.
    """
    return set(group.GetMDArrayNames())


def _is_geos(names: set[str]) -> bool:
    """Report whether a set of array names is a GEOS cubed-sphere grid.

    GEOS is keyed on ``corner_lons`` and ``corner_lats``; a file also carrying SCRIP's or
    ESMF's defining arrays is declined so the corner-coordinate drivers never both claim
    one file.

    Args:
        names: The array names present in the file.

    Returns:
        bool: ``True`` for a GEOS grid.
    """
    if _SCRIP_CORNERS & names or _ESMF_ARRAYS & names:
        return False
    return {_CORNER_LONS, _CORNER_LATS} <= names


class GeosDriver(BaseDriver):
    """Reader for GEOS native cubed-sphere NetCDF.

    The six per-face corner lattices (``corner_lons``/``corner_lats``, shape
    ``(6, ny, nx)``) are ravelled into a node list, and the quad face-node connectivity is
    reconstructed by index arithmetic -- the four corners of each ``(face, j, i)`` cell.
    The optional cell centres (``lons``/``lats``) become the face centres.

    Documented limitations:
        - **Connectivity is computed, not read.** GEOS stores a structured corner lattice
          with no explicit connectivity, so the quad faces are reconstructed by index
          arithmetic. Corners are not deduplicated across the ravel, so a face edge shared
          between cube panels carries two coincident nodes.
        - **Coordinates are carried in the file's degrees.** GEOS writes longitude and
          latitude in degrees, carried verbatim.
        - **No CRS.** The grid is lon/lat on a sphere with no EPSG authority, so ``crs``
          is ``None`` rather than guessed.
        - **Only the topology is read.** The model's data fields are not shipped as
          variables; this driver declares ``read_mesh`` only.

    Example:
        Read a GEOS grid (shown, not run as a doctest -- it needs a real NetCDF file),
        reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("test-c12.native.nc4")
            mesh.mesh.n_face          # the cubed-sphere cells
    """

    name = "geos"
    long_name = "GEOS cubed-sphere grid"
    suffixes = (".nc", ".nc4")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a GEOS cubed-sphere grid, by its corner lattices.

        Keyed on ``corner_lons`` and ``corner_lats`` and declining a file that also
        carries SCRIP's or ESMF's defining arrays, so a plain NetCDF or another model's
        ``.nc`` -- SCRIP, ESMF, MPAS, ICON, 3Di or TUFLOW FV -- is not claimed. Opening a
        non-NetCDF file raises inside GDAL and is reported as "not GEOS" rather than
        propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the corner lattices are present and no sibling format's
                arrays are.
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
        return _is_geos(names)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a GEOS grid into a :class:`ParsedMesh`.

        Args:
            path: The GEOS ``.nc``/``.nc4`` file.
            lazy: Unused -- this driver reads only the topology, which is eager.

        Returns:
            ParsedMesh: The node list from the corner lattices, the computed quad
                connectivity and the cell centres.

        Raises:
            FileFormatNotSupportedError: If a defining corner array is missing.
        """
        return _GeosReader(path).build()


class _GeosReader:
    """One GEOS file: ravel the corner lattices and compute the quad connectivity."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file, build the node list and reconstruct the faces.

        Returns:
            ParsedMesh: The surface mesh with its cell centres.

        Raises:
            FileFormatNotSupportedError: If a defining corner array is missing.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = _group_names(group)
                self._require(names)
                corner_lons = np.asarray(
                    group.OpenMDArray(_CORNER_LONS).ReadAsArray(), np.float64
                )
                corner_lats = np.asarray(
                    group.OpenMDArray(_CORNER_LATS).ReadAsArray(), np.float64
                )
                face_x, face_y = self._face_centres(group, names)
            finally:
                dataset.Close()
        node_x = corner_lons.ravel()
        node_y = corner_lats.ravel()
        faces = self._faces(corner_lons.shape)
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
            global_attributes={"source": "GEOS cubed-sphere grid"},
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
        missing = {_CORNER_LONS, _CORNER_LATS} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a GEOS grid: missing {sorted(missing)}"
            )

    @staticmethod
    def _faces(shape: tuple[int, ...]) -> np.ndarray:
        """Reconstruct the quad face-node connectivity by index arithmetic.

        The corner lattice is ``(n_face_panel, ny, nx)`` and ravels in C-order, so the
        corner at panel ``f``, row ``j``, column ``i`` is node ``f * ny * nx + j * nx +
        i``. Each cell ``(f, j, i)`` for ``j < ny - 1`` and ``i < nx - 1`` is the quad of
        its four surrounding corners, listed so a consistent winding is produced (the
        validator corrects it to counter-clockwise).

        Args:
            shape: The ``(n_panel, ny, nx)`` shape of a corner lattice.

        Returns:
            np.ndarray: The ``(n_panel * (ny - 1) * (nx - 1), 4)`` 0-based quad
                connectivity.
        """
        n_panel, n_y, n_x = shape
        panels = np.arange(n_panel)
        rows = np.arange(n_y - 1)
        cols = np.arange(n_x - 1)
        panel, row, col = np.meshgrid(panels, rows, cols, indexing="ij")
        base = panel * (n_y * n_x) + row * n_x + col
        lower_left = base
        lower_right = base + 1
        upper_right = base + n_x + 1
        upper_left = base + n_x
        return np.stack(
            [
                lower_left.ravel(),
                lower_right.ravel(),
                upper_right.ravel(),
                upper_left.ravel(),
            ],
            axis=1,
        ).astype(np.int64)

    @staticmethod
    def _face_centres(group, names: set[str]):
        """Return the cell centres as face_x/face_y, when present.

        Both ``lons`` and ``lats`` must be present; one without the other yields no
        centres rather than a half-populated pair. The centre lattices are ravelled the
        same way the corner grid is, so the order matches the computed faces.

        Args:
            group: The open root group.
            names: The array names present in the file.

        Returns:
            tuple: ``(face_x, face_y)`` in degrees, or ``(None, None)`` when absent.
        """
        if _CENTRE_LONS not in names or _CENTRE_LATS not in names:
            return None, None
        face_x = np.asarray(group.OpenMDArray(_CENTRE_LONS).ReadAsArray(), np.float64)
        face_y = np.asarray(group.OpenMDArray(_CENTRE_LATS).ReadAsArray(), np.float64)
        return face_x.ravel(), face_y.ravel()
