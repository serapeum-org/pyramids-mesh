"""ICON icosahedral-grid reader (task C7).

ICON (the DWD/MPI weather and climate model) stores its unstructured triangular grid as
NetCDF: cell centres in ``clon``/``clat``, triangle vertices in ``vlon``/``vlat``, and the
1-based ``vertex_of_cell`` table (shape ``nv=3`` x ``ncells``) that names the three
vertices of every cell. ``cell_area`` carries the per-cell area. The driver builds the
surface mesh -- vertices become the nodes, ``vertex_of_cell`` becomes the triangular
face-node connectivity -- and ships ``cell_area`` as a static face variable.

Read through the NetCDF substrate the vendored GDAL provides, with no declared
``netCDF4`` or ``h5py``. ICON grids use ``.nc``; they are reached through ``read_file``'s
NetCDF-suffix fallback (``pyramids-gis`` 0.69.0), since an ICON grid is not UGRID.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# The arrays that make a NetCDF file an ICON grid: the 1-based cell-vertex table and the
# cell-centre longitude. ``vlon`` is required too so a file with cells but no vertices is
# not mistaken for one.
_VERTEX_OF_CELL = "vertex_of_cell"
_CLON = "clon"
_CLAT = "clat"
_VLON = "vlon"
_VLAT = "vlat"
_CELL_AREA = "cell_area"

# MPAS -- the sibling unstructured-grid format that also claims ``.nc`` -- names its
# cell-vertex table ``verticesOnCell`` and declares ``Conventions = MPAS``. Either marker
# means the file is MPAS, not ICON, so ICON declines it rather than both drivers claiming
# the same file.
_MPAS_VERTEX_ARRAY = "verticesOnCell"
_MPAS_CONVENTION = "MPAS"


def _group_names(group) -> set[str]:
    """Return the set of root-group MD-array names."""
    return set(group.GetMDArrayNames())


def _convention(group) -> str:
    """Return the file's ``Conventions`` global attribute, or an empty string."""
    for attribute in group.GetAttributes():
        if attribute.GetName() == "Conventions":
            try:
                return str(attribute.Read())
            except RuntimeError:
                return ""
    return ""


def _is_icon(group) -> bool:
    """Report whether a root group is an ICON grid and not its MPAS look-alike.

    ICON is keyed on ``vertex_of_cell`` together with ``clon`` and ``vlon``; a file
    carrying MPAS's ``verticesOnCell`` array or declaring the MPAS convention is declined
    so the two ``.nc`` drivers never both claim one file.

    Args:
        group: The opened NetCDF root group.

    Returns:
        bool: ``True`` for an ICON grid.
    """
    names = _group_names(group)
    if _MPAS_VERTEX_ARRAY in names:
        return False
    if _MPAS_CONVENTION in _convention(group).upper():
        return False
    return {_VERTEX_OF_CELL, _CLON, _VLON} <= names


class IconDriver(BaseDriver):
    """Reader for ICON icosahedral triangular grids.

    The nodes come from the triangle vertices ``vlon``/``vlat``; the 1-based
    ``vertex_of_cell`` table (``nv=3`` x ``ncells``) is transposed to ``ncells`` x ``3``
    and re-based to 0 to give the triangular face-node connectivity. ``clon``/``clat`` are
    carried as the face centres and ``cell_area`` as a static face variable.

    Documented limitations:
        - **Coordinates are kept in radians.** ICON stores longitude and latitude in
          radians, and this driver preserves them verbatim: node and face-centre
          coordinates come back in radians, not degrees. Converting to degrees is a
          projection concern that belongs in the GIS layer upstream (pyramids), not in a
          file-format driver, so the native unit is carried unchanged.
        - **No CRS.** ICON declares a sphere (``grid_mapping_name = lat_long_on_sphere``,
          ``semi_major_axis`` ~ 6371229 m) but no EPSG authority, and radian coordinates
          do not match a standard degree-based geographic CRS, so ``crs`` is ``None``
          rather than a misleading geographic CRS.
        - **``vertex_of_cell`` is 1-based.** It is resolved to 0-based and the base
          recorded for a writer.
        - **MPAS is declined.** A ``.nc`` carrying MPAS's ``verticesOnCell`` array or the
          MPAS convention is not claimed, so ICON and MPAS never both open one file.
        - Values are eager: an ICON grid is a static mesh, so ``cell_area`` is read at
          parse time rather than behind a loader.

    Example:
        Read an ICON grid (shown, not run as a doctest -- it needs a real NetCDF file),
        reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("icon_grid.nc")
            area = mesh.get("cell_area")   # face-located cell areas
    """

    name = "icon"
    long_name = "ICON icosahedral grid"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is an ICON grid, by its defining arrays.

        Keyed on ``vertex_of_cell`` with ``clon``/``vlon`` so a plain NetCDF, a UGRID
        file or another model's ``.nc`` -- MPAS in particular -- is not claimed; opening a
        non-NetCDF file raises inside GDAL and is reported as "not ICON" rather than
        propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the defining ICON arrays are present and no MPAS marker is.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                return _is_icon(dataset.GetRootGroup())
            finally:
                dataset.Close()

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an ICON grid into a :class:`ParsedMesh`.

        Args:
            path: The ICON ``.nc`` grid.
            lazy: Ignored -- an ICON grid is a static mesh, so ``cell_area`` is read
                eagerly.

        Returns:
            ParsedMesh: The triangular surface mesh, the cell centres and ``cell_area``.

        Raises:
            FileFormatNotSupportedError: If a defining ICON array is missing.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                self._require(path, _group_names(group))
                node_x = np.asarray(group.OpenMDArray(_VLON).ReadAsArray(), np.float64)
                node_y = np.asarray(group.OpenMDArray(_VLAT).ReadAsArray(), np.float64)
                # vertex_of_cell is (nv=3, ncells); transpose to (ncells, 3) so each row
                # is one triangle. It is 1-based, re-based to 0 by validation.
                vertex_of_cell = np.asarray(
                    group.OpenMDArray(_VERTEX_OF_CELL).ReadAsArray(), np.int64
                )
                faces = np.ascontiguousarray(vertex_of_cell.T)
                face_x, face_y = self._face_centres(group, _group_names(group))
                variables = self._variables(group, _group_names(group), len(faces))
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # vertex_of_cell is 1-based; validation re-bases it and records the base.
            start_index=1,
            original_start_index=1,
            face_x=face_x,
            face_y=face_y,
            crs=None,
            variables=variables,
            global_attributes={"source": "ICON grid"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    @staticmethod
    def _require(path: str, names: set[str]) -> None:
        """Refuse a file missing a defining ICON array.

        Raises:
            FileFormatNotSupportedError: If ``vertex_of_cell``, ``vlon`` or ``vlat`` is
                absent.
        """
        missing = {_VERTEX_OF_CELL, _VLON, _VLAT} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an ICON grid: missing {sorted(missing)}"
            )

    @staticmethod
    def _face_centres(group, names: set[str]):
        """Return the cell centres as face_x/face_y, when present.

        Both ``clon`` and ``clat`` must be present; one without the other yields no
        centres rather than a half-populated pair.
        """
        if _CLON not in names or _CLAT not in names:
            return None, None
        face_x = np.asarray(group.OpenMDArray(_CLON).ReadAsArray(), np.float64)
        face_y = np.asarray(group.OpenMDArray(_CLAT).ReadAsArray(), np.float64)
        return face_x, face_y

    @staticmethod
    def _variables(group, names: set[str], n_face: int) -> dict[str, ParsedVariable]:
        """Build the static ``cell_area`` face variable, when present and well-shaped.

        A ``cell_area`` whose length is not the face count is left out rather than
        mislabelled onto the faces.

        Returns:
            dict[str, ParsedVariable]: ``{"cell_area": ...}`` when the array is present
                and face-length, else empty.
        """
        if _CELL_AREA not in names:
            return {}
        values = np.asarray(group.OpenMDArray(_CELL_AREA).ReadAsArray(), np.float64)
        if values.shape != (n_face,):
            return {}
        return {
            _CELL_AREA: ParsedVariable(
                location="face",
                shape=(n_face,),
                values=values,
                standard_name="cell_area",
                units="m2",
            )
        }
