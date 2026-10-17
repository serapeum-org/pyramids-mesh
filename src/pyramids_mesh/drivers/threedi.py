"""3Di results reader (task N2).

3Di writes its results as NetCDF, but not as a node list plus connectivity: each 2D cell
is given by its corner coordinates directly (``Mesh2DContour_x``/``Mesh2DContour_y``). The
real work is deduplicating those repeated corners into a shared node table and emitting
face-node connectivity into it.

It is a hydrodynamic model's native format, so it belongs here (design section 1), read
through the NetCDF substrate the vendored GDAL already provides -- the same
multidimensional route the other NetCDF drivers use, with no declared ``netCDF4`` or
``h5py``.

3Di results use the ``.nc`` suffix. ``UgridDataset.read_file`` tries its built-in UGRID
reader first and, finding no UGRID topology, falls back to a driver claiming the suffix
(``pyramids-gis`` 0.69.0's dispatch), which is how this driver is reached. A successful
UGRID read still pays nothing.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import warnings

import numpy as np

from osgeo import gdal, osr

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# The arrays that make a NetCDF file a 3Di result: the 2D cell contours.
_CONTOUR_X = "Mesh2DContour_x"
_CONTOUR_Y = "Mesh2DContour_y"
_FACE_XCC = "Mesh2DFace_xcc"
_FACE_YCC = "Mesh2DFace_ycc"
_TIME = "time"

# The 2D cell dimension (3Di calls a cell a "node"); a result array on it is face-located.
_CELL_DIM = "nMesh2D_nodes"

# 1D-network arrays; their presence means a coupled 1D model this package cannot represent
# alongside the 2D mesh, so it is warned about rather than silently dropped.
_MESH1D_PREFIX = "Mesh1D"

# Corner coordinates are rounded to this many decimals before hashing: float equality on a
# shared corner is not reliable, and a rounded key makes the dedup stable.
_DEDUP_DECIMALS = 6

_CRS_ATTRS = ("crs_wkt", "spatial_ref", "esri_pe_string", "projection", "epsg_code")


class ThreeDiDriver(BaseDriver):
    """Reader for 3Di result NetCDF.

    3Di stores 2D cell *contours* -- each cell's corner coordinates -- rather than a node
    list and connectivity. The driver deduplicates the corners into a shared node table
    and builds face-node connectivity into it, so a 3Di result becomes an ordinary
    :class:`~pyramids.netcdf.ugrid.UgridDataset`. Per-cell result arrays become
    face-located variables over time.

    Documented limitations:
        - **Corner dedup uses a rounded key, not float equality.** Shared corners are
          matched after rounding to a fixed tolerance; raw float comparison would leave
          duplicate nodes and split edges. The unique nodes are sorted, so the same file
          gives the same node order every run -- downstream indices stay stable.
        - **A coupled 1D network is reported, not represented.** A 3Di file with
          ``Mesh1D*`` arrays carries a 1D model coupled to the 2D mesh; pyramids has no
          coupling relation (design section 10), so the 2D mesh is returned and the 1D
          part is announced in a warning naming ``Network1d`` rather than dropped
          silently.
        - **Flow-line variables are not shipped.** A result on the ``nMesh2D_lines``
          dimension lives on flow links between cells, which the mesh model has no place
          for; only per-cell (face) variables are read.
        - Values are lazy: the loader reopens the file per read and seeks to its array,
          closing over a path, never an open handle (design section 7).
        - CRS comes only from a validated grid-mapping attribute; none is guessed.

    Example:
        Read a 3Di result. It is reached through ``read_file``'s NetCDF fallback because
        its ``.nc`` is not UGRID; shown rather than run as a doctest since it needs a real
        NetCDF file::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("results_3di.nc")   # cells -> shared nodes
            water_level = mesh.get("Mesh2D_s1")               # face-located, over time
    """

    name = "3di"
    long_name = "3Di results"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a 3Di result, by its defining contour arrays.

        Keyed on ``Mesh2DContour_x`` so a plain NetCDF or another model's ``.nc`` is not
        claimed; opening a non-NetCDF file raises inside GDAL and is reported as "not
        3Di" rather than propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the contour arrays are present.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                names = set(dataset.GetRootGroup().GetMDArrayNames())
            finally:
                dataset.Close()
        return {_CONTOUR_X, _CONTOUR_Y} <= names

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a 3Di file into a :class:`ParsedMesh`.

        Args:
            path: The 3Di ``.nc`` file.
            lazy: When ``True``, leave each variable behind a loader.

        Returns:
            ParsedMesh: Nodes deduplicated from the cell contours, face connectivity, the
                cell centres and the per-cell variables.

        Raises:
            FileFormatNotSupportedError: If the contour arrays are missing.
        """
        return _ThreeDiReader(path, lazy=lazy).build()


class _ThreeDiLoader:
    """A lazy loader for one 3Di face variable, reopening the file per read."""

    def __init__(self, path: str, array_name: str) -> None:
        self.path = path
        self.array_name = array_name

    def __call__(self) -> np.ndarray:
        """Reopen the file and read this variable's array."""
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                array = dataset.GetRootGroup().OpenMDArray(self.array_name)
                return np.asarray(array.ReadAsArray())
            finally:
                dataset.Close()


class _ThreeDiReader:
    """One 3Di file: dedup the contours and build the parsed mesh and variables."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Open the file, dedup contours to nodes, and assemble the mesh.

        Returns:
            ParsedMesh: The deduplicated 2D mesh with its per-cell variables.

        Raises:
            FileFormatNotSupportedError: If the contour arrays are missing.
        """
        with (
            gdal.ExceptionMgr(useExceptions=True),
            osr.ExceptionMgr(useExceptions=True),
        ):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                self._require(names)
                cx = np.asarray(group.OpenMDArray(_CONTOUR_X).ReadAsArray(), np.float64)
                cy = np.asarray(group.OpenMDArray(_CONTOUR_Y).ReadAsArray(), np.float64)
                node_x, node_y, faces = self._dedup(cx, cy)
                face_x, face_y = self._face_centres(group, names)
                times = self._times(group, names)
                variables = self._variables(group, names, times)
                crs = self._crs(group)
                self._warn_mesh1d(names)
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
            crs=crs,
            variables=variables,
            global_attributes={"source": "3Di results"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self, names: set[str]) -> None:
        """Refuse a file missing the contour arrays."""
        missing = {_CONTOUR_X, _CONTOUR_Y} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a 3Di result: missing {sorted(missing)}"
            )

    def _dedup(
        self, cx: np.ndarray, cy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deduplicate cell corners into a shared node table and face connectivity.

        Corners are matched on a rounded key so shared corners collapse to one node, and
        the unique nodes are sorted so the node order is deterministic. A corner that is
        not finite (a cell with fewer corners than the contour width) becomes ``-1`` fill.
        """
        n_cell, n_corner = cx.shape
        rounded = np.stack(
            [np.round(cx, _DEDUP_DECIMALS), np.round(cy, _DEDUP_DECIMALS)], axis=-1
        )
        finite = np.isfinite(cx) & np.isfinite(cy)
        unique, inverse = np.unique(
            rounded[finite].reshape(-1, 2), axis=0, return_inverse=True
        )
        faces = np.full((n_cell, n_corner), -1, dtype=np.int64)
        flat_index = np.ravel(np.asarray(inverse))
        cursor = 0
        for cell in range(n_cell):
            for corner in range(n_corner):
                if finite[cell, corner]:
                    faces[cell, corner] = int(flat_index[cursor])
                    cursor += 1
        node_x = unique[:, 0].astype(np.float64)
        node_y = unique[:, 1].astype(np.float64)
        return node_x, node_y, faces

    def _face_centres(self, group, names: set[str]):
        """Return the cell-centre coordinates as face_x/face_y, when present."""
        if _FACE_XCC not in names or _FACE_YCC not in names:
            return None, None
        face_x = np.asarray(group.OpenMDArray(_FACE_XCC).ReadAsArray(), np.float64)
        face_y = np.asarray(group.OpenMDArray(_FACE_YCC).ReadAsArray(), np.float64)
        return face_x, face_y

    def _times(self, group, names: set[str]):
        """Read the time coordinate, or ``None`` when the file has none."""
        if _TIME not in names:
            return None
        return np.asarray(group.OpenMDArray(_TIME).ReadAsArray(), np.float64)

    def _variables(self, group, names: set[str], times) -> dict[str, ParsedVariable]:
        """Build a lazy face variable for each per-cell result array.

        A ``Mesh2D_*`` array dimensioned on the 2D cell dimension is face-located;
        everything else (flow-line arrays, the 1D family, coordinates) is left out.
        """
        variables: dict[str, ParsedVariable] = {}
        for name in sorted(names):
            if not name.startswith("Mesh2D_"):
                continue
            array = group.OpenMDArray(name)
            dim_names = [d.GetName() for d in array.GetDimensions()]
            if _CELL_DIM not in dim_names:
                continue
            shape = tuple(d.GetSize() for d in array.GetDimensions())
            has_time = times is not None and dim_names[0] == _TIME
            loader = _ThreeDiLoader(self.path, name)
            variables[name] = ParsedVariable(
                location="face",
                shape=shape,
                loader=loader if self.lazy else None,
                values=None if self.lazy else loader(),
                time_values=times if has_time else None,
            )
        return variables

    def _crs(self, group) -> str | int | None:
        """Return a validated CRS from a grid-mapping attribute, or ``None``."""
        available = {a.GetName() for a in group.GetAttributes()}
        for name in _CRS_ATTRS:
            if name not in available:
                continue
            raw = group.GetAttribute(name).Read()
            value = raw[0] if isinstance(raw, list) else raw
            if isinstance(value, (int, np.integer)):
                reference = osr.SpatialReference()
                try:
                    reference.ImportFromEPSG(int(value))
                except RuntimeError:
                    continue
                return int(value)
            text = str(value).strip()
            if not text:
                continue
            reference = osr.SpatialReference()
            try:
                reference.ImportFromWkt(text)
                valid = reference.Validate() == 0
            except RuntimeError:
                continue
            if valid:
                return text
        return None

    def _warn_mesh1d(self, names: set[str]) -> None:
        """Warn that a coupled 1D network is present but not represented."""
        if any(name.startswith(_MESH1D_PREFIX) for name in names):
            warnings.warn(
                f"{self.path!r} also carries a coupled 1D network (Mesh1D*), which "
                f"pyramids cannot represent alongside the 2D mesh: there is no Network1d "
                f"coupling relation, so the 1D part is not in this result",
                stacklevel=4,
            )
