"""TUFLOW FV results reader -- the 2D slice (task N3).

TUFLOW FV writes NetCDF with an explicit node list and cell connectivity, cell-centre
coordinates and per-cell results over time. This driver reads the **2D slice**: the
surface mesh and its per-cell variables. A genuinely layered 3D file (more 3D cells than
2D cells) is refused, because its vertical levels need a layered mesh model this package
does not have yet (design section 21.1).

Read through the NetCDF substrate the vendored GDAL provides, with no declared
``netCDF4`` or ``h5py``. TUFLOW FV results use ``.nc``; they are reached through
``read_file``'s NetCDF-suffix fallback (``pyramids-core`` 0.69.0), since a TUFLOW FV file
is not UGRID.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh.drivers.threedi import _ThreeDiLoader  # reopen-per-read lazy loader
from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

_NODE_X = "node_X"
_NODE_Y = "node_Y"
_NODE_Z = "node_Zb"
_CELL_NODE = "cell_node"
_CELL_X = "cell_X"
_CELL_Y = "cell_Y"
_TIME = "ResTime"
# The time *dimension* is "Time"; the time *coordinate array* is "ResTime". A result's
# first axis is the dimension, so has-time is decided on the dimension name.
_TIME_DIM = "Time"
_STAT = "stat"

_CELLS_2D = "NumCells2D"
_CELLS_3D = "NumCells3D"

# Arrays that are topology, geometry or indexing rather than results; not shipped as
# variables.
_STRUCTURAL = frozenset(
    {
        _NODE_X,
        _NODE_Y,
        _NODE_Z,
        _CELL_NODE,
        _CELL_X,
        _CELL_Y,
        _TIME,
        "cell_Zb",
        "cell_A",
        "cell_Nvert",
        "idx2",
        "idx3",
        "layerface_Z",
        "NL",
    }
)


class TuflowFvDriver(BaseDriver):
    """Reader for TUFLOW FV results -- the 2D slice.

    The surface mesh comes from ``node_X``/``node_Y`` and the 1-based ``cell_node``
    connectivity (triangles padded with ``0`` in a quad-width table); ``node_Zb`` becomes
    a ``node_z`` variable and ``cell_X``/``cell_Y`` the face centres. Each per-cell result
    array becomes a face-located variable over the ``ResTime`` axis, and the ``stat``
    wet/dry status is one such face variable, carried rather than folded into the values.

    Documented limitations:
        - **A layered 3D file is refused.** When the file has more 3D cells than 2D cells
          (``NumCells3D`` > ``NumCells2D``), its variables are stacked over vertical
          levels this package does not model yet (design section 21.1); it is refused
          rather than flattened. A single-layer file (``NumCells3D`` == ``NumCells2D``) is
          a 2D mesh and reads normally, its ``NumCells3D`` variables mapped to faces.
        - **``cell_node`` is 1-based with ``0`` padding.** A ``0`` entry marks a triangle
          in the quad-width table and becomes ``-1`` fill; the base is recorded for a
          writer.
        - **``stat`` is carried as a face-located variable**, not folded into the values
          as NaN (design section 21.2): the per-step wet/dry status stays inspectable.
        - Values are lazy: the loader reopens the file per read, closing over a path.
        - There is no CRS in the format read here, so ``crs`` is ``None``.

    Example:
        Read a TUFLOW FV result (shown, not run as a doctest -- it needs a real NetCDF
        file), reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("results.nc")
            depth = mesh.get("D")          # face-located, over time
            wet = mesh.get("stat")         # per-step wet/dry status
    """

    name = "tuflowfv"
    long_name = "TUFLOW FV results (2D)"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is TUFLOW FV, by its node list plus cell connectivity.

        Keyed on ``cell_node`` with ``node_X`` so a plain NetCDF or another model's
        ``.nc`` -- including 3Di, which has no ``cell_node`` -- is not claimed.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the defining arrays are present.
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
        return {_CELL_NODE, _NODE_X, _NODE_Y} <= names

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a TUFLOW FV file's 2D slice into a :class:`ParsedMesh`.

        Args:
            path: The TUFLOW FV ``.nc`` file.
            lazy: When ``True``, leave each variable behind a loader.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` and the per-cell variables.

        Raises:
            FileFormatNotSupportedError: If the file is missing a defining array or is a
                layered 3D file.
        """
        return _TuflowFvReader(path, lazy=lazy).build()


class _TuflowFvReader:
    """One TUFLOW FV file: read the 2D surface mesh and its per-cell variables."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Open the file, build the 2D mesh, and gather the per-cell variables.

        Returns:
            ParsedMesh: The 2D surface mesh with its face variables.

        Raises:
            FileFormatNotSupportedError: On a missing array or a layered 3D file.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                self._require(names)
                dim_sizes = self._dim_sizes(group)
                self._guard_3d(dim_sizes)
                n_face = self._n_face(dim_sizes)
                node_x = np.asarray(
                    group.OpenMDArray(_NODE_X).ReadAsArray(), np.float64
                )
                node_y = np.asarray(
                    group.OpenMDArray(_NODE_Y).ReadAsArray(), np.float64
                )
                faces = self._faces(group)
                node_z = self._node_z(group, names)
                face_x, face_y = self._face_centres(group, names)
                times = self._times(group, names)
                variables = self._variables(group, names, dim_sizes, times, n_face)
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # cell_node is 1-based; resolved to 0-based here, base recorded for a writer.
            start_index=0,
            original_start_index=1,
            node_z=node_z,
            face_x=face_x,
            face_y=face_y,
            crs=None,
            variables=variables,
            global_attributes={"source": "TUFLOW FV results"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self, names: set[str]) -> None:
        """Refuse a file missing a defining array."""
        missing = {_CELL_NODE, _NODE_X, _NODE_Y} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a TUFLOW FV result: missing {sorted(missing)}"
            )

    @staticmethod
    def _dim_sizes(group) -> dict[str, int]:
        """Return the sizes of the cell dimensions, read from the connectivity array."""
        sizes: dict[str, int] = {}
        for name in group.GetMDArrayNames():
            for dimension in group.OpenMDArray(name).GetDimensions():
                sizes[dimension.GetName()] = dimension.GetSize()
        return sizes

    def _n_face(self, dim_sizes: dict[str, int]) -> int:
        """Return the 2D-cell count, or refuse a file that lacks that dimension.

        The sniff only checks for the defining *arrays*; a file missing the
        ``NumCells2D`` *dimension* would otherwise crash with a bare ``KeyError``.

        Raises:
            FileFormatNotSupportedError: If the ``NumCells2D`` dimension is absent.
        """
        n_face = dim_sizes.get(_CELLS_2D)
        if n_face is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a TUFLOW FV result: no {_CELLS_2D!r} dimension"
            )
        return n_face

    def _guard_3d(self, dim_sizes: dict[str, int]) -> None:
        """Refuse a layered 3D file -- more 3D cells than 2D cells.

        Raises:
            FileFormatNotSupportedError: If ``NumCells3D`` exceeds ``NumCells2D``, a file
                stacked over vertical levels the 2D slice cannot represent.
        """
        n2 = dim_sizes.get(_CELLS_2D)
        n3 = dim_sizes.get(_CELLS_3D)
        if n2 is not None and n3 is not None and n3 > n2:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a layered 3D TUFLOW FV file ({n3} 3D cells over {n2} "
                f"2D cells); its vertical levels are not modelled yet, so only a "
                f"single-layer (2D) file is read"
            )

    def _faces(self, group) -> np.ndarray:
        """Resolve the 1-based ``cell_node`` table to 0-based connectivity.

        A ``0`` entry is the format's padding for a triangle in the quad-width table and
        becomes ``-1`` fill; every other id drops to 0-based.
        """
        cell_node = np.asarray(group.OpenMDArray(_CELL_NODE).ReadAsArray(), np.int64)
        faces = np.where(cell_node == 0, -1, cell_node - 1)
        return faces

    def _node_z(self, group, names: set[str]):
        """Read node elevation as node_z, when present."""
        if _NODE_Z not in names:
            return None
        return np.asarray(group.OpenMDArray(_NODE_Z).ReadAsArray(), np.float64)

    def _face_centres(self, group, names: set[str]):
        """Return the cell centres as face_x/face_y, when present."""
        if _CELL_X not in names or _CELL_Y not in names:
            return None, None
        face_x = np.asarray(group.OpenMDArray(_CELL_X).ReadAsArray(), np.float64)
        face_y = np.asarray(group.OpenMDArray(_CELL_Y).ReadAsArray(), np.float64)
        return face_x, face_y

    def _times(self, group, names: set[str]):
        """Read the ``ResTime`` coordinate, or ``None`` when absent."""
        if _TIME not in names:
            return None
        return np.asarray(group.OpenMDArray(_TIME).ReadAsArray(), np.float64)

    def _variables(
        self, group, names: set[str], dim_sizes: dict[str, int], times, n_face: int
    ) -> dict[str, ParsedVariable]:
        """Build a face variable for each per-cell result array.

        An array whose last dimension is the 2D- or (single-layer) 3D-cell dimension is a
        face variable; structural and node arrays are skipped. ``stat`` rides along as an
        ordinary face variable -- the wet/dry status, carried not folded.
        """
        variables: dict[str, ParsedVariable] = {}
        for name in sorted(names):
            if name in _STRUCTURAL or name.startswith("node_"):
                continue
            array = group.OpenMDArray(name)
            dim_names = [d.GetName() for d in array.GetDimensions()]
            if not dim_names or dim_names[-1] not in (_CELLS_2D, _CELLS_3D):
                continue
            if dim_sizes.get(dim_names[-1]) != n_face:
                continue
            shape = tuple(d.GetSize() for d in array.GetDimensions())
            has_time = times is not None and dim_names[0] == _TIME_DIM
            loader = _ThreeDiLoader(self.path, name)
            variables[name] = ParsedVariable(
                location="face",
                shape=shape,
                loader=loader if self.lazy else None,
                values=None if self.lazy else loader(),
                time_values=times if has_time else None,
            )
        return variables
