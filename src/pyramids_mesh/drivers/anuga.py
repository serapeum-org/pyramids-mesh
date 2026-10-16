"""ANUGA SWW driver (task N1).

SWW is ANUGA's NetCDF output: node coordinates, triangle connectivity (``volumes``) and
per-timestep node results. It is a hydrodynamic model's native format, so it belongs here
(design section 1), and it needs no dependency beyond the NetCDF substrate the vendored
GDAL already provides -- the same multidimensional route the HDF5 drivers use, activated
by ``import pyramids``, with no declared ``netCDF4`` or ``h5py``.

``.sww`` is not one of pyramids' NetCDF suffixes (``.nc`` and the rest), so a plain
``UgridDataset.read_file("model.sww")`` reaches this driver through the registry. A SWW
file that happened to be named ``.nc`` would not -- that is the upstream dispatch
limitation tracked for the 3Di and TUFLOW FV drivers, not a concern for ``.sww``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import warnings

import numpy as np

from osgeo import gdal, osr

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# GDAL exceptions are enabled per operation with ``gdal.ExceptionMgr`` rather than
# globally at import: importing a driver must not mutate the interpreter's GDAL
# state for everything else, and the try/except RuntimeError guards here need
# exceptions on only while they run.

# The arrays that make a NetCDF file a SWW: a triangle table plus node coordinates.
_VOLUMES = "volumes"
_X = "x"
_Y = "y"

# Elevation is carried under either name; whichever is present becomes node_z, and both
# are then kept out of the result-variable loop so neither reappears as a plain variable.
_ELEVATION_NAMES = ("elevation", "z")
_TIME = "time"

# Arrays that are coordinates or topology rather than results.
_STRUCTURAL = frozenset({_VOLUMES, _X, _Y, _TIME, *_ELEVATION_NAMES})

# Root attributes that may carry a CRS. SWW usually has none; a value is used only when
# the vendored PROJ validates it, never guessed from a UTM zone number.
_CRS_ATTRS = ("crs_wkt", "spatial_ref", "esri_pe_string", "projection")


class SwwDriver(BaseDriver):
    """Reader for the ANUGA SWW format.

    NetCDF carrying a triangular mesh (``x``, ``y``, ``volumes``), node elevation and
    per-timestep node results. The driver declares ``read_mesh``, ``read_data`` and
    ``lazy_data``: the topology is read eagerly to build the mesh, and every variable --
    the elevation included -- is left behind a loader that reopens the file and reads its
    one array.

    Documented limitations:
        - **The connectivity base is asserted, not assumed.** ANUGA is 0-based, and an
          off-by-one shifts every face by one node while still producing a
          plausible-looking mesh. The base is taken from ``volumes.min()`` and must be
          ``0`` or ``1``; anything else is refused rather than read wrong.
        - **A variable's location is resolved by its NetCDF dimension name, not its
          length.** In a tiny file the node and triangle counts can coincide, so a
          variable dimensioned on the points dimension is node-located and one on the
          volumes dimension is face-located -- decided by name, where length would be
          ambiguous.
        - **A variable on neither the point nor the volume dimension is skipped**, with a
          warning naming it. The mesh model has nowhere to put a value indexed by
          something other than nodes or faces.
        - ``stage`` is carried as the file writes it -- absolute water level, not depth.
          It is not renamed or turned into a derived quantity.
        - **CRS comes only from a validated attribute.** SWW usually stores a zone or
          projection label rather than a full CRS; a value is used only when the vendored
          PROJ accepts it, and is otherwise left ``None`` rather than guessed from a UTM
          zone.
        - Node elevation becomes a ``node_z`` variable, never a coordinate (design
          section 6).

    Examples:
        - Write a small SWW and read it back. The triangle table comes through as the
          face connectivity, 0-based:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.anuga import (
            ...     SwwDriver,
            ...     build_minimal_fixture_sww,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.sww")
            ...     build_minimal_fixture_sww(path)
            ...     dataset = SwwDriver().read(path)
            ...     connectivity = dataset.mesh.face_node_connectivity.data.tolist()
            ...     names = sorted(dataset.data_variable_names)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> connectivity
            [[0, 1, 2], [0, 2, 3]]
            >>> names
            ['node_z', 'stage']

        - Elevation is a node variable and ``stage`` keeps its time axis; both are
          deferred until read, so the values are fetched inside the temporary directory:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.anuga import (
            ...     SwwDriver,
            ...     build_minimal_fixture_sww,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.sww")
            ...     build_minimal_fixture_sww(path)
            ...     dataset = SwwDriver().read(path)
            ...     stage = dataset.get("stage")
            ...     location, shape = stage.location, stage.shape
            ...     first_step = stage.data[0].tolist()
            ...     elevation = dataset.get("node_z").location
            >>> location, shape
            ('node', (2, 4))
            >>> first_step
            [0.5, 0.25, 0.75, 0.125]
            >>> elevation
            'node'
    """

    name = "sww"
    long_name = "ANUGA SWW"
    suffixes = (".sww",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a SWW, by the presence of its defining arrays.

        A SWW is a NetCDF file carrying ``volumes`` plus ``x`` and ``y``. Opening a file
        that is not NetCDF at all raises inside GDAL, which is caught here and reported
        as "not a SWW" rather than allowed to propagate.

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
        return {_VOLUMES, _X, _Y} <= names

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a SWW file into a :class:`ParsedMesh`.

        Args:
            path: The SWW file to read.
            lazy: When ``True``, leave each variable behind a loader; when ``False``,
                read every value now.

        Returns:
            ParsedMesh: Nodes, triangle faces and the node-located variables.

        Raises:
            FileFormatNotSupportedError: If the file is missing a defining array, or its
                connectivity base is neither 0 nor 1.
        """
        return _SwwReader(path, lazy=lazy).build()


def build_minimal_fixture_sww(path: str) -> None:
    """Write the smallest valid SWW file, for the cross-driver invariant harness.

    The unit square split into two counter-clockwise triangles, with node elevation and
    one time-varying node variable (``stage``) over two steps. Named per driver because
    the ``anuga`` module could hold more than one.

    Args:
        path: Where to write the file.

    Examples:
        - The file reads back as four nodes, two faces and two variables:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.anuga import (
            ...     SwwDriver,
            ...     build_minimal_fixture_sww,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.sww")
            ...     build_minimal_fixture_sww(path)
            ...     dataset = SwwDriver().read(path)
            >>> dataset.mesh.n_node
            4
            >>> sorted(dataset.data_variable_names)
            ['node_z', 'stage']
    """
    write_sww(
        path,
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        volumes=[[0, 1, 2], [0, 2, 3]],
        elevation=[2.0, 2.5, 3.0, 3.5],
        times=[0.0, 3600.0],
        node_variables={
            "stage": [[0.5, 0.25, 0.75, 0.125], [1.5, 1.25, 1.75, 1.125]],
        },
    )


def write_sww(
    path: str,
    *,
    node_x: list[float],
    node_y: list[float],
    volumes: list[list[int]],
    elevation: list[float] | None = None,
    times: list[float] | None = None,
    node_variables: dict[str, list] | None = None,
    face_variables: dict[str, list] | None = None,
    foreign_variables: dict[str, list] | None = None,
    crs_wkt: str | None = None,
    elevation_name: str = "elevation",
) -> None:
    """Write a SWW file for a fixture, with the variants the tests need.

    On the module surface as the only writer of this format in the package: the test
    variants (a CRS attribute, a 1-based connectivity, a face-located variable, a static
    node variable) all come from its keyword arguments, so one implementation of the
    layout is exercised.

    Args:
        path: Where to write the file.
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        volumes: Triangle connectivity, one ``(v0, v1, v2)`` row per triangle.
        elevation: Node elevation, or ``None`` to omit it.
        times: The time coordinate, or ``None`` for a frame with no time dimension.
        node_variables: Per-node arrays, each shaped ``(n_time, n_node)`` when ``times``
            is given or ``(n_node,)`` when it is not.
        face_variables: Per-triangle arrays, shaped against the volumes dimension.
        foreign_variables: 1-D arrays each written on its own fresh dimension, for
            testing the skip path -- a variable the mesh model cannot place.
        crs_wkt: A WKT string to write as a ``crs_wkt`` root attribute, or ``None``.
        elevation_name: Which name to store elevation under (``"elevation"`` or ``"z"``).
    """
    n_node = len(node_x)
    n_vol = len(volumes)
    n_vert = len(volumes[0])
    driver = gdal.GetDriverByName("netCDF")
    dataset = driver.CreateMultiDimensional(path)
    group = dataset.GetRootGroup()
    dims = {
        "number_of_points": group.CreateDimension("number_of_points", "", "", n_node),
        "number_of_volumes": group.CreateDimension("number_of_volumes", "", "", n_vol),
        "number_of_vertices": group.CreateDimension(
            "number_of_vertices", "", "", n_vert
        ),
    }
    if times is not None:
        dims["number_of_timesteps"] = group.CreateDimension(
            "number_of_timesteps", "", "", len(times)
        )
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    f32 = gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    i32 = gdal.ExtendedDataType.Create(gdal.GDT_Int32)

    def write(name, dim_names, datatype, values, dtype):
        array = group.CreateMDArray(name, [dims[d] for d in dim_names], datatype)
        array.Write(np.asarray(values, dtype=dtype))

    write(_X, ["number_of_points"], f64, node_x, np.float64)
    write(_Y, ["number_of_points"], f64, node_y, np.float64)
    write(
        _VOLUMES,
        ["number_of_volumes", "number_of_vertices"],
        i32,
        volumes,
        np.int32,
    )
    if elevation is not None:
        write(elevation_name, ["number_of_points"], f64, elevation, np.float64)
    if times is not None:
        write(_TIME, ["number_of_timesteps"], f64, times, np.float64)
    for name, values in (node_variables or {}).items():
        dim_names = (
            ["number_of_timesteps", "number_of_points"]
            if times is not None
            else ["number_of_points"]
        )
        write(name, dim_names, f32, values, np.float32)
    for name, values in (face_variables or {}).items():
        dim_names = (
            ["number_of_timesteps", "number_of_volumes"]
            if times is not None
            else ["number_of_volumes"]
        )
        write(name, dim_names, f32, values, np.float32)
    for position, (name, values) in enumerate(
        sorted((foreign_variables or {}).items())
    ):
        column = np.asarray(values, dtype=np.float64)
        foreign_dim = group.CreateDimension(
            f"foreign_dim_{position}", "", "", len(column)
        )
        array = group.CreateMDArray(name, [foreign_dim], f64)
        array.Write(column)
    if crs_wkt is not None:
        attribute = group.CreateAttribute(
            "crs_wkt", [], gdal.ExtendedDataType.CreateString()
        )
        attribute.Write(crs_wkt)
    dataset.FlushCache()
    # `Close()`, not `del`: the vendored netCDF driver does not release the file on
    # garbage collection alone, so a same-process reopen fails without this.
    dataset.Close()


class _SwwLoader:
    """A lazy loader for one SWW array.

    Closes over the path and the array name, never an open handle (design section 7):
    reading it reopens the file, reads the one array, and closes.
    """

    def __init__(self, path: str, array_name: str) -> None:
        self.path = path
        self.array_name = array_name

    def __call__(self) -> np.ndarray:
        """Reopen the file and read this one array."""
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                array = dataset.GetRootGroup().OpenMDArray(self.array_name)
                return np.asarray(array.ReadAsArray())
            finally:
                dataset.Close()


class _SwwReader:
    """One SWW file, read topology-first with every variable left for a loader."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Open the file, read the topology, and defer every variable.

        Returns:
            ParsedMesh: Nodes, triangle faces and the node/face variables.

        Raises:
            FileFormatNotSupportedError: On a missing defining array or an impossible
                connectivity base.
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
                node_x = np.asarray(
                    group.OpenMDArray(_X).ReadAsArray(), dtype=np.float64
                )
                node_y = np.asarray(
                    group.OpenMDArray(_Y).ReadAsArray(), dtype=np.float64
                )
                volumes_array = group.OpenMDArray(_VOLUMES)
                faces, base = self._faces(volumes_array, len(node_x))
                point_dim = group.OpenMDArray(_X).GetDimensions()[0].GetName()
                volume_dim = volumes_array.GetDimensions()[0].GetName()
                time_dim = self._time_dim(group, names)
                crs = self._crs(group)
                variables = self._variables(
                    group, names, point_dim, volume_dim, time_dim
                )
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=base,
            original_start_index=base,
            crs=crs,
            variables=variables,
            global_attributes={"source": "ANUGA SWW"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self, names: set[str]) -> None:
        """Refuse a file missing any of the defining arrays."""
        missing = {_VOLUMES, _X, _Y} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a SWW file: missing {sorted(missing)}"
            )

    def _faces(self, volumes_array, n_node: int) -> tuple[np.ndarray, int]:
        """Read the triangle table and determine its 0- or 1-based numbering.

        Raises:
            FileFormatNotSupportedError: If the base is neither 0 nor 1, or an index
                falls outside the node range -- either a corrupt file or the wrong base.
        """
        faces = np.asarray(volumes_array.ReadAsArray(), dtype=np.int64)
        base = int(faces.min())
        if base not in (0, 1):
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a volumes table based at {base}, which is neither "
                f"0 (ANUGA's convention) nor 1; the numbering cannot be trusted"
            )
        if int(faces.max()) - base > n_node - 1:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a volumes entry {int(faces.max())} that no node "
                f"index reaches for {n_node} nodes at base {base}"
            )
        return faces, base

    @staticmethod
    def _time_dim(group, names: set[str]) -> str | None:
        """Return the time coordinate's own dimension name, or ``None`` if there is none.

        Derived from the ``time`` array rather than assumed, for the same reason the
        point and volume dimensions are: a variable's time axis is recognised by the
        dimension the file's own time coordinate uses, not by a hardcoded literal.
        """
        if _TIME not in names:
            return None
        dimensions = group.OpenMDArray(_TIME).GetDimensions()
        return dimensions[0].GetName() if dimensions else None

    def _crs(self, group) -> str | None:
        """Return a validated CRS from a root attribute, or ``None``.

        SWW usually stores a zone or projection label rather than a full CRS, so a value
        is used only when the vendored PROJ validates it.
        """
        available = {attribute.GetName() for attribute in group.GetAttributes()}
        for name in _CRS_ATTRS:
            if name not in available:
                continue
            wkt = group.GetAttribute(name).ReadAsString()
            if not wkt:
                continue
            reference = osr.SpatialReference()
            try:
                reference.ImportFromWkt(wkt)
            except RuntimeError:
                continue
            if reference.Validate() == 0:
                return wkt
        return None

    def _variables(
        self,
        group,
        names: set[str],
        point_dim: str,
        volume_dim: str,
        time_dim: str | None,
    ) -> dict[str, ParsedVariable]:
        """Build a lazy variable for elevation and every result array.

        Elevation is emitted as a node-located ``node_z`` variable, and every other
        non-structural array is placed by its dimension names. A variable on neither the
        point nor the volume dimension is skipped with a warning -- the elevation
        included, which is why its assignment is guarded like the rest.
        """
        variables: dict[str, ParsedVariable] = {}
        times = self._times(group, names)
        elevation = next((name for name in _ELEVATION_NAMES if name in names), None)
        if elevation is not None:
            node_z = self._variable(
                group, elevation, point_dim, volume_dim, times, time_dim
            )
            if node_z is not None:
                variables["node_z"] = node_z
        for name in sorted(names - _STRUCTURAL):
            variable = self._variable(
                group, name, point_dim, volume_dim, times, time_dim
            )
            if variable is not None:
                variables[name] = variable
        return variables

    def _times(self, group, names: set[str]) -> np.ndarray | None:
        """Read the time coordinate eagerly, or ``None`` when the file has none."""
        if _TIME not in names:
            return None
        return np.asarray(group.OpenMDArray(_TIME).ReadAsArray(), dtype=np.float64)

    def _variable(
        self,
        group,
        name: str,
        point_dim: str,
        volume_dim: str,
        times: np.ndarray | None,
        time_dim: str | None,
    ) -> ParsedVariable | None:
        """Build one lazy variable, placed by its dimension names.

        Returns ``None`` -- with a warning -- for an array on neither the point nor the
        volume dimension, which the mesh model cannot index.
        """
        array = group.OpenMDArray(name)
        dim_names = [dimension.GetName() for dimension in array.GetDimensions()]
        shape = tuple(dimension.GetSize() for dimension in array.GetDimensions())
        if point_dim in dim_names:
            location = "node"
        elif volume_dim in dim_names:
            location = "face"
        else:
            warnings.warn(
                f"{self.path!r} variable {name!r} is on dimensions {dim_names}, neither "
                f"the node nor the face dimension, so it cannot be placed on the mesh "
                f"and is skipped",
                stacklevel=5,
            )
            return None
        has_time = times is not None and time_dim is not None and time_dim in dim_names
        loader = _SwwLoader(self.path, name)
        return ParsedVariable(
            location=location,
            shape=shape,
            loader=loader if self.lazy else None,
            values=None if self.lazy else loader(),
            time_values=times if has_time else None,
        )
