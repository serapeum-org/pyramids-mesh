"""FLO-2D reader (task D10).

A FLO-2D project is not one file but a **directory** of interdependent files. This driver
reads the two that define a floodplain grid and its results:

- ``DEPFP.OUT`` -- one line per element, ``<id> <x> <y> <max floodplain depth>``. FLO-2D
  elements are square cells on a regular grid; the line gives each cell's centre.
- ``TIMDEP.HDF5`` -- the time-varying output, an HDF5 tree whose group
  ``TIMDEP NETCDF OUTPUT RESULTS`` holds one sub-group per quantity (``FLOW DEPTH``,
  ``Velocity MAG`` and the rest), each with ``Values`` ``(n_time, n_element)``, ``Times``,
  ``Mins`` and ``Maxs`` -- the same shape XMDF uses.

The square cells are lifted to a **quad mesh**: four corner nodes per cell, shared corners
deduplicated into one node table. HDF5 is read through the GDAL multidimensional API over
the GDAL vendored in the ``pyramids-core`` wheel (activated by ``import pyramids``), with no
declared dependency and no ``h5py``. See ``planning/design.md`` sections 20.3 and 20.4.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import warnings

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# The two components that make a directory a readable FLO-2D project here.
_COORDS = "depfp.out"  # per-element id, x, y, max depth
_RESULTS = "timdep.hdf5"  # time-varying HDF5 output
_REQUIRED = (_COORDS, _RESULTS)

# The HDF5 layout inside TIMDEP.HDF5.
_RESULTS_GROUP = "TIMDEP NETCDF OUTPUT RESULTS"
_VALUES = "Values"
_TIMES = "Times"
_MINS = "Mins"
_MAXS = "Maxs"

# Coordinates round to this many decimals before the shared-corner dedup; a FLO-2D grid is
# regular, so the key is exact for the metre spacings these files use.
_DEDUP_DECIMALS = 6


class Flo2dDriver(BaseDriver):
    """Reader for the FLO-2D floodplain-grid format.

    A FLO-2D project is a directory. ``read_file`` cannot dispatch a bare directory (it has
    no suffix), so the driver claims ``.out`` and is reached either by pointing ``read_file``
    at the ``DEPFP.OUT`` component, or by passing the directory with ``driver="flo2d"``.
    Either way ``can_open`` confirms the defining components are present.

    Documented limitations:
        - **The input is a directory.** ``DEPFP.OUT`` (element centres) and ``TIMDEP.HDF5``
          (time series) must both be present; a lone ``.out`` elsewhere is declined.
        - **Square cells become quad faces.** Each element is a square whose side is the grid
          spacing; four corner nodes are built per cell and shared corners deduplicated. The
          element centres are carried as the face centres.
        - **Only 2D scalar results are shipped.** Each ``TIMDEP`` quantity whose ``Values`` is
          ``(n_time, n_element)`` becomes a face-located variable over its ``Times`` axis; a
          vector quantity (a third axis, e.g. ``Velocity``) is skipped, the ``Velocity MAG``
          magnitude being the scalar form. A genuinely 3D result is not modelled yet.
        - **Values are lazy** -- the loader reopens the file per read, closing over a path.
        - ``Mins``/``Maxs`` ride along as per-timestep ``minimum``/``maximum`` attributes, and
          the ``DEPFP.OUT`` max-depth column is a static ``max_floodplain_depth`` face
          variable.
        - There is no elevation or CRS in the files read here, so ``node_z`` and ``crs`` are
          ``None``.

    Example:
        Read a FLO-2D project by its directory (shown, not run as a doctest -- it needs a
        real project directory)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("flo2d_project", driver="flo2d")
            depth = mesh.get("FLOW DEPTH")      # face-located, over time
    """

    name = "flo2d"
    long_name = "FLO-2D"
    # FLO-2D is a directory; read_file reaches it through the DEPFP.OUT component.
    suffixes = (".out",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is a FLO-2D project directory or a component within one.

        Overrides the base check (which only ever sniffs a file): a FLO-2D project is a
        directory, reached directly or through one of its component files. Never raises.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a FLO-2D directory, or a component file whose directory is one.
        """
        project = self._project_dir(path)
        if project is None:
            return False
        try:
            return self._sniff(project)
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            warnings.warn(  # pragma: no cover
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False  # pragma: no cover

    @staticmethod
    def _project_dir(path: str) -> str | None:
        """Resolve the project directory from a directory or a component-file path."""
        if os.path.isdir(path):
            return path
        if os.path.isfile(path) and os.path.basename(path).lower() in _REQUIRED:
            return os.path.dirname(path) or "."
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether the directory holds the required FLO-2D components."""
        present = {entry.lower() for entry in os.listdir(path)}
        return all(component in present for component in _REQUIRED)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a FLO-2D project into a :class:`ParsedMesh`.

        Args:
            path: The project directory, or one of its component files.
            lazy: When ``True``, leave each time-varying variable behind a loader.

        Returns:
            ParsedMesh: The quad mesh with its per-cell result variables.

        Raises:
            FileFormatNotSupportedError: If the path is not a FLO-2D project or a component
                is missing.
        """
        project = self._project_dir(path)
        if project is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not a FLO-2D project directory or a component file inside one"
            )
        return _Flo2dReader(project, lazy=lazy).build()


class _Flo2dLoader:
    """A lazy loader for one TIMDEP quantity's ``Values`` array.

    Closes over the HDF5 path and the group path, never an open handle (design section 7):
    reading it reopens the file, walks to the array, and closes.
    """

    def __init__(self, path: str, group_path: tuple[str, ...], array_name: str) -> None:
        self.path = path
        self.group_path = group_path
        self.array_name = array_name

    def __call__(self) -> np.ndarray:
        """Reopen the file and read this quantity's array."""
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                for name in self.group_path:
                    group = group.OpenGroup(name)
                return np.asarray(group.OpenMDArray(self.array_name).ReadAsArray())
            finally:
                dataset.Close()


class _Flo2dReader:
    """One FLO-2D project: read the element grid and the TIMDEP results."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Read the components and assemble the quad mesh with its results.

        Returns:
            ParsedMesh: The surface mesh and its per-cell variables.

        Raises:
            FileFormatNotSupportedError: If a required component is missing or empty.
        """
        self._require()
        centres = self._read_coords()
        node_x, node_y, faces = self._build_quads(centres)
        variables: dict[str, ParsedVariable] = {}
        variables["max_floodplain_depth"] = ParsedVariable(
            location="face", shape=(len(centres),), values=centres[:, 2].copy()
        )
        self._add_results(len(centres), variables)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=0,
            node_z=None,
            face_x=centres[:, 0].copy(),
            face_y=centres[:, 1].copy(),
            crs=None,
            variables=variables,
            global_attributes={"source": "FLO-2D results"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self) -> None:
        """Refuse a directory missing a required component."""
        present = {entry.lower() for entry in os.listdir(self.path)}
        missing = [name for name in _REQUIRED if name not in present]
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a FLO-2D project: missing {missing}"
            )

    def _component(self, name: str) -> str:
        """Return the real (case-preserved) path of a required component."""
        for entry in os.listdir(self.path):
            if entry.lower() == name:
                return os.path.join(self.path, entry)
        raise FileFormatNotSupportedError(  # pragma: no cover - guarded by _require
            f"{self.path!r} is not a FLO-2D project: missing {name!r}"
        )

    def _read_coords(self) -> np.ndarray:
        """Read ``DEPFP.OUT`` as an ``(n_element, 3)`` array of ``x, y, max_depth``."""
        rows = np.loadtxt(self._component(_COORDS), dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] < 4:
            raise FileFormatNotSupportedError(
                f"{self.path!r} DEPFP.OUT is not the expected '<id> <x> <y> <depth>' table"
            )
        return rows[:, 1:4]  # x, y, max depth (the 1-based id in column 0 is dropped)

    def _build_quads(
        self, centres: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lift the square cells to a quad mesh, deduplicating shared corner nodes."""
        x = centres[:, 0]
        y = centres[:, 1]
        half = self._cell_half(x, y)
        # Four corners per cell, counter-clockwise from the lower-left.
        offsets = np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]], np.float64
        )
        corners = np.empty((len(centres), 4, 2), np.float64)
        corners[:, :, 0] = x[:, None] + offsets[:, 0]
        corners[:, :, 1] = y[:, None] + offsets[:, 1]
        flat = corners.reshape(-1, 2)
        unique, inverse = np.unique(
            np.round(flat, _DEDUP_DECIMALS), axis=0, return_inverse=True
        )
        node_x = unique[:, 0].astype(np.float64)
        node_y = unique[:, 1].astype(np.float64)
        faces = inverse.reshape(len(centres), 4).astype(np.int64)
        return node_x, node_y, faces

    def _cell_half(self, x: np.ndarray, y: np.ndarray) -> float:
        """Half the grid spacing, from the smallest gap between distinct coordinates."""
        spacings = [self._min_gap(x), self._min_gap(y)]
        spacing = min(s for s in spacings if s is not None)
        return spacing / 2.0

    @staticmethod
    def _min_gap(values: np.ndarray) -> float | None:
        """Return the smallest positive gap between distinct values, or ``None``."""
        distinct = np.unique(values)
        if distinct.size < 2:
            return None
        return float(np.min(np.diff(distinct)))

    def _add_results(self, n_face: int, variables: dict[str, ParsedVariable]) -> None:
        """Add a lazy face variable for each 2D scalar TIMDEP quantity."""
        hdf5 = self._component(_RESULTS)
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(hdf5, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                root = dataset.GetRootGroup()
                if _RESULTS_GROUP not in root.GetGroupNames():
                    return
                results = root.OpenGroup(_RESULTS_GROUP)
                for quantity in results.GetGroupNames():
                    self._add_quantity(hdf5, results, quantity, n_face, variables)
            finally:
                dataset.Close()

    def _add_quantity(
        self,
        hdf5: str,
        results,
        quantity: str,
        n_face: int,
        variables: dict[str, ParsedVariable],
    ) -> None:
        """Add one TIMDEP quantity group as a face variable, when it is 2D scalar."""
        group = results.OpenGroup(quantity)
        names = group.GetMDArrayNames()
        if _VALUES not in names or _TIMES not in names:
            return
        shape = tuple(d.GetSize() for d in group.OpenMDArray(_VALUES).GetDimensions())
        if len(shape) != 2 or shape[1] != n_face:
            return  # a vector quantity (extra axis) or a width that is not the face count
        times = np.asarray(group.OpenMDArray(_TIMES).ReadAsArray(), np.float64)
        attributes: dict[str, object] = {}
        if _MINS in names:
            attributes["minimum"] = np.asarray(group.OpenMDArray(_MINS).ReadAsArray())
        if _MAXS in names:
            attributes["maximum"] = np.asarray(group.OpenMDArray(_MAXS).ReadAsArray())
        loader = _Flo2dLoader(hdf5, (_RESULTS_GROUP, quantity), _VALUES)
        variables[quantity] = ParsedVariable(
            location="face",
            shape=shape,
            loader=loader if self.lazy else None,
            values=None if self.lazy else loader(),
            time_values=times,
            attributes=attributes,
        )
