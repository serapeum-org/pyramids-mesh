"""ADCIRC reader (task D12).

ADCIRC is a coastal-circulation model. Its mesh is the ASCII grid ``fort.14`` -- a node
list ``(id, x, y, depth)`` followed by a triangular element table ``(id, 3, n1, n2, n3)``
and boundary sections -- and its scalar results are ``fort.63`` / ``fort.63.nc`` (a
time series of sea-surface elevation on the nodes). This reader builds the surface mesh
from ``fort.14`` and, when a matching ``fort.63.nc`` sits beside it, attaches the
node-located result variables over time.

``fort.14``/``fort.63`` have no file suffix and ``fort.63.nc`` ends ``.nc``, so -- as with
the Esri TIN directory format -- the driver claims a component suffix and overrides
``can_open`` to recognise a ``fort.14`` file, or a directory holding one. The NetCDF
results are read through the GDAL multidimensional API vendored in the ``pyramids-core``
wheel and activated by ``import pyramids`` -- no declared ``netCDF4``/``h5py`` dependency.
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
from pyramids_mesh.drivers._gr3 import read_gr3_grid, sniff_gr3

_GRID_NAME = "fort.14"
_RESULT_NAMES = ("fort.63.nc",)  # the NetCDF elevation series read here
# Root NetCDF arrays that are coordinates/topology, not node result series.
_NON_RESULT = frozenset({"x", "y", "depth", "element", "time", "adcirc_mesh"})
# ADCIRC marks dry/missing nodes with this sentinel; used when no CF fill attribute.
_ADCIRC_FILL = -99999.0
_TIME_DIM = "time"
_NODE_DIM = "node"


class AdcircDriver(BaseDriver):
    """Reader for the ADCIRC ``fort.14`` grid and its ``fort.63.nc`` results.

    The mesh comes from ``fort.14``: node coordinates, the node ``depth`` (becomes a
    ``node_z`` variable), and the 1-based triangular element table. When a ``fort.63.nc``
    file sits beside it, each node-located ``(time, node)`` array becomes a variable over
    the ``time`` axis; values are lazy.

    Documented limitations:
        - **The input is a ``fort.14`` file (or its directory).** ``fort.14`` has no
          suffix, so ``read_file`` reaches it through the ``fort.14`` path directly or a
          directory passed with ``driver="adcirc"``; ``can_open`` accepts either.
        - **Triangular elements only.** ADCIRC grids are triangulations; a non-triangular
          element type is refused rather than guessed.
        - **Element ids are 1-based.** They are resolved to 0-based and the base recorded
          for a writer.
        - **Results come from ``fort.63.nc`` only.** A plain ASCII ``fort.63`` is not read
          here; without a ``fort.63.nc`` sibling the bare mesh is returned.
        - **No CRS.** ``fort.14`` carries lon/lat but declares no authority, so ``crs`` is
          ``None`` rather than guessed. ``depth`` is ADCIRC bathymetric depth (positive
          down), carried verbatim as ``node_z``.

    Example:
        Read an ADCIRC grid and its elevation series (shown, not run as a doctest -- it
        needs the real ``fort.14``/``fort.63.nc`` pair)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("run/fort.14")
            zeta = mesh.get("zeta")       # node-located, over time
            bed = mesh.get("node_z")      # ADCIRC depth
    """

    name = "adcirc"
    long_name = "ADCIRC"
    # fort.14 dispatches by its ".14" suffix; can_open also accepts its directory.
    suffixes = (".14",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is a ``fort.14`` grid or a directory holding one.

        Overrides the base check, which declines every directory: an ADCIRC run is
        reached through its ``fort.14`` file or the directory containing it. Never
        raises -- a sniff error is warned and declined.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a ``fort.14`` file, or a directory holding one, that sniffs
                as an ADCIRC grid.
        """
        grid = self._grid_file(path)
        if grid is None:
            return False
        try:
            return self._sniff(grid)
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            warnings.warn(
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False

    @staticmethod
    def _grid_file(path: str) -> str | None:
        """Resolve the ``fort.14`` grid path from a file or a directory."""
        if os.path.isfile(path) and path.lower().endswith(".14"):
            return path
        if os.path.isdir(path):
            candidate = os.path.join(path, _GRID_NAME)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens as an ADCIRC ``fort.14`` grid.

        Cheap and discriminating: line 2 is the ``NE NP`` count pair (two positive ints)
        and the first node line is ``id x y depth`` (one int then three floats). Shared
        with the SCHISM reader through :func:`pyramids_mesh.drivers._gr3.sniff_gr3`, since
        ``fort.14`` and ``.gr3`` open identically.
        """
        return sniff_gr3(path)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a ``fort.14`` grid (and a sibling ``fort.63.nc``) into a ParsedMesh.

        Args:
            path: The ``fort.14`` file or its directory.
            lazy: When ``True``, leave each result variable behind a loader.

        Returns:
            ParsedMesh: The triangular surface mesh, ``node_z``, and the node result
                variables when a ``fort.63.nc`` sibling is present.

        Raises:
            FileFormatNotSupportedError: If the path is not an ADCIRC grid or an element
                is not a triangle.
        """
        grid = self._grid_file(path)
        if grid is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an ADCIRC fort.14 grid or a directory holding one"
            )
        node_x, node_y, node_z, faces = self._read_grid(grid)
        variables = self._read_results(os.path.dirname(grid) or ".", lazy=lazy)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # fort.14 element ids are 1-based; validate re-bases and records the base.
            start_index=1,
            node_z=node_z,
            crs=None,
            variables=variables,
            global_attributes={"source": "ADCIRC fort.14"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_grid(self, grid: str):
        """Parse ``fort.14`` into node coordinates, elevation and triangle connectivity.

        Delegates the node and element blocks to the shared ``fort.14``/``.gr3`` parser
        with ``allow_quads=False``, so a non-triangular element is refused rather than
        guessed, as ADCIRC grids are triangulations.

        Raises:
            FileFormatNotSupportedError: If an element is not a 3-node triangle.
        """
        with open(grid, "r", encoding="latin-1") as handle:
            return read_gr3_grid(handle, allow_quads=False, source=grid)

    def _read_results(self, directory: str, *, lazy: bool) -> dict[str, ParsedVariable]:
        """Read node result series from a sibling ``fort.63.nc``, if present.

        Returns:
            dict[str, ParsedVariable]: One node-located variable per ``(time, node)``
                array, keyed by its NetCDF name; empty when no ``fort.63.nc`` sits beside
                the grid.
        """
        result = next(
            (
                os.path.join(directory, name)
                for name in _RESULT_NAMES
                if os.path.isfile(os.path.join(directory, name))
            ),
            None,
        )
        if result is None:
            return {}
        variables: dict[str, ParsedVariable] = {}
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(result, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                times = None
                if _TIME_DIM in names:
                    times = np.asarray(
                        group.OpenMDArray(_TIME_DIM).ReadAsArray(), np.float64
                    )
                for name in sorted(names):
                    if name in _NON_RESULT:
                        continue
                    array = group.OpenMDArray(name)
                    dim_names = [d.GetName() for d in array.GetDimensions()]
                    if dim_names != [_TIME_DIM, _NODE_DIM]:
                        continue
                    shape = tuple(d.GetSize() for d in array.GetDimensions())
                    loader = _AdcircLoader(result, name)
                    variables[name] = ParsedVariable(
                        location="node",
                        shape=shape,
                        loader=loader if lazy else None,
                        values=None if lazy else loader(),
                        standard_name=_read_attr(array, "standard_name"),
                        units=_read_attr(array, "units"),
                        nodata=_read_fill(array),
                        time_values=times,
                    )
            finally:
                dataset.Close()
        return variables


def _read_attr(array, name: str):
    """Return a string attribute's value from a GDAL MD-array, or ``None``."""
    for attribute in array.GetAttributes():
        if attribute.GetName() == name:
            try:
                return str(attribute.Read())
            except RuntimeError:
                return None
    return None


def _read_fill(array) -> float:
    """Return the no-data fill value for a result array.

    Reads ``_FillValue`` then ``missing_value`` if declared; otherwise falls back to
    ADCIRC's conventional dry/missing-node sentinel ``-99999.0`` (ADCIRC marks dry nodes
    with this value and does not always declare a CF fill attribute), so a dry node is
    recorded as no-data rather than carried through as a real elevation. The ``-99999.0``
    fallback is ADCIRC-specific and is applied to **every** result series that declares no
    CF fill attribute, not only elevation -- ADCIRC uses the sentinel across its outputs.
    """
    attributes = {a.GetName(): a for a in array.GetAttributes()}
    for name in ("_FillValue", "missing_value"):
        attribute = attributes.get(name)
        if attribute is not None:
            try:
                return float(np.asarray(attribute.Read()).ravel()[0])
            except (RuntimeError, ValueError, TypeError, IndexError):
                pass
    return _ADCIRC_FILL


class _AdcircLoader:
    """A lazy loader for one ``fort.63.nc`` array.

    Closes over the file path and the array name, never an open handle (design section
    7): reading it reopens the file, reads the array, and closes.
    """

    def __init__(self, path: str, array_name: str) -> None:
        self.path = path
        self.array_name = array_name

    def __call__(self) -> np.ndarray:
        """Reopen the NetCDF file and read this array."""
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                return np.asarray(group.OpenMDArray(self.array_name).ReadAsArray())
            finally:
                dataset.Close()
