"""Mesh file-format drivers for the pyramids stack.

`pyramids-mesh` reads and writes the file formats of hydrodynamic and hydraulic
mesh models — 2DM, Selafin, DFSU, XMDF, PLY and the rest of the hydrodynamic formats —
as `pyramids.netcdf.ugrid` objects. `pyramids.netcdf.ugrid` owns the UGRID data
model (`Mesh2d`, `Network1d`, `UgridDataset`), its NetCDF I/O, and every operation
that follows from a topology already in hand — connectivity and graph algorithms,
regridding and sampling, gap filling, mesh-to-vector conversion, and construction
from a raster or a polygon layer. What it cannot do is open other people's mesh
files; that is this package's job, and its only job.

Drivers are discovered by `pyramids-core` through the `pyramids.mesh_drivers`
entry-point group, so `UgridDataset.read_file("model.slf")` resolves to a driver
here once this package is installed. Dependency-backed drivers sit behind extras
(`[dhi]`) and are skipped at discovery when their extra
is absent. See ADR-0005 for the scope decision.

The package root deliberately re-exports nothing: it carries `__version__` and the
import contract that activates the vendored osgeo bindings, and that is all. Installing
the package is the whole integration, so nothing here needs importing. A driver class is
still reachable by its module path (`pyramids_mesh.drivers.sms.TwoDmDriver`, which is how
its entry point names it) for the rare case of wanting one directly.

Examples:
    - The root carries the installed version and nothing else. `__all__` is empty
      because there is deliberately nothing to re-export:

        >>> import pyramids_mesh
        >>> pyramids_mesh.__version__.split(".")[0].isdigit()
        True
        >>> pyramids_mesh.__all__
        []

    - A driver is reached by its module path, not from the root, and reports the
      suffixes it claims:

        >>> from pyramids_mesh.drivers.sms import TwoDmDriver
        >>> TwoDmDriver.name
        '2dm'
        >>> TwoDmDriver.suffixes
        ('.2dm',)
"""

# Import pyramids FIRST — before anything else, and keep it first. pyramids-core
# bundles GDAL's osgeo SWIG bindings and native libs inside its platform wheel
# and activates them on import (puts `_vendor/osgeo` on sys.path, sets
# GDAL_DATA / PROJ_DATA, registers the Windows DLL directory). Any module here
# that does `from osgeo import gdal` at load time needs this line to have run
# first, or that import fails wherever no separate GDAL is installed.
import pyramids  # noqa: F401

try:
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


# The distribution name, not `__name__`: the import name is `pyramids_mesh` and
# the distribution is `pyramids-mesh`. importlib.metadata normalizes `_` to `-`
# so either spelling resolves today, but the metadata lookup is a distribution
# lookup — spell it the way the metadata does.
try:
    __version__ = version("pyramids-mesh")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__: list[str] = []
