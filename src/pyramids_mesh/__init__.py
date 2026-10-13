"""Mesh file-format drivers for the pyramids stack.

`pyramids-mesh` reads and writes the file formats of hydrodynamic and hydraulic
mesh models — 2DM, Selafin, DFSU, XMDF, PLY and the rest of MDAL's driver table —
as `pyramids.netcdf.ugrid` objects. `pyramids.netcdf.ugrid` owns the UGRID data
model (`Mesh2d`, `Network1d`, `UgridDataset`), its NetCDF I/O, and every operation
that follows from a topology already in hand — connectivity and graph algorithms,
regridding and sampling, gap filling, mesh-to-vector conversion, and construction
from a raster or a polygon layer. What it cannot do is open other people's mesh
files; that is this package's job, and its only job.

Drivers are discovered by `pyramids-gis` through the `pyramids.mesh_drivers`
entry-point group, so `UgridDataset.read_file("model.slf")` resolves to a driver
here once this package is installed. Dependency-backed drivers sit behind extras
(`[dhi]`, `[interchange]`, `[hdf]`) and are skipped at discovery when their extra
is absent. See ADR-0005 for the scope decision.

The public surface is empty at `0.1.0` — this release is the repository scaffold
(packaging, CI, docs) plus `__version__`. Every name added later is re-exported
here, so `from pyramids_mesh import ...` stays the import worth using.

Examples:
    - The installed version string:

        >>> import pyramids_mesh
        >>> isinstance(pyramids_mesh.__version__, str)
        True
"""

# Import pyramids FIRST — before anything else, and keep it first. pyramids-gis
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
