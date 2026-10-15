"""Import and packaging contract of the `pyramids_mesh` distribution.

These are the tests that have to pass before any mesh code exists: the package
imports, it reports the version its metadata declares, the vendored osgeo that
`import pyramids` activates is reachable, and the UGRID tier this package builds
on is importable from the environment.
"""

import pyramids  # noqa: F401  # isort:skip

import pytest

import pyramids_mesh


def test_version_matches_metadata():
    """`__version__` comes from the installed distribution metadata."""
    from importlib.metadata import version

    assert pyramids_mesh.__version__ == version("pyramids-mesh")
    assert pyramids_mesh.__version__ != "unknown"


def test_osgeo_resolves_through_pyramids():
    """`from osgeo import gdal` works because importing pyramids activated it.

    Nothing in these environments installs a top-level osgeo; the bindings live
    inside the pyramids-gis wheel. This test is the canary for the import-order
    contract that `pyramids_mesh/__init__.py` and `tests/conftest.py` both rely
    on — if it fails, a stray refactor moved or dropped the `import pyramids`.
    """
    from osgeo import gdal

    assert gdal.VersionInfo() != ""


def test_ugrid_tier_is_available():
    """The pyramids UGRID surface this package extends is importable."""
    from pyramids.netcdf.ugrid import Mesh2d, UgridDataset

    assert Mesh2d is not None
    assert UgridDataset is not None


def test_private_upstream_api_this_package_depends_on_still_exists():
    """Canary for the private upstream names the drivers have no public route to.

    The published driver contract says to reuse pyramids' own exceptions and to build a
    `MeshVariable` with eager or lazy data, but neither has a public path: nothing named
    `*Error` is re-exported from `pyramids` or `pyramids.base`, and the data fields are
    underscored. The floor is `>=0.68.0` with no upper bound, so a patch release could
    rename either without it being breaking upstream. This fails loudly if that happens,
    instead of surfacing as an ImportError from a driver mid-read.
    """
    from dataclasses import fields

    from pyramids.base._errors import FileFormatNotSupportedError
    from pyramids.netcdf.ugrid import MeshVariable

    assert issubclass(FileFormatNotSupportedError, Exception)
    names = {field.name for field in fields(MeshVariable)}
    assert {"_data", "_loader"} <= names, names


def test_unit_square_mesh_fixture(unit_square_mesh):
    """The shared fixture hands back a two-triangle topology."""
    node_x, node_y, face_node_connectivity = unit_square_mesh

    assert node_x.shape == node_y.shape == (4,)
    assert face_node_connectivity.shape == (2, 3)
