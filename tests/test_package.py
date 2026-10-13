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


def test_unit_square_mesh_fixture(unit_square_mesh):
    """The shared fixture hands back a two-triangle topology."""
    node_x, node_y, face_node_connectivity = unit_square_mesh

    assert node_x.shape == node_y.shape == (4,)
    assert face_node_connectivity.shape == (2, 3)


