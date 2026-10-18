"""Tests for the GEOS cubed-sphere reader, against a real committed sample.

GEOS stores a structured corner lattice per cube face (``corner_lons``/``corner_lats``,
shape ``(6, ny, nx)``) with optional cell centres (``lons``/``lats``). This driver ravels
the corner lattice into a node list and reconstructs the quad connectivity by index
arithmetic. The fixture is a genuine C12 GEOS grid committed under ``tests/data/geos/``
(Apache-2.0, UXarray test data): six faces of a 13 x 13 corner lattice, giving 1014 nodes
and 864 quad cells. The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np

from pyramids_mesh.drivers.geos import GeosDriver
from pyramids_mesh.drivers.scrip import ScripDriver
from pyramids_mesh.drivers.esmf import EsmfDriver

_DATA = Path(__file__).parent.parent / "data"
_GRID = _DATA / "geos" / "test-c12.native.nc4"
_SCRIP = _DATA / "scrip" / "outCSne8.nc"
_ESMF = _DATA / "esmf" / "ne4np4-esmf.nc"
_MPAS = _DATA / "mpas" / "mesh.QU.1920km.151026.nc"
_ICON = _DATA / "icon" / "icon_grid_R02B04_subset.nc"


def _read():
    """Read the committed GEOS grid, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GeosDriver().read(str(_GRID))


def test_corner_lattice_becomes_nodes_and_quads():
    """Six 13x13 corner lattices give 1014 nodes and 864 quad faces."""
    mesh = _read().mesh
    assert mesh.n_node == 1014  # 6 x 13 x 13 ravelled corners
    assert mesh.n_face == 864  # 6 x 12 x 12 cells
    data = mesh.face_node_connectivity.data
    assert data.shape == (864, 4)
    assert bool((data != -1).all())  # every cell is a full quad
    assert int(data.min()) == 0
    assert int(data.max()) == 1013


def test_connectivity_base_is_recorded_as_zero():
    """The computed connectivity is 0-based, so the writer base is zero."""
    assert _read().mesh.face_node_connectivity.original_start_index == 0


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise, as the mesh model requires."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_cell_centres_populate_face_xy():
    """lons/lats become the face centres, one per cell."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert mesh.face_y is not None
    assert len(mesh.face_x) == 864


def test_coordinates_are_longitude_latitude_in_degrees():
    """GEOS writes degrees, so latitudes span the poles and longitudes stay in 0..360."""
    mesh = _read().mesh
    assert mesh.node_y.min() >= -90.0 - 1e-6
    assert mesh.node_y.max() <= 90.0 + 1e-6
    assert mesh.node_x.min() >= -1e-6
    assert mesh.node_x.max() <= 360.0 + 1e-6


def test_crs_is_none_on_the_sphere():
    """The grid is lon/lat on a sphere with no EPSG authority, so crs is None."""
    assert _read().mesh.crs is None


def test_can_open_accepts_geos_and_declines_junk(tmp_path):
    """can_open accepts the GEOS grid and declines non-NetCDF junk and a directory."""
    driver = GeosDriver()
    assert driver.can_open(str(_GRID)) is True

    junk = tmp_path / "plain.nc4"
    junk.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_can_open_declines_every_sibling_nc_format():
    """The GEOS sniff declines the SCRIP, ESMF, MPAS and ICON .nc samples."""
    driver = GeosDriver()
    assert driver.can_open(str(_SCRIP)) is False
    assert driver.can_open(str(_ESMF)) is False
    assert driver.can_open(str(_MPAS)) is False
    assert driver.can_open(str(_ICON)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else -- topology only."""
    assert set(GeosDriver().capabilities) == {"read_mesh"}


def test_faces_helper_computes_quads_by_index_arithmetic():
    """A single 2x2-corner panel yields one quad of its four ravelled corners."""
    from pyramids_mesh.drivers.geos import _GeosReader

    faces = _GeosReader._faces((1, 2, 2))
    assert faces.tolist() == [[0, 1, 3, 2]]
