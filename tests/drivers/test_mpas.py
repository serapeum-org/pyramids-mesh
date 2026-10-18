"""Tests for the MPAS reader (C6), against a real committed sample.

MPAS stores a spherical Voronoi tessellation; this driver reads the dual -- the Voronoi
corners become nodes and each primal cell becomes a face listed from ``verticesOnCell``.
The fixture is a genuine MPAS quasi-uniform grid committed under ``tests/data/mpas/``
(Apache-2.0, UXarray test data): nCells=162, nVertices=320, 12 pentagons among hexagons,
``verticesOnCell`` 1-based with ``0`` padding. The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np

from pyramids_mesh.drivers.mpas import MpasDriver
from pyramids_mesh.drivers.threedi import ThreeDiDriver
from pyramids_mesh.drivers.tuflowfv import TuflowFvDriver

_MESH = Path(__file__).parent.parent / "data" / "mpas" / "mesh.QU.1920km.151026.nc"
_THREEDI = Path(__file__).parent.parent / "data" / "3di" / "results_3di.nc"


def _read():
    """Read the committed MPAS grid, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MpasDriver().read(str(_MESH))


def test_dual_vertices_become_nodes_and_cells_become_faces():
    """The 320 dual vertices are the nodes and the 162 primal cells are the faces."""
    mesh = _read().mesh
    assert mesh.n_node == 320
    assert mesh.n_face == 162
    data = mesh.face_node_connectivity.data
    assert data.shape == (162, 6)
    assert int(data.min()) == -1  # the pentagons' padding slot
    assert int(data[data != -1].min()) == 0  # 0-based after the 1-based resolve
    assert int(data.max()) == 319


def test_connectivity_is_mixed_five_and_six_node_fill_padded():
    """A hex-dominant mesh gives 12 pentagons among 150 hexagons, padded to width six."""
    data = _read().mesh.face_node_connectivity.data
    widths = (data != -1).sum(axis=1)
    counts = {
        int(width): int(count)
        for width, count in zip(*np.unique(widths, return_counts=True))
    }
    assert counts == {5: 12, 6: 150}
    # every padded row pads only in its trailing slots, never between real corners
    for row in data:
        real = row != -1
        assert list(real) == sorted(real, reverse=True)


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise, as the mesh model requires."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_nodes_are_longitude_latitude_in_native_radians():
    """Coordinates are the dual vertices' lon/lat carried verbatim in the file's radians.

    The driver does not convert to degrees -- that projection concern belongs upstream in
    pyramids -- so latitude stays within +-pi/2 and longitude within [0, 2*pi), the same
    convention the ICON reader follows.
    """
    mesh = _read().mesh
    assert mesh.node_y.min() >= -np.pi / 2 - 1e-6
    assert mesh.node_y.max() <= np.pi / 2 + 1e-6
    assert mesh.node_x.min() >= -1e-6
    assert mesh.node_x.max() <= 2 * np.pi + 1e-6
    # radians: a latitude never exceeds +-1.58; a degrees conversion would reach past it.
    assert mesh.node_y.max() < 1.58


def test_crs_is_none_on_the_unit_sphere():
    """The grid lives on a unit sphere with no EPSG authority, so crs is None."""
    assert _read().mesh.crs is None


def test_can_open_accepts_mpas_and_declines_non_mpas_and_junk(tmp_path):
    """can_open accepts the MPAS grid but declines another model's .nc and junk."""
    driver = MpasDriver()
    assert driver.can_open(str(_MESH)) is True
    # a real NetCDF that is not MPAS (a 3Di result) is declined, not stolen
    assert driver.can_open(str(_THREEDI)) is False

    junk = tmp_path / "plain.nc"
    junk.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_other_nc_drivers_do_not_steal_the_mpas_grid():
    """The 3Di and TUFLOW FV .nc drivers decline the MPAS grid; dispatch is unambiguous."""
    assert ThreeDiDriver().can_open(str(_MESH)) is False
    assert TuflowFvDriver().can_open(str(_MESH)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else -- topology only."""
    assert set(MpasDriver().capabilities) == {"read_mesh"}
