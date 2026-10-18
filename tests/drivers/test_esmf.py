"""Tests for the ESMF reader, against a real committed sample.

ESMF stores an unstructured mesh with an explicit node list (``nodeCoords``), a per-element
node table (``elementConn``) and a per-element node count (``numElementConn``). The fixture
is the canonical ``ne4np4-esmf`` mesh committed under ``tests/data/esmf/`` (CDF-1 classic
NetCDF, 866 nodes / 936 elements, a mix of triangles and quads). The tests read the
committed bytes: the node list becomes the nodes and ``elementConn`` the fill-padded,
mixed-degree faces.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np

from pyramids_mesh.drivers.esmf import EsmfDriver
from pyramids_mesh.drivers.scrip import ScripDriver
from pyramids_mesh.drivers.geos import GeosDriver

_DATA = Path(__file__).parent.parent / "data"
_MESH = _DATA / "esmf" / "ne4np4-esmf.nc"
_SCRIP = _DATA / "scrip" / "outCSne8.nc"
_GEOS = _DATA / "geos" / "test-c12.native.nc4"
_MPAS = _DATA / "mpas" / "mesh.QU.1920km.151026.nc"
_ICON = _DATA / "icon" / "icon_grid_R02B04_subset.nc"


def _read():
    """Read the committed ESMF mesh, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EsmfDriver().read(str(_MESH))


def test_node_list_and_elements_build_the_mesh():
    """The 866 nodes become nodes and the 936 elements become faces."""
    mesh = _read().mesh
    assert mesh.n_node == 866
    assert mesh.n_face == 936
    assert mesh.face_node_connectivity.data.shape == (936, 4)


def test_num_element_conn_drives_the_padding():
    """numElementConn gives 144 triangles among 792 quads, padded to width four."""
    data = _read().mesh.face_node_connectivity.data
    widths = (data != -1).sum(axis=1)
    counts = {
        int(width): int(count)
        for width, count in zip(*np.unique(widths, return_counts=True))
    }
    assert counts == {3: 144, 4: 792}
    # every padded row pads only in its trailing slots, never between real nodes
    for row in data:
        real = row != -1
        assert list(real) == sorted(real, reverse=True)


def test_element_conn_is_one_based_resolved_to_zero_based():
    """The 1-based elementConn arrives 0-based; the base 1 is recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    data = connectivity.data
    assert int(data[data != -1].min()) == 0  # 0-based after the 1-based resolve
    assert int(data.max()) == 865


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise, as the mesh model requires."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_coordinates_are_longitude_latitude_in_degrees():
    """nodeCoords column 0 is longitude and column 1 latitude, carried in degrees."""
    mesh = _read().mesh
    assert mesh.node_y.min() >= -90.0 - 1e-6
    assert mesh.node_y.max() <= 90.0 + 1e-6
    assert mesh.node_x.min() >= -1e-6
    assert mesh.node_x.max() <= 360.0 + 1e-6


def test_crs_is_none_on_the_sphere():
    """The mesh is lon/lat on a sphere with no EPSG authority, so crs is None."""
    assert _read().mesh.crs is None


def test_can_open_accepts_esmf_and_declines_junk(tmp_path):
    """can_open accepts the ESMF mesh and declines non-NetCDF junk and a directory."""
    driver = EsmfDriver()
    assert driver.can_open(str(_MESH)) is True

    junk = tmp_path / "plain.nc"
    junk.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_can_open_declines_every_sibling_nc_format():
    """The ESMF sniff declines the SCRIP, GEOS, MPAS and ICON .nc samples."""
    driver = EsmfDriver()
    assert driver.can_open(str(_SCRIP)) is False
    assert driver.can_open(str(_GEOS)) is False
    assert driver.can_open(str(_MPAS)) is False
    assert driver.can_open(str(_ICON)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else -- topology only."""
    assert set(EsmfDriver().capabilities) == {"read_mesh"}


def test_faces_helper_resolves_count_and_base():
    """The face builder keeps numElementConn nodes, re-bases, and pads the rest."""
    from pyramids_mesh.drivers.esmf import _EsmfReader

    element_conn = np.array([[1, 2, 3, 4], [5, 6, 7, 9]])
    counts = np.array([3, 4])  # a triangle then a quad
    faces = _EsmfReader._faces(element_conn, counts, start_index=1)
    assert faces.tolist() == [[0, 1, 2, -1], [4, 5, 6, 8]]
