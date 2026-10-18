"""Tests for the FVCOM native grid reader (C8), against a real committed trio.

An FVCOM native grid is a trio of ASCII files sharing a basename. A genuine channel
grid (MIT, estuarine-utokyo/xfvcom) is committed under ``tests/data/fvcom/``:
``chn_grd.dat`` (871 nodes / 1620 triangles), ``chn_dep.dat`` (per-node bathymetry) and
``chn_obc.dat`` (21 open-boundary nodes). The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset, mesh_drivers
from pyramids_mesh.drivers.fvcom import FvcomDriver, _header_count

_DATA = Path(__file__).parent.parent / "data" / "fvcom"
_GRD = _DATA / "chn_grd.dat"


def _read(path=_GRD):
    """Read the committed FVCOM grid, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FvcomDriver().read(str(path))


def test_reads_nodes_and_triangular_faces():
    """The grid reads 871 nodes and 1620 triangular faces."""
    mesh = _read().mesh
    assert mesh.n_node == 871
    assert mesh.n_face == 1620
    assert mesh.face_node_connectivity.data.shape == (1620, 3)


def test_cell_ids_are_one_based_resolved_to_zero_based():
    """The 1-based cell table is resolved to 0-based, and the base recorded."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) >= 0
    assert int(connectivity.data.max()) == 870


def test_faces_are_counter_clockwise():
    """Every triangle comes back counter-clockwise after validation."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        xs = mesh.node_x[row]
        ys = mesh.node_y[row]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_node_depth_becomes_node_z():
    """The depth column of chn_dep.dat becomes a node-located node_z variable."""
    node_z = _read().get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (871,)
    # The first three dep rows carry depths 10, 28, 46 in node order.
    assert node_z.data[:3].tolist() == [10.0, 28.0, 46.0]


def test_node_coordinates_come_from_the_grid_file():
    """Node x/y are the grid file's own coordinates, not the dep file's."""
    mesh = _read().mesh
    # The first grid node sits at (800000, 0); the channel spans y in [0, 4e5].
    assert mesh.node_x[0] == pytest.approx(800000.0)
    assert mesh.node_y[0] == pytest.approx(0.0)


def test_open_boundary_nodes_are_metadata():
    """The chn_obc.dat node ids ride along (0-based) on the global attributes."""
    attributes = _read().global_attributes
    assert attributes["open_boundary_node_count"] == 21
    nodes = np.asarray(attributes["open_boundary_nodes"])
    assert nodes.shape == (21,)
    # Node ids 1..21 (1-based) resolve to 0..20.
    assert nodes.tolist() == list(range(21))


def test_can_open_accepts_grid_file_and_directory():
    """can_open accepts the _grd.dat file and the directory that holds it."""
    driver = FvcomDriver()
    assert driver.can_open(str(_GRD)) is True
    assert driver.can_open(str(_DATA)) is True


def test_can_open_declines_non_fvcom_dat(tmp_path):
    """can_open sniffs the header, so a foreign .dat (even named _grd.dat) is declined."""
    driver = FvcomDriver()
    # A _grd.dat whose header is not the FVCOM Node/Cell Number pair.
    fake = tmp_path / "fake_grd.dat"
    fake.write_text("DATASET\nND 4\nNC 2\n")
    assert driver.can_open(str(fake)) is False
    # A .dat that is not a _grd.dat component at all.
    other = tmp_path / "results.dat"
    other.write_text("Node Number = 3\nCell Number = 1\n")
    assert driver.can_open(str(other)) is False
    # A directory with no _grd.dat inside.
    assert driver.can_open(str(tmp_path)) is False


def test_dispatch_is_collision_free_and_routes_to_fvcom():
    """No other registered driver claims the grid; dispatch resolves to fvcom alone."""
    assert FvcomDriver().can_open(str(_GRD)) is True
    others = {name: d for name, d in mesh_drivers().items() if name != "fvcom"}
    for name, driver in others.items():
        assert driver.can_open(str(_GRD)) is False, f"{name} also claims the FVCOM grid"
    if "fvcom" in mesh_drivers():  # once the entry point is registered
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = UgridDataset.read_file(str(_GRD))
        assert dataset.mesh.n_node == 871


def test_read_of_a_non_grid_path_is_a_format_error(tmp_path):
    """read() on a path that is neither a _grd.dat file nor a directory holding one."""
    other = tmp_path / "notes.dat"
    other.write_text("not fvcom")
    with pytest.raises(FileFormatNotSupportedError, match="not an FVCOM"):
        FvcomDriver().read(str(other))


def test_truncated_grid_is_a_format_error(tmp_path):
    """A grid file shorter than its header promises is a clean format error."""
    grd = tmp_path / "short_grd.dat"
    grd.write_text(
        "Node Number = 3\nCell Number = 2\n1 1 2 3 1\n"
    )  # one cell, no nodes
    with pytest.raises(FileFormatNotSupportedError, match="truncated"):
        FvcomDriver().read(str(grd))


def test_header_count_rejects_a_line_without_an_equals():
    """A header line lacking '=' is a format error (the no-separator guard)."""
    with pytest.raises(FileFormatNotSupportedError, match="= <int>"):
        _header_count("Node Number 871", "Node Number")


def test_header_count_rejects_a_non_integer_count():
    """A header whose value after '=' is not an integer is a format error."""
    with pytest.raises(FileFormatNotSupportedError, match="= <int>"):
        _header_count("OBC Node Number = many", "OBC Node Number")


def test_can_open_warns_and_declines_an_unreadable_grid_entry(tmp_path):
    """A directory whose _grd.dat entry is itself a directory makes the sniff fail."""
    (tmp_path / "bogus_grd.dat").mkdir()
    with pytest.warns(UserWarning, match="failed to sniff"):
        assert FvcomDriver().can_open(str(tmp_path)) is False


def test_can_open_declines_a_one_line_grid(tmp_path):
    """A _grd.dat with fewer than two significant lines fails the header sniff."""
    grd = tmp_path / "tiny_grd.dat"
    grd.write_text("Node Number = 3\n")
    assert FvcomDriver().can_open(str(grd)) is False


def test_read_of_a_headerless_grid_is_a_format_error(tmp_path):
    """A grid file with fewer than two significant lines is a format error on read."""
    grd = tmp_path / "bare_grd.dat"
    grd.write_text("Node Number = 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="Node/Cell Number"):
        FvcomDriver().read(str(grd))


def _solo_grid_text():
    """A minimal single-triangle _grd.dat body: three nodes and one cell."""
    return (
        "Node Number = 3\n"
        "Cell Number = 1\n"
        "1 1 2 3 0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
    )


def test_grid_without_siblings_has_no_depth_or_boundary(tmp_path):
    """A _grd.dat read with no _dep.dat or _obc.dat beside it carries neither."""
    grd = tmp_path / "solo_grd.dat"
    grd.write_text(_solo_grid_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = FvcomDriver().read(str(grd))
    assert dataset.get("node_z") is None
    assert "open_boundary_nodes" not in dataset.global_attributes


def test_depth_file_shorter_than_nodes_is_a_format_error(tmp_path):
    """A _dep.dat with fewer depth rows than the grid's nodes is a format error."""
    (tmp_path / "chan_grd.dat").write_text(_solo_grid_text())
    (tmp_path / "chan_dep.dat").write_text(
        "Node Number = 3\n0.0 0.0 10.0\n1.0 0.0 20.0\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="fewer than"):
        FvcomDriver().read(str(tmp_path / "chan_grd.dat"))


def test_can_open_declines_a_directory_without_a_grid(tmp_path):
    """A directory holding no _grd.dat resolves to no grid file and is declined."""
    (tmp_path / "readme.txt").write_text("not a grid")
    assert FvcomDriver().can_open(str(tmp_path)) is False
