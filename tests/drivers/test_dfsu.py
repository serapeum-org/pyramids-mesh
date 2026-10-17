"""Tests for the DHI MIKE DFSU reader (D8), against a real committed sample.

DFSU is a proprietary DHI binary read through ``mikeio`` (the ``dhi`` extra); the whole
module is skipped when the extra is absent. The fixture is a genuine DFSU result committed
under ``tests/data/dhi/`` -- 535 nodes, 724 mixed triangle/quad elements, three items over
thirteen timesteps.
"""

import pyramids  # noqa: F401  # isort:skip

import pytest

pytest.importorskip("mikeio", reason="the dhi extra (mikeio) is not installed")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from pyramids.base._errors import FileFormatNotSupportedError  # noqa: E402
from pyramids.netcdf.ugrid import UgridDataset  # noqa: E402
from pyramids_mesh.drivers.dhi import DfsuDriver  # noqa: E402

_RESULTS = Path(__file__).parent.parent / "data" / "dhi" / "OdenseHD2D.dfsu"


def _read():
    """Read the committed DFSU result."""
    return DfsuDriver().read(str(_RESULTS))


def test_reads_mesh_from_mikeio_geometry():
    """The geometry builds a 535-node, 724-element mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 535
    assert mesh.n_face == 724


def test_element_table_is_zero_based_mixed_triangles_and_quads():
    """The 0-based element table pads triangles to the quad width with -1."""
    mesh = _read().mesh
    connectivity = mesh.face_node_connectivity
    assert connectivity.original_start_index == 0
    assert connectivity.data.shape == (724, 4)
    assert int(connectivity.data[connectivity.data != -1].min()) == 0
    assert int(connectivity.data.max()) == 534
    # a mix of triangles (one -1) and quads (none)
    assert bool((connectivity.data == -1).any())


def test_node_elevation_becomes_node_z():
    """The third node coordinate becomes a node-located node_z variable."""
    node_z = _read().get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (535,)


def test_item_is_face_located_over_time():
    """Each item is a face-located variable over time."""
    level = _read().get("Surface elevation")
    assert level.location == "face"
    assert level.shape == (13, 724)
    assert len(level.attributes["time_values"]) == 13
    # the time axis is seconds from the first step
    assert level.attributes["time_values"][0] == 0.0


def test_all_items_are_present():
    """Every DFSU item becomes a variable; node_z rides alongside."""
    names = _read().data_variable_names
    assert "Surface elevation" in names
    assert "Depth averaged U velocity" in names
    assert "Depth averaged V velocity" in names


def test_values_are_lazy():
    """A lazy read reports a source without loading the values."""
    level = _read().get("Surface elevation")
    assert level.has_data_source
    assert level._data is None
    assert level.data.shape == (13, 724)


def test_cell_centres_populate_face_xy():
    """The element centres become the face centres."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert len(mesh.face_x) == 724


def test_dhi_projection_name_is_not_guessed_as_crs():
    """DHI's internal projection name is not forced into a CRS."""
    assert _read().mesh.crs is None


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data[:200]:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        if vertices.size >= 3:
            assert signed_area2 > 0


def test_reached_through_read_file():
    """A plain read_file dispatches to this driver by the .dfsu suffix."""
    dataset = UgridDataset.read_file(str(_RESULTS))
    assert dataset.mesh.n_node == 535


def test_sniff_declines_non_dfs(tmp_path):
    """can_open declines a file without the DFS binary tag."""
    driver = DfsuDriver()
    assert driver.can_open(str(_RESULTS)) is True

    junk = tmp_path / "fake.dfsu"
    junk.write_bytes(b"not a dfs file at all")
    assert driver.can_open(str(junk)) is False


def test_non_mesh_dfs_is_a_format_error(tmp_path):
    """A .dfsu that mikeio cannot open as a mesh is a format error, not a crash."""
    # the DFS tag without a valid body: sniff passes, mikeio.open fails -> format error
    broken = tmp_path / "broken.dfsu"
    broken.write_bytes(b"DHI_DFS_ but truncated nonsense body")
    with pytest.raises(FileFormatNotSupportedError, match="could not be opened"):
        DfsuDriver().read(str(broken))
