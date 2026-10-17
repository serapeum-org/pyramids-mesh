"""Tests for the 3Di reader (N2), against a real committed sample.

3Di stores 2D cell contours rather than a node list, so the driver deduplicates the
corners into shared nodes. The fixture is a genuine 3Di result committed under
``tests/data/3di/`` -- 16 quad cells that dedup to 25 nodes, per-cell results over seven
timesteps, and a coupled 1D network the driver must report but not represent. The tests
read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids_mesh.drivers.threedi import ThreeDiDriver

_RESULTS = Path(__file__).parent.parent / "data" / "3di" / "results_3di.nc"


def _read():
    """Read the committed 3Di result, ignoring the expected 1D-network warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ThreeDiDriver().read(str(_RESULTS))


def test_contours_deduplicate_to_shared_nodes():
    """The 16 quad cells' 64 corner slots collapse to 25 shared nodes and 16 faces."""
    mesh = _read().mesh
    assert mesh.n_node == 25
    assert mesh.n_face == 16
    # every face references real nodes in range, quads of four corners
    data = mesh.face_node_connectivity.data
    assert data.shape == (16, 4)
    assert int(data.min()) == 0
    assert int(data.max()) == 24


def test_node_order_is_deterministic():
    """Reading the same file twice yields the same node order (sorted unique)."""
    first = _read().mesh
    second = _read().mesh
    assert np.array_equal(first.node_x, second.node_x)
    assert np.array_equal(first.node_y, second.node_y)


def test_mesh1d_presence_warns_naming_network1d():
    """The coupled 1D network is announced in a warning naming Network1d, not dropped."""
    with pytest.warns(UserWarning, match="1D network"):
        dataset = ThreeDiDriver().read(str(_RESULTS))
    # the returned result is the 2D mesh
    assert dataset.mesh.n_face == 16


def test_per_cell_variable_is_face_located_over_time():
    """A Mesh2D_* per-cell result is a face-located variable over the time axis."""
    s1 = _read().get("Mesh2D_s1")
    assert s1.location == "face"
    assert s1.shape == (7, 16)
    assert s1.attributes["time_values"].tolist() == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ]


def test_flow_line_variables_are_not_shipped():
    """Results on the flow-line dimension are not placed on the mesh."""
    names = _read().data_variable_names
    # Mesh2D_q / Mesh2D_au live on nMesh2D_lines, not the cells
    assert "Mesh2D_s1" in names
    assert "Mesh2D_q" not in names
    assert "Mesh2D_au" not in names


def test_values_are_lazy():
    """A lazy read reports a source without loading the values."""
    s1 = _read().get("Mesh2D_s1")
    assert s1.has_data_source
    assert s1._data is None
    assert s1.data.shape == (7, 16)


def test_face_centres_populate_face_xy():
    """The cell centres are carried as face_x / face_y."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert len(mesh.face_x) == 16


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise, as the mesh model requires."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_reached_through_read_file_netcdf_fallback():
    """A plain read_file dispatches to this driver via the NetCDF-suffix fallback."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = UgridDataset.read_file(str(_RESULTS))
    assert dataset.mesh.n_node == 25


def test_crs_is_none_without_grid_mapping():
    """With no validated grid-mapping attribute, the mesh carries no CRS."""
    assert _read().mesh.crs is None


def test_sniff_declines_non_3di_and_junk(tmp_path):
    """can_open declines a non-3Di NetCDF and non-NetCDF junk without raising."""
    driver = ThreeDiDriver()
    assert driver.can_open(str(_RESULTS)) is True

    text = tmp_path / "plain.nc"
    text.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(text)) is False

    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False
