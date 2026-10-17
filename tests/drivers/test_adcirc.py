"""Tests for the ADCIRC reader (D12), against a real committed sample.

The fixture is the genuine ADCIRC "Shinnecock Inlet" example committed under
``tests/data/adcirc/`` (CC0-1.0, NOAA): ``fort.14`` (3070 nodes, 5780 triangles) and the
matching ``fort.63.nc`` (a 57-step sea-surface-elevation series on the nodes). The tests
read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.adcirc import AdcircDriver

_DATA = Path(__file__).parent.parent / "data" / "adcirc"
_GRID = _DATA / "fort.14"


def _read(path=_GRID):
    """Read the committed ADCIRC grid (and its fort.63.nc results)."""
    return AdcircDriver().read(str(path))


def test_reads_nodes_and_triangles():
    """fort.14 builds a 3070-node, 5780-triangle mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 3070
    assert mesh.n_face == 5780
    assert mesh.face_node_connectivity.data.shape == (5780, 3)


def test_connectivity_is_one_based_resolved_to_zero_based():
    """The 1-based element table arrives 0-based, with the base recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 3069


def test_node_depth_becomes_node_z():
    """The fort.14 node depth column becomes a node-located node_z variable."""
    dataset = _read()
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (3070,)
    # node 1 of the committed grid: depth 4.2878041267
    assert float(node_z.data[0]) == pytest.approx(4.2878041267, rel=1e-6)


def test_node_coordinates_match_the_grid():
    """Node 1's lon/lat match the committed fort.14."""
    mesh = _read().mesh
    assert float(mesh.node_x[0]) == pytest.approx(-72.0576782709, rel=1e-9)
    assert float(mesh.node_y[0]) == pytest.approx(40.9902316949, rel=1e-9)


def test_zeta_is_node_located_over_time():
    """The fort.63.nc elevation series attaches as a node variable over 57 steps."""
    zeta = _read().get("zeta")
    assert zeta.location == "node"
    assert zeta.shape == (57, 3070)
    assert len(zeta.attributes["time_values"]) == 57
    assert zeta.data.shape == (57, 3070)


def test_values_are_lazy():
    """A lazy read reports a source without loading the values."""
    zeta = _read().get("zeta")
    assert zeta.has_data_source
    assert zeta._data is None
    assert zeta.data.shape == (57, 3070)


def test_faces_are_counter_clockwise():
    """Every sampled face comes back counter-clockwise after validation."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data[:300]:
        xs = mesh.node_x[row]
        ys = mesh.node_y[row]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_bare_mesh_without_results(tmp_path):
    """Without a fort.63.nc sibling, the grid reads as a bare mesh (node_z only)."""
    grid = tmp_path / "fort.14"
    grid.write_bytes(_GRID.read_bytes())
    dataset = AdcircDriver().read(str(grid))
    assert dataset.mesh.n_face == 5780
    assert dataset.get("node_z") is not None
    assert "zeta" not in dataset.data_variable_names


def test_reached_through_a_directory(tmp_path):
    """A directory holding fort.14 + fort.63.nc reads via driver= on the directory."""
    (tmp_path / "fort.14").write_bytes(_GRID.read_bytes())
    (tmp_path / "fort.63.nc").write_bytes((_DATA / "fort.63.nc").read_bytes())
    dataset = AdcircDriver().read(str(tmp_path))
    assert dataset.mesh.n_node == 3070
    assert dataset.get("zeta").shape == (57, 3070)


def test_can_open_accepts_grid_file_and_directory():
    """can_open accepts the fort.14 file and its directory."""
    driver = AdcircDriver()
    assert driver.can_open(str(_GRID)) is True
    assert driver.can_open(str(_DATA)) is True


def test_can_open_declines_non_adcirc(tmp_path):
    """can_open declines a non-fort.14 file, an empty dir, and a plain text file."""
    driver = AdcircDriver()
    other = tmp_path / "notes.14"
    other.write_text("not an adcirc grid\n", encoding="utf-8")
    assert driver.can_open(str(other)) is False
    assert driver.can_open(str(tmp_path)) is False
    plain = tmp_path / "notes.txt"
    plain.write_text("hello")
    assert driver.can_open(str(plain)) is False


def test_non_triangular_element_is_refused(tmp_path):
    """A non-triangular (type 4) element is refused rather than guessed."""
    grid = tmp_path / "fort.14"
    grid.write_text(
        "quad grid\n"
        "1  4\n"
        "1  0.0  0.0  1.0\n"
        "2  1.0  0.0  1.0\n"
        "3  1.0  1.0  1.0\n"
        "4  0.0  1.0  1.0\n"
        "1  4  1  2  3  4\n",
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="only reads triangular"):
        AdcircDriver().read(str(grid))


def test_read_of_a_non_grid_path_is_a_format_error(tmp_path):
    """read() on a path that is neither a fort.14 nor a directory holding one is refused."""
    other = tmp_path / "notes.txt"
    other.write_text("not a grid")
    with pytest.raises(FileFormatNotSupportedError, match="not an ADCIRC fort.14"):
        AdcircDriver().read(str(other))


def test_declares_read_mesh_and_lazy():
    """The driver advertises read_mesh, read_data and lazy_data."""
    assert set(AdcircDriver.capabilities) == {"read_mesh", "read_data", "lazy_data"}
