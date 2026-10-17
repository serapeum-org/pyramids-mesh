"""Tests for the TUFLOW FV reader (N3, 2D slice), against a real committed sample.

The fixture is a genuine TUFLOW FV result committed under ``tests/data/tuflowfv/`` --
1497 nodes, 2866 cells (triangles padded in a quad-width table), per-cell results over
sixteen timesteps and the ``stat`` wet/dry status. It is single-layer (2D). The 3D
refusal is checked by exercising the guard directly, since a real 3D sample is several MB.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset
from pyramids_mesh.drivers.tuflowfv import TuflowFvDriver, _TuflowFvReader

_RESULTS = Path(__file__).parent.parent / "data" / "tuflowfv" / "Cone_003.nc"


def _read():
    """Read the committed TUFLOW FV result."""
    return TuflowFvDriver().read(str(_RESULTS))


def test_reads_topology_from_nodes_and_cell_node():
    """The node list and cell connectivity build a 1497-node, 2866-face mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 1497
    assert mesh.n_face == 2866


def test_cell_node_is_one_based_with_zero_padding():
    """The 1-based connectivity arrives 0-based; a 0 (triangle pad) becomes -1 fill."""
    mesh = _read().mesh
    connectivity = mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.max()) == 1496
    # the quad-width table holds triangles, padded with -1
    assert bool((connectivity.data == -1).any())
    real = connectivity.data[connectivity.data != -1]
    assert int(real.min()) == 0


def test_node_elevation_becomes_node_z():
    """node_Zb becomes a node-located node_z variable."""
    dataset = _read()
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (1497,)


def test_cell_centres_populate_face_xy():
    """cell_X/cell_Y become the face centres."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert len(mesh.face_x) == 2866


def test_per_cell_variable_is_face_located_over_time():
    """A per-cell result is a face-located variable over the ResTime axis."""
    depth = _read().get("D")
    assert depth.location == "face"
    assert depth.shape == (16, 2866)
    assert len(depth.attributes["time_values"]) == 16


def test_stat_is_carried_as_a_face_variable_not_folded():
    """The wet/dry status is a standalone face variable, not folded into the values."""
    dataset = _read()
    stat = dataset.get("stat")
    assert stat.location == "face"
    assert stat.shape == (16, 2866)
    # the depth values are not overwritten with NaN where stat is dry
    assert not bool(np.isnan(dataset.get("D").data).all())


def test_values_are_lazy():
    """A lazy read reports a source without loading the values."""
    depth = _read().get("D")
    assert depth.has_data_source
    assert depth._data is None
    assert depth.data.shape == (16, 2866)


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data[:200]:  # a sample is enough
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        if vertices.size >= 3:
            assert signed_area2 > 0


def test_reached_through_read_file_netcdf_fallback():
    """A plain read_file dispatches to this driver via the NetCDF-suffix fallback."""
    dataset = UgridDataset.read_file(str(_RESULTS))
    assert dataset.mesh.n_node == 1497


def test_layered_3d_file_is_refused():
    """A layered 3D file (more 3D cells than 2D) is refused, naming the limitation.

    Exercised against the guard directly: a real 3D sample is several megabytes, too
    large to commit for one refusal, and this is the exact condition that marks 3D.
    """
    reader = _TuflowFvReader("layered.nc", lazy=True)
    with pytest.raises(FileFormatNotSupportedError, match="layered 3D"):
        reader._guard_3d({"NumCells2D": 100, "NumCells3D": 300})


def test_single_layer_is_not_refused():
    """A single-layer file (NumCells3D == NumCells2D) is a 2D mesh and is accepted."""
    reader = _TuflowFvReader("flat.nc", lazy=True)
    reader._guard_3d({"NumCells2D": 100, "NumCells3D": 100})  # no raise


def test_sniff_declines_non_tuflow_and_junk(tmp_path):
    """can_open declines a non-TUFLOW NetCDF and non-NetCDF junk without raising."""
    driver = TuflowFvDriver()
    assert driver.can_open(str(_RESULTS)) is True

    junk = tmp_path / "plain.nc"
    junk.write_bytes(b"\x00 not netcdf \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "d.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False
