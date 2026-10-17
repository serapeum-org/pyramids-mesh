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


class TestOptionalArraysAbsent:
    """The reader's optional-array branches, exercised without a crafted NetCDF file.

    ``node_Zb``, ``cell_X``/``cell_Y`` and ``ResTime`` are optional in the format. The real
    committed sample carries all three, so their *absent* branches are checked directly:
    each method consults the name set alone before touching the file, so an empty name set
    takes the absent path without reading any bytes -- no fabricated data, just the branch.
    """

    def test_node_z_is_none_when_node_zb_absent(self):
        """``_node_z`` returns ``None`` when the file carries no ``node_Zb``."""
        reader = _TuflowFvReader("absent.nc", lazy=True)
        assert reader._node_z(None, set()) is None

    def test_face_centres_are_none_when_cell_xy_absent(self):
        """``_face_centres`` returns ``(None, None)`` without both ``cell_X`` and ``cell_Y``."""
        reader = _TuflowFvReader("absent.nc", lazy=True)
        # only one of the pair present -> still declined
        assert reader._face_centres(None, {"cell_X"}) == (None, None)
        assert reader._face_centres(None, set()) == (None, None)

    def test_times_are_none_when_restime_absent(self):
        """``_times`` returns ``None`` when the ``ResTime`` coordinate is absent."""
        reader = _TuflowFvReader("absent.nc", lazy=True)
        assert reader._times(None, set()) is None


def test_require_refuses_a_file_missing_a_defining_array():
    """``_require`` raises when a defining array (node list / connectivity) is missing."""
    reader = _TuflowFvReader("partial.nc", lazy=True)
    with pytest.raises(FileFormatNotSupportedError, match="not a TUFLOW FV result"):
        reader._require({"node_X", "node_Y"})  # cell_node missing


class _FakeDim:
    """A stand-in for a GDAL dimension: just a name and a size."""

    def __init__(self, name: str, size: int = 7) -> None:
        self._name = name
        self._size = size

    def GetName(self) -> str:
        return self._name

    def GetSize(self) -> int:
        return self._size


class _FakeArray:
    """A stand-in for a GDAL MD-array that reports only its dimensions."""

    def __init__(self, dim_names: tuple[str, ...]) -> None:
        self._dims = [_FakeDim(name) for name in dim_names]

    def GetDimensions(self) -> list:
        return self._dims


class _FakeGroup:
    """A stand-in GDAL group that opens the fake arrays it was given."""

    def __init__(self, arrays: dict) -> None:
        self._arrays = arrays

    def OpenMDArray(self, name: str) -> _FakeArray:
        return self._arrays[name]


class TestVariableSelection:
    """``_variables`` selects face arrays and skips the rest -- the loop's two continues.

    Driving the selection logic with a fake group isolates the branch decisions from any
    file: no TUFLOW FV bytes are fabricated, only the shape of the decision is asserted.
    """

    def test_array_whose_last_dim_is_not_a_cell_dim_is_skipped(self):
        """A non-cell array (e.g. a coordinate) is not shipped as a face variable."""
        reader = _TuflowFvReader("fake.nc", lazy=True)
        group = _FakeGroup({"somewhere": _FakeArray(("Time", "Scalar"))})
        result = reader._variables(
            group, {"somewhere"}, {"NumCells2D": 5}, times=None, n_face=5
        )
        assert result == {}

    def test_cell_array_of_the_wrong_width_is_skipped(self):
        """A cell-dim array whose size is not the face count is skipped, not mislabelled."""
        reader = _TuflowFvReader("fake.nc", lazy=True)
        group = _FakeGroup({"bar": _FakeArray(("Time", "NumCells2D"))})
        result = reader._variables(
            group, {"bar"}, {"NumCells2D": 1}, times=None, n_face=2866
        )
        assert result == {}

    def test_static_face_array_has_no_time_axis(self):
        """A face array without a leading ``Time`` dimension carries no time values."""
        reader = _TuflowFvReader("fake.nc", lazy=True)
        group = _FakeGroup({"bed": _FakeArray(("NumCells2D",))})
        result = reader._variables(
            group, {"bed"}, {"NumCells2D": 7}, times=np.array([0.0]), n_face=7
        )
        assert set(result) == {"bed"}
        assert result["bed"].location == "face"
        assert result["bed"].time_values is None
