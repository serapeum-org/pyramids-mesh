"""Tests for the Selafin writer (W2), round-tripping the committed sample.

The writer emits the same Fortran-record layout the D3 reader parses, so the contract is a
read -> write -> read round trip of ``tests/data/minimal.slf`` that preserves the mesh, the
connectivity base, the variable names/units and the time steps.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset
from pyramids_mesh.drivers.telemac import SelafinDriver

_SLF = Path(__file__).parent.parent / "data" / "minimal.slf"


def _read(path):
    """Read a Selafin file into a dataset."""
    return SelafinDriver().read(str(path))


def test_node_z_is_geometry_not_a_written_data_variable(tmp_path):
    """A carried-in ``node_z`` is mesh geometry; the writer must not emit it as a result."""
    src = _read(_SLF)
    with_z = src.with_variable("node_z", np.zeros(src.mesh.n_node), location="node")
    out = tmp_path / "with_z.slf"
    SelafinDriver().write(with_z, str(out))
    back = _read(str(out))
    assert "node_z" not in back.data_variable_names
    assert "WATER DEPTH" in back.data_variable_names


def test_node_series_refuses_a_variable_whose_time_axis_differs():
    """A node variable with fewer steps than the dataset is refused, not index-errored."""
    short = SimpleNamespace(data=np.zeros((3, 4)))
    with pytest.raises(FileFormatNotSupportedError, match="time axis"):
        SelafinDriver._node_series([("EXTRA", short)], 2)


def test_node_series_builds_one_block_per_step_when_aligned():
    """An aligned node variable yields one value block per time step."""
    var = SimpleNamespace(data=np.arange(8, dtype=float).reshape(2, 4))
    blocks = SelafinDriver._node_series([("D", var)], 2)
    assert len(blocks) == 2
    assert blocks[0][0] == [0.0, 1.0, 2.0, 3.0]


def test_capabilities_declare_write():
    """The driver advertises the writer capabilities alongside the reader ones."""
    caps = set(SelafinDriver.capabilities)
    assert {"read_mesh", "read_data", "lazy_data", "write_mesh", "write_data"} == caps


def test_round_trip_preserves_mesh_and_results(tmp_path):
    """read -> write -> read keeps nodes, connectivity (and its base), vars and times."""
    original = _read(_SLF)
    out = tmp_path / "round_trip.slf"
    SelafinDriver().write(original, str(out))
    back = _read(str(out))

    assert back.mesh.n_node == original.mesh.n_node
    assert back.mesh.n_face == original.mesh.n_face
    np.testing.assert_allclose(back.mesh.node_x, original.mesh.node_x)
    np.testing.assert_allclose(back.mesh.node_y, original.mesh.node_y)
    np.testing.assert_array_equal(
        back.mesh.face_node_connectivity.data,
        original.mesh.face_node_connectivity.data,
    )
    assert (
        back.mesh.face_node_connectivity.original_start_index
        == original.mesh.face_node_connectivity.original_start_index
    )
    assert list(back.data_variable_names) == list(original.data_variable_names)
    name = list(original.data_variable_names)[0]
    assert back.get(name).units == original.get(name).units
    np.testing.assert_allclose(back.get(name).data, original.get(name).data)
    np.testing.assert_allclose(list(back.time_values), list(original.time_values))


def test_round_trip_through_to_file_dispatch(tmp_path):
    """The write reaches the driver through ``UgridDataset.to_file(driver="selafin")``."""
    original = _read(_SLF)
    out = tmp_path / "via_facade.slf"
    original.to_file(str(out), driver="selafin")
    assert _read(str(out)).mesh.n_node == original.mesh.n_node


def test_single_precision_option_is_lossy_but_close(tmp_path):
    """``precision="single"`` writes float32; coordinates survive to single tolerance."""
    original = _read(_SLF)
    out = tmp_path / "single.slf"
    SelafinDriver().write(original, str(out), precision="single")
    back = _read(str(out))
    np.testing.assert_allclose(back.mesh.node_x, original.mesh.node_x, rtol=1e-6)
    np.testing.assert_allclose(back.mesh.node_y, original.mesh.node_y, rtol=1e-6)


def test_unknown_precision_is_refused(tmp_path):
    """A precision other than single/double is refused rather than silently defaulted."""
    original = _read(_SLF)
    with pytest.raises(FileFormatNotSupportedError, match="precision"):
        SelafinDriver().write(original, str(tmp_path / "x.slf"), precision="quad")


def test_write_refuses_a_dataset_without_faces():
    """A faces-less dataset cannot be written as a Selafin mesh."""
    mesh = type("_Mesh", (), {"face_node_connectivity": None})()
    dataset = type("_Dataset", (), {"mesh": mesh})()
    with pytest.raises(FileFormatNotSupportedError, match="no face connectivity"):
        SelafinDriver().write(dataset, "unused.slf")


def test_write_refuses_mixed_element_widths(tmp_path):
    """Selafin needs a uniform element width, so a mixed tri/quad mesh is refused."""
    mesh = _read(_SLF).mesh
    # a quad-width connectivity table holding a triangle (padded -1) and a quad
    quad_and_tri = type(
        "_Conn",
        (),
        {
            "data": np.array([[0, 1, 2, -1], [0, 1, 2, 3]]),
            "original_start_index": 1,
        },
    )()
    fake_mesh = type(
        "_Mesh",
        (),
        {
            "face_node_connectivity": quad_and_tri,
            "node_x": mesh.node_x,
            "node_y": mesh.node_y,
        },
    )()
    dataset = type(
        "_Dataset",
        (),
        {"mesh": fake_mesh, "data_variable_names": [], "time_values": None},
    )()
    with pytest.raises(FileFormatNotSupportedError, match="same node count"):
        SelafinDriver().write(dataset, str(tmp_path / "mixed.slf"))
