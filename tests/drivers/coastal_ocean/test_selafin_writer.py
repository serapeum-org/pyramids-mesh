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
from pyramids.netcdf.ugrid import MeshDataset

from pyramids_mesh.drivers.coastal_ocean.telemac import SelafinDriver

_SLF = Path(__file__).parent.parent.parent / "data" / "minimal.slf"


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


def test_dropping_node_z_warns(tmp_path):
    """Dropping a carried-in ``node_z`` is announced, since Selafin cannot store it."""
    with_z = _read(_SLF).with_variable(
        "node_z", np.zeros(_read(_SLF).mesh.n_node), location="node"
    )
    with pytest.warns(UserWarning, match="node_z"):
        SelafinDriver().write(with_z, str(tmp_path / "warned.slf"))


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
    """The write reaches the driver through ``MeshDataset.to_file(driver="selafin")``."""
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


def _header_of(path):
    """Return ``(title, iparam, date, ipobo)`` read back from a written Selafin file."""
    from pyramids_mesh._binary import _read_record, detect_endianness, read_record_ints

    endian = detect_endianness(str(path), expect_first_record=80)
    with open(path, "rb") as handle:
        title = _read_record(handle, endian).decode("latin-1")
        counts = read_record_ints(handle, endian, 2)
        for _ in range(int(counts[0] + counts[1])):
            _read_record(handle, endian)
        iparam = read_record_ints(handle, endian, 10).tolist()
        date = read_record_ints(handle, endian, 6).tolist() if iparam[9] == 1 else None
        dims = read_record_ints(handle, endian, 4)
        read_record_ints(handle, endian, int(dims[0]) * int(dims[2]))
        ipobo = read_record_ints(handle, endian, int(dims[1])).tolist()
    return title[:72].strip(), iparam, date, ipobo


def test_the_writer_keeps_the_datasets_own_title(tmp_path):
    """A written file carries the dataset's title, not the package's placeholder.

    The title lands in the file's own identifying record, which every post-processor shows,
    so stamping a library name over the model's name loses real information.
    """
    src = _read(_SLF)
    src.global_attributes["title"] = "Rouse profile test"
    out = tmp_path / "titled.slf"
    src.to_file(str(out), driver="selafin")

    title, _iparam, _date, _ipobo = _header_of(out)
    assert title == "Rouse profile test"


def test_a_dataset_without_a_title_gets_the_neutral_default(tmp_path):
    """With nothing to carry, the title is the neutral package name."""
    src = _read(_SLF)
    src.global_attributes.pop("title", None)
    out = tmp_path / "untitled.slf"
    src.to_file(str(out), driver="selafin")

    title, _iparam, _date, _ipobo = _header_of(out)
    assert title == "pyramids-mesh"


def test_the_writer_restores_the_boundary_numbering_and_its_count(tmp_path):
    """IPOBO goes back into its own record, and IPARAM[7] is derived from it.

    TELEMAC reads the boundary numbering to attach boundary conditions; a file written with
    zeros there is geometry only, whatever mesh it carries.
    """
    src = _read(_SLF)
    src = MeshDataset.read_file(str(_SLF), driver="selafin")
    staged = tmp_path / "staged.slf"
    # Give the source a boundary numbering by writing one and reading it back.
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    write_selafin(
        str(staged),
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        ikle=[[1, 2, 3], [1, 3, 4]],
        variable_names=["WATER DEPTH     M               "],
        times=[0.0],
        values=[[[0.5, 0.25, 0.75, 0.125]]],
        ipobo=[1, 2, 0, 3],
    )
    src = _read(staged)
    out = tmp_path / "restored.slf"
    src.to_file(str(out), driver="selafin")

    _title, iparam, _date, ipobo = _header_of(out)
    assert ipobo == [1, 2, 0, 3]
    assert iparam[7] == 3


def test_writing_without_a_boundary_numbering_warns(tmp_path):
    """An in-memory mesh cannot have IPOBO derived for it, so the caller is told."""
    src = _read(_SLF)
    out = tmp_path / "geometry_only.slf"

    with pytest.warns(UserWarning, match="all-zero IPOBO"):
        src.to_file(str(out), driver="selafin")

    _title, iparam, _date, ipobo = _header_of(out)
    assert set(ipobo) == {0}
    assert iparam[7] == 0


def test_the_writer_restores_the_date_record_and_its_flag(tmp_path):
    """A preserved ``start_date`` is written back as the six-int record, flag included."""
    src = _read(_SLF)
    src.global_attributes["start_date"] = "1993-07-14T06:30:15"
    out = tmp_path / "dated.slf"
    src.to_file(str(out), driver="selafin")

    _title, iparam, date, _ipobo = _header_of(out)
    assert iparam[9] == 1
    assert date == [1993, 7, 14, 6, 30, 15]
    assert _read(out).global_attributes["start_date"] == "1993-07-14T06:30:15"


@pytest.mark.parametrize("start_date", ["not-a-date", "1993-07-14", 19930714, None])
def test_an_unusable_start_date_writes_no_date_record(tmp_path, start_date):
    """A malformed attribute writes no record rather than half of one.

    The flag and the record have to agree: setting the flag without a readable date would
    desynchronise every record after it.
    """
    src = _read(_SLF)
    if start_date is None:
        src.global_attributes.pop("start_date", None)
    else:
        src.global_attributes["start_date"] = start_date
    out = tmp_path / "undated.slf"
    src.to_file(str(out), driver="selafin")

    _title, iparam, date, _ipobo = _header_of(out)
    assert iparam[9] == 0
    assert date is None


def test_the_boundary_variable_is_not_written_as_a_result(tmp_path):
    """The carried boundary numbering must not become a sixth variable on write."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    staged = tmp_path / "staged.slf"
    write_selafin(
        str(staged),
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        ikle=[[1, 2, 3], [1, 3, 4]],
        variable_names=["WATER DEPTH     M               "],
        times=[0.0],
        values=[[[0.5, 0.25, 0.75, 0.125]]],
        ipobo=[1, 2, 0, 3],
    )
    src = _read(staged)
    assert "boundary_node_number" in src.variable_names

    out = tmp_path / "written.slf"
    src.to_file(str(out), driver="selafin")

    from pyramids_mesh._binary import _read_record, detect_endianness, read_record_ints

    endian = detect_endianness(str(out), expect_first_record=80)
    with open(out, "rb") as handle:
        _read_record(handle, endian)
        counts = read_record_ints(handle, endian, 2)
        names = [
            _read_record(handle, endian).decode("latin-1")[:16].strip()
            for _ in range(int(counts[0] + counts[1]))
        ]

    assert names == ["WATER DEPTH"]
    # It comes back on the re-read, because the writer restored the IPOBO record it belongs
    # in -- not because it was written as a result.
    assert _read(out).data_variable_names == [
        "WATER DEPTH",
        "boundary_node_number",
    ]


def test_write_selafin_refuses_a_boundary_record_of_the_wrong_length(tmp_path):
    """IPOBO is one entry per node; a short array would desynchronise the file."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    with pytest.raises(ValueError, match="one entry per node"):
        write_selafin(
            str(tmp_path / "bad.slf"),
            node_x=[0.0, 1.0, 1.0, 0.0],
            node_y=[0.0, 0.0, 1.0, 1.0],
            ikle=[[1, 2, 3], [1, 3, 4]],
            variable_names=[],
            times=[],
            values=[],
            ipobo=[1, 2],
        )


def test_write_selafin_refuses_a_date_of_the_wrong_length(tmp_path):
    """The date record is exactly six ints."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    with pytest.raises(ValueError, match="6 fields"):
        write_selafin(
            str(tmp_path / "bad.slf"),
            node_x=[0.0, 1.0, 1.0, 0.0],
            node_y=[0.0, 0.0, 1.0, 1.0],
            ikle=[[1, 2, 3], [1, 3, 4]],
            variable_names=[],
            times=[],
            values=[],
            date=[1993, 7, 14],
        )
