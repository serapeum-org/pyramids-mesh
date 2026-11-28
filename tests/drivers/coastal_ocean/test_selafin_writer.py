"""Tests for the Selafin writer (W2), round-tripping the committed sample.

The writer emits the same Fortran-record layout the D3 reader parses, so the contract is a
read -> write -> read round trip of ``tests/data/minimal.slf`` that preserves the mesh, the
connectivity base, the variable names/units and the time steps.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MeshDataset, UnsupportedMeshFormatError

from pyramids_mesh.drivers.coastal_ocean.telemac import SelafinDriver

from .selafin_header import read_header

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
    assert "node_z" not in back.variable_names
    assert "WATER DEPTH" in back.variable_names


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
    assert list(back.variable_names) == list(original.variable_names)
    name = list(original.variable_names)[0]
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
        {"mesh": fake_mesh, "variable_names": [], "time_values": None},
    )()
    with pytest.raises(FileFormatNotSupportedError, match="same node count"):
        SelafinDriver().write(dataset, str(tmp_path / "mixed.slf"))


def _header_of(path):
    """Return ``(title, iparam, date, ipobo)`` read back from a written Selafin file."""
    header = read_header(path)
    return header["title"], header["iparam"], header["date"], header["ipobo"]


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
    # Give the source a boundary numbering by writing one and reading it back.
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


@pytest.mark.parametrize("start_date", ["not-a-date", "1993-13-01T00:00:00", 19930714])
def test_an_unusable_start_date_writes_no_date_record_and_warns(tmp_path, start_date):
    """An unusable attribute writes no record rather than half of one, and says so.

    The flag and the record have to agree: setting the flag without a readable date would
    desynchronise every record after it. Dropping it silently is the worse failure, though
    -- the caller believes the file carries a date it will not -- so it warns.
    """
    src = _read(_SLF)
    src.global_attributes["start_date"] = start_date
    out = tmp_path / "undated.slf"

    with pytest.warns(UserWarning, match="without a date record"):
        src.to_file(str(out), driver="selafin")

    _title, iparam, date, _ipobo = _header_of(out)
    assert iparam[9] == 0, f"the date flag must stay clear, got {iparam[9]}"
    assert date is None


def test_an_absent_start_date_writes_no_record_and_stays_quiet(tmp_path):
    """No attribute at all is the ordinary case and must not warn.

    Only a present-but-unusable value is worth a warning; an in-memory mesh simply has no
    date, which is not a problem to report.
    """
    src = _read(_SLF)
    src.global_attributes.pop("start_date", None)
    out = tmp_path / "undated.slf"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src.to_file(str(out), driver="selafin")

    about_dates = [str(w.message) for w in caught if "date record" in str(w.message)]
    assert not about_dates, f"an absent start_date must not warn: {about_dates}"
    _title, iparam, date, _ipobo = _header_of(out)
    assert iparam[9] == 0
    assert date is None


def test_a_date_only_start_date_is_written_as_midnight(tmp_path):
    """``YYYY-MM-DD`` is a real instant, so it is written rather than dropped.

    ``datetime.fromisoformat`` accepts it, and refusing a value a date parser accepts would
    lose a date the dataset really carries.
    """
    src = _read(_SLF)
    src.global_attributes["start_date"] = "1993-07-14"
    out = tmp_path / "midnight.slf"
    src.to_file(str(out), driver="selafin")

    _title, iparam, date, _ipobo = _header_of(out)
    assert iparam[9] == 1
    assert date == [1993, 7, 14, 0, 0, 0], f"got {date}"
    assert _read(out).global_attributes["start_date"] == "1993-07-14T00:00:00"


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

    assert read_header(out)["names"] == ["WATER DEPTH"]
    # It comes back on the re-read, because the writer restored the IPOBO record it belongs
    # in -- not because it was written as a result.
    assert _read(out).variable_names == [
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


def test_boundary_numbering_of_the_wrong_length_is_ignored_with_a_warning():
    """A boundary variable that does not match the node count is not written.

    The reader always produces one entry per node, so this is the defensive path for a
    dataset assembled by hand or resized after a read: writing a short array into the IPOBO
    record would desynchronise every record after it, so it is dropped like an absent one.
    """
    variable = SimpleNamespace(data=np.array([1, 2]), location="node")
    dataset = SimpleNamespace(
        variable_names=["boundary_node_number"],
        get=lambda name: variable,
    )
    with pytest.warns(UserWarning, match="all-zero IPOBO"):
        numbering = SelafinDriver._boundary_numbering(dataset, 4, "mismatched.slf")

    assert numbering is None, f"a mismatched numbering must be dropped, got {numbering}"


def test_boundary_numbering_is_taken_when_it_matches_the_node_count():
    """A correctly sized boundary variable is passed through as plain ints."""
    variable = SimpleNamespace(data=np.array([1, 0, 2, 0]), location="node")
    dataset = SimpleNamespace(
        variable_names=["boundary_node_number"],
        get=lambda name: variable,
    )
    numbering = SelafinDriver._boundary_numbering(dataset, 4, "matched.slf")

    assert numbering == [1, 0, 2, 0], (
        f"expected the numbering verbatim, got {numbering}"
    )


@pytest.mark.parametrize(
    "start_date",
    [
        "abcd-ef-ghTij:kl:mn",
        "1993-07-14T06:30:xx",
        "----T::",
        "1993-00-00T12:00:00",
        "1993-07-14T99:99:99",
        19930714,
    ],
)
def test_date_fields_refuses_anything_that_is_not_an_instant(start_date):
    """A value that is not a calendar instant is refused, whatever its shape.

    Splitting on separators accepted all of these: the three month-0/hour-24 strings parse
    into six integers and would be written as a date record no reader could interpret.
    """
    with pytest.warns(UserWarning, match="without a date record"):
        resolved = SelafinDriver._date_fields(start_date, "out.slf")

    assert resolved is None, f"a non-instant must be refused, got {resolved}"


@pytest.mark.parametrize(
    "date_record",
    [
        [1993, 7, 14, 6, 30, 15],
        [1, 1, 1, 0, 0, 0],
        [9999, 12, 31, 23, 59, 59],
        [2024, 2, 29, 12, 0, 0],
    ],
)
def test_the_reader_and_the_writer_agree_on_every_instant(tmp_path, date_record):
    """Whatever the reader emits, the writer parses back to the same six ints.

    The two are each other's inverse, so the property that matters is the round trip rather
    than one hand-written happy case -- including a leap day and both ends of the year
    range ``datetime`` allows.
    """
    path = _write_dated(tmp_path, date_record)
    start_date = _read(path).global_attributes["start_date"]

    assert SelafinDriver._date_fields(start_date, "out.slf") == date_record


def test_derive_iparam_sets_the_boundary_count_and_date_flag():
    """IPARAM is derived from what is actually written, never asserted independently."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import _derive_iparam

    with_date = _derive_iparam(np.array([0, 4, 0, 7]), True)
    without = _derive_iparam(np.zeros(4, dtype=int), False)

    assert with_date == [1, 0, 0, 0, 0, 0, 0, 2, 0, 1], f"got {with_date}"
    assert without == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], f"got {without}"


def test_a_connectivity_without_a_recorded_base_is_written_1_based(tmp_path):
    """With no recorded base, IKLE is written 1-based -- the format's own convention.

    A mesh built in memory has never been through a Selafin read, so it carries no base to
    restore; assuming 0 would write node ids the format reserves.
    """
    source = _read(_SLF)
    conn = type(
        "_Conn",
        (),
        {
            "data": np.asarray(source.mesh.face_node_connectivity.data),
            "original_start_index": None,
        },
    )()
    mesh = type(
        "_Mesh",
        (),
        {
            "face_node_connectivity": conn,
            "node_x": np.asarray(source.mesh.node_x),
            "node_y": np.asarray(source.mesh.node_y),
        },
    )()
    dataset = type(
        "_Dataset",
        (),
        {
            "mesh": mesh,
            "variable_names": [],
            "variable_names": [],
            "global_attributes": {},
            "time_values": None,
            "get": staticmethod(lambda name: None),
        },
    )()

    out = tmp_path / "based.slf"
    with pytest.warns(UserWarning, match="all-zero IPOBO"):
        SelafinDriver().write(dataset, str(out))

    back = _read(out)
    assert back.mesh.n_node == source.mesh.n_node
    np.testing.assert_array_equal(
        np.asarray(back.mesh.face_node_connectivity.data),
        np.asarray(source.mesh.face_node_connectivity.data),
    )


def test_a_mesh_with_no_node_variables_writes_no_timesteps(tmp_path):
    """Geometry with nothing to report over time writes zero time records.

    The time axis exists only to carry variable slabs, so a mesh with no node variables gets
    none -- not one empty step.
    """
    source = _read(_SLF)
    dataset = type(
        "_Dataset",
        (),
        {
            "mesh": source.mesh,
            "variable_names": [],
            "variable_names": [],
            "global_attributes": {"title": "geometry only"},
            "time_values": None,
            "get": staticmethod(lambda name: None),
        },
    )()

    out = tmp_path / "geometry.slf"
    with pytest.warns(UserWarning, match="all-zero IPOBO"):
        SelafinDriver().write(dataset, str(out))

    title, _iparam, _date, _ipobo = _header_of(out)
    back = _read(out)
    steps = 0 if back.time_values is None else len(np.asarray(back.time_values))
    assert title == "geometry only", f"the title should survive, got {title!r}"
    assert back.variable_names == []
    assert steps == 0, f"a mesh with no variables should write no steps, got {steps}"


def test_a_non_node_located_node_z_is_left_alone(tmp_path):
    """The node_z warning is for a node-located field; a face-located one is not it.

    ``node_z`` is dropped because Selafin stores only x/y per node. A variable that happens
    to share the name but lives on faces is not that field, so it is neither warned about nor
    treated as geometry.
    """
    source = _read(_SLF)
    face_z = source.with_variable(
        "node_z", np.zeros(source.mesh.n_face), location="face"
    )

    out = tmp_path / "face_z.slf"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelafinDriver().write(face_z, str(out))

    # Matched on the message's own wording, not on "node_z": the IPOBO warning quotes the
    # output path, and tmp_path carries this test's name.
    about_z = [str(w.message) for w in caught if "bed " in str(w.message)]
    assert not about_z, f"unexpected node_z warning: {about_z}"
    assert _read(out).variable_names == ["WATER DEPTH"]


def _write_dated(tmp_path, date_record):
    """Write a minimal Selafin file carrying ``date_record`` and return its path."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    path = tmp_path / f"dated_{'_'.join(str(v) for v in date_record)}.slf"
    write_selafin(
        str(path),
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        ikle=[[1, 2, 3], [1, 3, 4]],
        variable_names=["WATER DEPTH     M               "],
        times=[0.0],
        values=[[[0.5, 0.25, 0.75, 0.125]]],
        date=date_record,
    )
    return path


def test_the_end_of_day_hour_is_normalised_rather_than_dropped(tmp_path):
    """A date record with hour 24 is ISO-8601's end of day, not a corrupt field.

    ``datetime()`` refuses it and ``datetime.fromisoformat()`` -- the writer's parser --
    rolls it into the next day. The reader normalises it the same way so the two agree:
    dropping what the other end accepts would lose a usable date.
    """
    path = _write_dated(tmp_path, [1993, 7, 14, 24, 0, 0])
    start_date = _read(path).global_attributes["start_date"]

    assert start_date == "1993-07-15T00:00:00", f"got {start_date}"
    assert SelafinDriver._date_fields(start_date, "out.slf") == [1993, 7, 15, 0, 0, 0]


def _write_with_origin(tmp_path, name="origin.slf"):
    """Write a Selafin file whose coordinates are small offsets from a projected origin."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    path = tmp_path / name
    write_selafin(
        str(path),
        node_x=[0.0, 1000.25, 1000.5, 0.75],
        node_y=[0.0, 0.5, 1000.125, 1000.0],
        ikle=[[1, 2, 3], [1, 3, 4]],
        variable_names=["WATER DEPTH     M               "],
        times=[0.0],
        values=[[[0.5, 0.25, 0.75, 0.125]]],
        iparam=[1, 0, 315000, 5345000, 0, 0, 0, 0, 0, 0],
    )
    return path


def test_the_coordinate_origin_reaches_the_dataset(tmp_path):
    """The origin is kept so a writer can restore the split the file used.

    The node coordinates are absolute, so the origin is not needed to place the mesh; it is
    needed to write the mesh back the way it was stored.
    """
    dataset = _read(_write_with_origin(tmp_path))

    assert dataset.global_attributes["x_origin"] == 315000
    assert dataset.global_attributes["y_origin"] == 5345000
    assert np.asarray(dataset.mesh.node_x)[0] == pytest.approx(315000.0)


def test_a_file_without_an_origin_declares_none(tmp_path):
    """A zero origin adds no attribute, so nothing suggests a split that is not there."""
    dataset = _read(_SLF)

    assert "x_origin" not in dataset.global_attributes
    assert "y_origin" not in dataset.global_attributes


def test_the_origin_is_restored_so_single_precision_stays_exact(tmp_path):
    """Writing offsets from the declared origin keeps `precision="single"` lossless here.

    Folding the origin into the coordinates and writing float32 spends the whole mantissa
    on the large constant: the same mesh lost 0.125 m in y before this. The source file
    stored those coordinates losslessly at float32 precisely because the constant lived in
    the integer origin, so the writer puts it back there.
    """
    source = _read(_write_with_origin(tmp_path))

    for precision in ("double", "single"):
        out = tmp_path / f"{precision}.slf"
        source.to_file(str(out), driver="selafin", precision=precision)
        back = _read(out)

        _title, iparam, _date, _ipobo = _header_of(out)
        assert iparam[2:4] == [315000, 5345000], f"{precision}: got {iparam[2:4]}"
        # Exact, not approximate: assert_allclose keeps rtol=1e-7 even at atol=0, which
        # at y around 5,345,000 is a 0.53 m tolerance -- wide enough for the 0.125 m loss
        # this test exists to catch.
        np.testing.assert_array_equal(
            np.asarray(source.mesh.node_x), np.asarray(back.mesh.node_x)
        )
        np.testing.assert_array_equal(
            np.asarray(source.mesh.node_y), np.asarray(back.mesh.node_y)
        )


def test_a_non_integer_origin_attribute_is_ignored(tmp_path):
    """A nonsense origin attribute writes a zero origin rather than failing the write."""
    source = _read(_SLF)
    source.global_attributes["x_origin"] = "not-a-number"
    source.global_attributes["y_origin"] = None
    out = tmp_path / "bad_origin.slf"

    with pytest.warns(UserWarning, match="all-zero IPOBO"):
        source.to_file(str(out), driver="selafin")

    _title, iparam, _date, _ipobo = _header_of(out)
    assert iparam[2:4] == [0, 0], f"got {iparam[2:4]}"
    np.testing.assert_allclose(
        np.asarray(_read(out).mesh.node_x), np.asarray(source.mesh.node_x)
    )


def test_an_ipobo_entry_too_large_for_int32_is_refused(tmp_path):
    """A numbering that cannot be stored is refused rather than narrowed.

    Every integer record in the file is int32, so ``2**31`` becomes ``-2147483648`` on the
    way in -- a plausible-looking boundary number rather than an obviously lost one.
    """
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    with pytest.raises(ValueError, match="fit in the int32"):
        write_selafin(
            str(tmp_path / "wrapped.slf"),
            node_x=[0.0, 1.0, 1.0, 0.0],
            node_y=[0.0, 0.0, 1.0, 1.0],
            ikle=[[1, 2, 3], [1, 3, 4]],
            variable_names=[],
            times=[],
            values=[],
            ipobo=[1, 2, 0, 2**31],
        )


def test_an_explicit_date_flag_without_a_date_is_refused(tmp_path):
    """A flag with no record behind it would leave every later record one short."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    with pytest.raises(ValueError, match="no date was given"):
        write_selafin(
            str(tmp_path / "flagged.slf"),
            node_x=[0.0, 1.0, 1.0, 0.0],
            node_y=[0.0, 0.0, 1.0, 1.0],
            ikle=[[1, 2, 3], [1, 3, 4]],
            variable_names=[],
            times=[],
            values=[],
            iparam=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        )


def test_a_date_under_a_cleared_explicit_flag_is_refused(tmp_path):
    """The inverse contradiction is refused too, rather than silently dropping the date."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    with pytest.raises(ValueError, match="clears the date flag"):
        write_selafin(
            str(tmp_path / "unflagged.slf"),
            node_x=[0.0, 1.0, 1.0, 0.0],
            node_y=[0.0, 0.0, 1.0, 1.0],
            ikle=[[1, 2, 3], [1, 3, 4]],
            variable_names=[],
            times=[],
            values=[],
            iparam=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            date=[1993, 7, 14, 6, 30, 15],
        )


def test_a_face_located_boundary_variable_is_not_written_as_ipobo(tmp_path):
    """IPOBO is one entry per node, so a same-named face field is not it.

    The warning has to say which problem it hit: "carries no numbering" would send the
    reader looking for a missing variable rather than a mislocated one.
    """
    variable = SimpleNamespace(data=np.zeros(2), location="face")
    dataset = SimpleNamespace(
        variable_names=["boundary_node_number"],
        get=lambda name: variable,
    )

    with pytest.warns(UserWarning, match="is face-located"):
        numbering = SelafinDriver._boundary_numbering(dataset, 4, "faces.slf")

    assert numbering is None


def test_a_wrong_sized_boundary_variable_reports_the_sizes(tmp_path):
    """The warning names the mismatch rather than claiming the numbering is absent."""
    variable = SimpleNamespace(data=np.array([1, 2]), location="node")
    dataset = SimpleNamespace(
        variable_names=["boundary_node_number"],
        get=lambda name: variable,
    )

    with pytest.warns(UserWarning, match="has 2 entries for 4 nodes"):
        SelafinDriver._boundary_numbering(dataset, 4, "short.slf")


def test_a_datetime_start_date_is_written_as_the_date_record(tmp_path):
    """A ``datetime`` attribute is written as-is rather than refused for not being a string.

    The reader publishes ISO-8601 text, but a caller assembling a dataset by hand is more
    likely to hold a ``datetime``, and it needs no parsing to be usable.
    """
    from datetime import datetime

    src = _read(_SLF)
    src.global_attributes["start_date"] = datetime(1993, 7, 14, 6, 30, 15)
    out = tmp_path / "from_datetime.slf"
    src.to_file(str(out), driver="selafin")

    _title, iparam, date, _ipobo = _header_of(out)
    assert iparam[9] == 1
    assert date == [1993, 7, 14, 6, 30, 15], f"got {date}"


def _intact_destination(tmp_path, name="existing.slf"):
    """Write a valid Selafin file to act as a destination that must survive a failed write."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    path = tmp_path / name
    write_selafin(
        str(path),
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        ikle=[[1, 2, 3], [1, 3, 4]],
        variable_names=[],
        times=[],
        values=[],
    )
    return path, path.stat().st_size


def test_an_over_long_title_is_truncated_so_the_file_stays_readable(tmp_path):
    """The title must not grow into the format marker in the fixed-width title record.

    73 characters pushed ``SERAFIN``/``SERAFIND`` off the end of the 80-byte record, and
    since the marker is what declares the float width, the file then failed to read with a
    record-length error from a completely different part of the file.
    """
    src = _read(_SLF)
    src.global_attributes["title"] = "T" * 73
    out = tmp_path / "long_title.slf"

    with pytest.warns(UserWarning, match="truncated to 72 characters"):
        src.to_file(str(out), driver="selafin")

    header = read_header(out)
    assert header["marker"] == "SERAFIND", (
        f"the marker must survive, got {header['marker']!r}"
    )
    assert header["title"] == "T" * 72
    assert _read(out).mesh.n_node == 4, "the written file must still read"


def test_a_title_latin_1_cannot_hold_is_replaced_not_raised(tmp_path):
    """An unencodable character is replaced, and the destination is never truncated.

    The title record is latin-1, which has no en dash. Encoding inside the opened output
    left a valid 356-byte file at the destination as 0 bytes.
    """
    src = _read(_SLF)
    src.global_attributes["title"] = "Rouse \u2013 profile"
    out, size = _intact_destination(tmp_path)

    with pytest.warns(UserWarning, match="characters replaced in its title"):
        src.to_file(str(out), driver="selafin")

    assert out.stat().st_size > size, (
        "the write should have replaced the file, not emptied it"
    )
    assert _read(out).global_attributes["title"] == "Rouse ? profile"


def test_a_title_of_the_wrong_type_falls_back_to_the_default(tmp_path):
    """A non-string title warns and uses the default rather than raising a bare TypeError."""
    src = _read(_SLF)
    src.global_attributes["title"] = ["a", "b"]
    out = tmp_path / "listed.slf"

    with pytest.warns(UserWarning, match="is a list"):
        src.to_file(str(out), driver="selafin")

    assert read_header(out)["title"] == "pyramids-mesh"


def test_an_origin_outside_int32_is_refused_before_the_file_is_touched(tmp_path):
    """The origin is range-checked with ipobo, so a bad one cannot half-write a file.

    Reaching ``np.array`` inside the opened output raised OverflowError and left a 104-byte
    stub where a valid file had been.
    """
    src = _read(_SLF)
    src.global_attributes["x_origin"] = 2**31
    src.global_attributes["y_origin"] = 0
    out, size = _intact_destination(tmp_path, "origin.slf")

    # to_file wraps the driver's ValueError, which is what a caller going through the
    # facade actually sees; the driver's own message is carried inside it.
    with pytest.raises(UnsupportedMeshFormatError, match="must fit in the int32"):
        src.to_file(str(out), driver="selafin")

    assert out.stat().st_size == size, (
        "a refused write must leave the destination alone"
    )


def test_a_timezone_aware_start_date_keeps_the_instant(tmp_path):
    """The record has no offset field, so the instant is converted rather than shifted.

    Dropping ``+02:00`` would move the time axis by two hours and say nothing about it.
    """
    from datetime import datetime, timedelta, timezone

    src = _read(_SLF)
    src.global_attributes["start_date"] = datetime(
        1993, 7, 14, 6, 30, 15, tzinfo=timezone(timedelta(hours=2))
    )
    out = tmp_path / "tz.slf"

    with pytest.warns(UserWarning, match="converted to UTC"):
        src.to_file(str(out), driver="selafin")

    assert read_header(out)["date"] == [1993, 7, 14, 4, 30, 15]


def test_a_sub_second_start_date_warns_that_it_is_dropped(tmp_path):
    """The record stores whole seconds, so losing the microseconds is announced."""
    from datetime import datetime

    src = _read(_SLF)
    src.global_attributes["start_date"] = datetime(1993, 7, 14, 6, 30, 15, 123456)
    out = tmp_path / "micro.slf"

    with pytest.warns(UserWarning, match="sub-second part"):
        src.to_file(str(out), driver="selafin")

    assert read_header(out)["date"] == [1993, 7, 14, 6, 30, 15]


def test_an_explicit_iparam_plus_an_origin_is_refused(tmp_path):
    """Passing both would mean one of two IPARAM[2]/[3] sources silently losing.

    An explicit ``iparam`` already carries the origin, so the two can disagree; the caller
    knows which they meant and ``write_selafin`` does not. Refused like the analogous
    ``date``/``iparam`` clash rather than discarding the argument without a word.
    """
    from pyramids_mesh.drivers.coastal_ocean.telemac import write_selafin

    with pytest.raises(ValueError, match="not both"):
        write_selafin(
            str(tmp_path / "clash.slf"),
            node_x=[0.0, 1.0, 1.0, 0.0],
            node_y=[0.0, 0.0, 1.0, 1.0],
            ikle=[[1, 2, 3], [1, 3, 4]],
            variable_names=[],
            times=[],
            values=[],
            iparam=[1, 0, 315000, 5345000, 0, 0, 0, 0, 0, 0],
            origin=(315000, 5345000),
        )
