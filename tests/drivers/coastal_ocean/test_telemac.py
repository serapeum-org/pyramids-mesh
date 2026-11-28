"""Tests for the TELEMAC Selafin reader (D3).

Selafin is the first binary driver and the one that proves the lazy path, so the tests
exercise the endianness probe, the single/double precision split, the coordinate-origin
trap (wrong and undetectable from the data), the 3D and parallel guards, and a lazy read
that defers the variable slabs. Fixtures come from ``write_selafin`` so one writer of the
binary layout is exercised across every variant.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest
from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.coastal_ocean import telemac
from pyramids_mesh.drivers.coastal_ocean.telemac import (
    SelafinDriver,
    build_minimal_fixture_selafin,
    write_selafin,
)

_NODE_X = [0.0, 1.0, 1.0, 0.0]
_NODE_Y = [0.0, 0.0, 1.0, 1.0]
_IKLE = [[1, 2, 3], [1, 3, 4]]
_NAMES = ["WATER DEPTH     M               "]
_TIMES = [0.0, 3600.0]
_VALUES = [[[0.5, 0.25, 0.75, 0.125]], [[1.5, 1.25, 1.75, 1.125]]]


def _write(tmp_path, name="model.slf", **overrides):
    """Write a Selafin fixture, overriding any ``write_selafin`` keyword."""
    kwargs = dict(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        ikle=_IKLE,
        variable_names=_NAMES,
        times=_TIMES,
        values=_VALUES,
    )
    kwargs.update(overrides)
    path = str(tmp_path / name)
    write_selafin(path, **kwargs)
    return path


@pytest.mark.parametrize("endian", ["<", ">"])
def test_selafin_detects_little_and_big_endian(tmp_path, endian):
    """A file reads identically whichever byte order it was written in."""
    dataset = SelafinDriver().read(_write(tmp_path, endian=endian))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_selafin_rejects_non_selafin_file(tmp_path):
    """A file whose first record is not an 80-byte title is declined, not read."""
    path = tmp_path / "other.slf"
    path.write_bytes(b"\x10\x00\x00\x00not a selafin title at all")

    assert SelafinDriver().can_open(str(path)) is False
    with pytest.raises(FileFormatNotSupportedError):
        SelafinDriver().read(str(path))


def test_selafin_double_precision_variant_reads(tmp_path):
    """A SERAFIND file is read as float64 for every float record."""
    dataset = SelafinDriver().read(_write(tmp_path, double=True))

    variable = dataset.get("WATER DEPTH")
    assert variable.data[0].tolist() == [0.5, 0.25, 0.75, 0.125]
    assert np.allclose(dataset.mesh.node_x, _NODE_X)


def test_selafin_coordinate_origin_is_applied(tmp_path):
    """IPARAM origins are added to the coordinates -- the trap that is silent if missed.

    The stored coordinates are offsets from the origin; read without adding it, the mesh
    sits near zero instead of at its real location, and nothing in the data could tell.
    """
    iparam = [1, 0, 1000, 2000, 0, 0, 0, 0, 0, 0]
    dataset = SelafinDriver().read(_write(tmp_path, iparam=iparam))

    assert dataset.mesh.node_x.tolist() == [1000.0, 1001.0, 1001.0, 1000.0]
    assert dataset.mesh.node_y.tolist() == [2000.0, 2000.0, 2001.0, 2001.0]


def test_selafin_ikle_is_zero_based_after_read(tmp_path):
    """The 1-based IKLE arrives 0-based, with the file's base recorded for a writer."""
    dataset = SelafinDriver().read(_write(tmp_path))

    connectivity = dataset.mesh.face_node_connectivity
    assert int(connectivity.data.min()) == 0
    assert connectivity.original_start_index == 1


def test_selafin_quad_ndp4_reads(tmp_path):
    """A four-node element block reads as quad faces."""
    path = _write(
        tmp_path,
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        ikle=[[1, 2, 3, 4]],
        values=[[[0.5, 0.25, 0.75, 0.125]], [[1.5, 1.25, 1.75, 1.125]]],
    )
    dataset = SelafinDriver().read(path)

    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize("date_flag", [0, 1])
def test_selafin_date_record_present_and_absent(tmp_path, date_flag):
    """The file reads whether or not the optional date record is present.

    The flag and the record have to agree, so the date is passed exactly when the flag is
    set -- a flag with nothing behind it would leave every later record one short.
    """
    iparam = [1, 0, 0, 0, 0, 0, 0, 0, 0, date_flag]
    date = [2000, 1, 1, 0, 0, 0] if date_flag else None
    dataset = SelafinDriver().read(_write(tmp_path, iparam=iparam, date=date))

    assert dataset.mesh.n_node == 4
    assert dataset.get("WATER DEPTH").shape == (2, 4)


@pytest.mark.parametrize("planes", [0, 1])
def test_selafin_plane_count_zero_or_one_is_2d(tmp_path, planes):
    """A plane count of 0 or 1 is a 2D mesh and reads normally.

    Some tools write 1 for a 2D mesh, which is also valid; neither value is a real
    3D file.
    """
    iparam = [1, 0, 0, 0, 0, 0, planes, 0, 0, 0]
    dataset = SelafinDriver().read(_write(tmp_path, iparam=iparam))

    assert dataset.mesh.n_node == 4


def test_selafin_layered_3d_file_is_refused(tmp_path):
    """A genuine 3D file (planes > 1) is refused, not read as a malformed 2D mesh.

    The fixture is a real layered file: three planes over a 4-node surface give 12 nodes
    and prism (6-node) connectivity. The earlier cut returned that 12-node, 6-vertex mesh
    while claiming "surface only"; this pins the refusal instead.
    """
    nx = [0.0, 1.0, 1.0, 0.0] * 3
    ny = [0.0, 0.0, 1.0, 1.0] * 3
    prisms = [[1, 2, 3, 5, 6, 7], [1, 3, 4, 5, 7, 8]]
    iparam = [1, 0, 0, 0, 0, 0, 3, 0, 0, 0]
    path = _write(
        tmp_path,
        node_x=nx,
        node_y=ny,
        ikle=prisms,
        variable_names=["Z               M               "],
        times=[0.0],
        values=[[list(range(12))]],
        iparam=iparam,
    )
    with pytest.raises(FileFormatNotSupportedError, match="layered 3D Selafin file"):
        SelafinDriver().read(path)


def test_selafin_parallel_subdomain_file_refused(tmp_path):
    """A file with interface points -- the parallel marker -- is refused.

    Interface points are nodes shared between partitions, so a nonzero count means the
    numbering is local to one sub-domain. Keyed on IPARAM[8] alone, per the Selafin
    header's slot semantics.
    """
    iparam = [1, 0, 0, 0, 0, 0, 0, 0, 3, 0]
    with pytest.raises(FileFormatNotSupportedError, match="parallel sub-domain"):
        SelafinDriver().read(_write(tmp_path, iparam=iparam))


def test_selafin_serial_file_with_boundary_points_reads(tmp_path):
    """A serial file whose boundary-point field is nonzero still reads.

    IPARAM[7] (boundary points) is a mesh property, nonzero for essentially every real
    mesh. Keying the parallel refusal on it -- as the first cut did -- refused ordinary
    serial files; this pins that it does not.
    """
    iparam = [1, 0, 0, 0, 0, 0, 0, 128, 0, 0]
    dataset = SelafinDriver().read(_write(tmp_path, iparam=iparam))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_selafin_lazy_read_seeks_without_loading_all(tmp_path):
    """A lazy read reports a source for each variable without loading its values."""
    dataset = SelafinDriver().read(_write(tmp_path), lazy=True)

    variable = dataset.get("WATER DEPTH")
    assert variable.has_data_source
    assert variable._data is None

    assert variable.data.tolist() == [
        [0.5, 0.25, 0.75, 0.125],
        [1.5, 1.25, 1.75, 1.125],
    ]


def test_selafin_eager_read_holds_values_directly(tmp_path):
    """A non-lazy read carries the values on the variable rather than a loader."""
    dataset = SelafinDriver().read(_write(tmp_path), lazy=False)

    variable = dataset.get("WATER DEPTH")
    assert variable._data is not None
    assert variable.data[0].tolist() == [0.5, 0.25, 0.75, 0.125]


def test_selafin_variable_name_and_unit_split_by_position(tmp_path):
    """The 32-char record splits into a 16-char name and 16-char unit, by position."""
    names = ["BOTTOM FRICTION M2/S            "]
    dataset = SelafinDriver().read(_write(tmp_path, variable_names=names))

    variable = dataset.get("BOTTOM FRICTION")
    assert variable.units == "M2/S"


def test_selafin_multiword_name_keeps_its_tail(tmp_path):
    """A name filling its 16 chars is not truncated at the first space."""
    names = ["FREE SURFACE    M               "]
    dataset = SelafinDriver().read(_write(tmp_path, variable_names=names))

    assert "FREE SURFACE" in dataset.data_variable_names


def test_selafin_time_values_are_read(tmp_path):
    """The per-timestep time values are carried on the variable."""
    dataset = SelafinDriver().read(_write(tmp_path))

    variable = dataset.get("WATER DEPTH")
    assert variable.attributes["time_values"].tolist() == [0.0, 3600.0]


def test_selafin_duplicate_variable_names_are_kept_apart(tmp_path):
    """Two identically-named variables stay distinct rather than collapsing to one."""
    names = [
        "VELOCITY U      M/S             ",
        "VELOCITY U      M/S             ",
    ]
    values = [
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        [[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]],
    ]
    dataset = SelafinDriver().read(
        _write(tmp_path, variable_names=names, values=values)
    )

    assert "VELOCITY U" in dataset.data_variable_names
    assert "VELOCITY U (2)" in dataset.data_variable_names


def test_selafin_partial_final_timestep_warns_and_is_dropped(tmp_path):
    """A file ending mid-timestep reads its whole steps and warns about the remainder."""
    path = _write(tmp_path)
    with open(path, "ab") as handle:
        # A dangling time record with no variable record after it: a partial step.
        word = np.array([4], dtype="<i4").tobytes()
        handle.write(word + np.array([7200.0], dtype="<f4").tobytes() + word)

    with pytest.warns(UserWarning, match="partial timestep"):
        dataset = SelafinDriver().read(path)

    assert dataset.get("WATER DEPTH").shape == (2, 4)


def test_selafin_corrupt_trailing_word_is_refused(tmp_path):
    """A trailing length word disagreeing with the leading one is refused."""
    path = _write(tmp_path)
    data = bytearray(Path(path).read_bytes())
    data[-1] = (data[-1] + 1) % 256
    Path(path).write_bytes(bytes(data))

    with pytest.raises(FileFormatNotSupportedError, match="not a readable Selafin"):
        SelafinDriver().read(path)


def test_selafin_can_open_false_on_junk(tmp_path):
    """``can_open`` declines empty, foreign and directory paths without raising."""
    driver = SelafinDriver()
    empty = tmp_path / "empty.slf"
    empty.write_bytes(b"")
    foreign = tmp_path / "other.slf"
    foreign.write_text("this is plainly text\n", encoding="utf-8")
    directory = tmp_path / "folder"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(foreign)) is False
    assert driver.can_open(str(directory)) is False


def test_selafin_committed_fixture_reads_and_matches_the_builder(tmp_path):
    """The committed fixture reads, and matches ``build_minimal_fixture_selafin``."""
    committed = Path(__file__).parent.parent.parent / "data" / "minimal.slf"
    dataset = SelafinDriver().read(str(committed))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2

    generated = tmp_path / "generated.slf"
    build_minimal_fixture_selafin(str(generated))
    assert generated.read_bytes() == committed.read_bytes()


def test_selafin_can_open_true_on_a_valid_file(tmp_path):
    """``can_open`` accepts a valid Selafin file -- the sniff's success path."""
    assert SelafinDriver().can_open(_write(tmp_path)) is True


def test_write_selafin_rejects_mismatched_values(tmp_path):
    """The fixture writer refuses a values list that does not match the timestep count."""
    with pytest.raises(ValueError, match="one entry per timestep"):
        write_selafin(
            str(tmp_path / "bad.slf"),
            node_x=_NODE_X,
            node_y=_NODE_Y,
            ikle=_IKLE,
            variable_names=_NAMES,
            times=[0.0, 1.0],
            values=[[[0.5, 0.25, 0.75, 0.125]]],
        )


def test_selafin_truncated_header_is_refused(tmp_path):
    """A file cut off inside its header is refused rather than read partially."""
    path = _write(tmp_path)
    data = Path(path).read_bytes()
    # Keep only the title record (its 4+80+4 bytes), dropping everything after.
    Path(path).write_bytes(data[:88])
    with pytest.raises(FileFormatNotSupportedError, match="not a readable Selafin"):
        SelafinDriver().read(path)


def test_selafin_zero_timestep_file_reads_empty_series(tmp_path):
    """A file with the header but no timesteps reads variables of zero length."""
    path = _write(tmp_path, times=[], values=[])
    dataset = SelafinDriver().read(path)

    variable = dataset.get("WATER DEPTH")
    assert variable.shape == (0, 4)
    assert variable.data.shape == (0, 4)


def test_selafin_header_ending_before_names_is_refused(tmp_path):
    """A file whose header ends before its promised name records is refused.

    Truncating right after the variable-count record leaves the first name record at end
    of file, which the raw-record reader reports rather than reading past.
    """
    path = _write(tmp_path)
    data = Path(path).read_bytes()
    # title record (4+80+4=88) + nbv record (4+8+4=16) = 104 bytes; names come next.
    Path(path).write_bytes(data[:104])
    with pytest.raises(FileFormatNotSupportedError, match="ends inside its header"):
        SelafinDriver().read(path)


def test_selafin_lazy_loader_translates_a_short_read(tmp_path):
    """A lazy slab read that comes up short raises the format error, like the eager path.

    Exercised at the loader directly: a bad offset/width is the failure a lazy read can
    hit when the caller touches .data, and it must surface as FileFormatNotSupportedError
    rather than the helper's bare ValueError.
    """
    from pyramids_mesh.drivers.coastal_ocean.telemac import _SelafinLoader

    path = tmp_path / "tiny.slf"
    path.write_bytes(b"\x00" * 8)
    loader = _SelafinLoader(str(path), [0], 100, endian="<", double=False)
    with pytest.raises(FileFormatNotSupportedError, match="not a readable Selafin"):
        loader()


def test_selafin_single_layer_extraction_reads_despite_the_plane_count(tmp_path):
    """A triangular mesh is read even when IPARAM[6] claims several planes.

    TELEMAC's single-layer extraction copies the parent run's plane count into the header
    while writing plain triangles, so the plane count cannot decide what the elements are.
    NDP can: 3 here, so the file is a 2D surface mesh and is read as one.
    """
    path = _write(tmp_path, iparam=[1, 0, 0, 0, 0, 0, 5, 0, 0, 0])
    dataset = SelafinDriver().read(path)

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    assert dataset.mesh.face_node_connectivity.data.shape == (2, 3)


def test_selafin_quad_mesh_reads_despite_the_plane_count(tmp_path):
    """The same holds for 4-node quadrilaterals, the other 2D element width."""
    path = _write(
        tmp_path,
        ikle=[[1, 2, 3, 4]],
        iparam=[1, 0, 0, 0, 0, 0, 7, 0, 0, 0],
    )
    dataset = SelafinDriver().read(path)

    assert dataset.mesh.face_node_connectivity.data.shape == (1, 4)


def test_selafin_prism_mesh_is_refused_whatever_the_plane_count(tmp_path):
    """NDP=6 is a layered mesh and is refused even when the plane count says 2D.

    The mirror of the extraction case: a header claiming one plane does not make prism
    connectivity readable as a flat 2D mesh, so the element width decides here too.
    """
    nx = [0.0, 1.0, 1.0, 0.0] * 2
    ny = [0.0, 0.0, 1.0, 1.0] * 2
    path = _write(
        tmp_path,
        node_x=nx,
        node_y=ny,
        ikle=[[1, 2, 3, 5, 6, 7]],
        variable_names=["Z               M               "],
        times=[0.0],
        values=[[list(range(8))]],
        iparam=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    )
    with pytest.raises(FileFormatNotSupportedError, match="NDP=6"):
        SelafinDriver().read(path)


def test_selafin_title_reaches_the_dataset(tmp_path):
    """The title record is the run's own name for itself and is kept, stripped of padding."""
    path = _write(tmp_path, title="Rouse profile test")
    dataset = SelafinDriver().read(path)

    assert dataset.global_attributes["title"] == "Rouse profile test"


def test_selafin_blank_title_adds_no_attribute(tmp_path):
    """A title record of pure padding is absent, not an empty string."""
    path = _write(tmp_path, title="   ")
    dataset = SelafinDriver().read(path)

    assert "title" not in dataset.global_attributes


def test_selafin_date_record_becomes_the_start_date(tmp_path):
    """The date record anchors the time axis, so it is carried as an ISO-8601 string."""
    path = _write(
        tmp_path,
        iparam=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        date=[1993, 7, 14, 6, 30, 15],
    )
    dataset = SelafinDriver().read(path)

    assert dataset.global_attributes["start_date"] == "1993-07-14T06:30:15"


def test_selafin_absent_date_record_leaves_no_start_date(tmp_path):
    """Without the IPARAM[9] flag there is no date record and no attribute to add."""
    dataset = SelafinDriver().read(_write(tmp_path))

    assert "start_date" not in dataset.global_attributes


def test_selafin_zero_date_record_is_a_placeholder_not_year_zero(tmp_path):
    """An all-zero date record is a placeholder; reporting year 0 would be worse than none."""
    path = _write(
        tmp_path,
        iparam=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        date=[0, 0, 0, 0, 0, 0],
    )
    dataset = SelafinDriver().read(path)

    assert "start_date" not in dataset.global_attributes


def test_selafin_boundary_numbering_is_carried_on_a_node_variable(tmp_path):
    """IPOBO is an ordering that cannot be rederived, so the read keeps it.

    It arrives as a static node variable rather than as an attribute, which is what lets the
    writer put it back into its own record.
    """
    path = _write(tmp_path, ipobo=[1, 2, 0, 3])
    dataset = SelafinDriver().read(path)

    numbering = np.asarray(dataset.get("boundary_node_number").data)
    assert numbering.tolist() == [1.0, 2.0, 0.0, 3.0]
    assert dataset.get("boundary_node_number").location == "node"
    assert dataset.global_attributes["n_boundary_nodes"] == 3


def test_selafin_all_zero_boundary_record_adds_no_variable(tmp_path):
    """A mesh with no boundary numbering gains nothing from an array of zeros."""
    dataset = SelafinDriver().read(_write(tmp_path))

    assert "boundary_node_number" not in dataset.data_variable_names
    assert "n_boundary_nodes" not in dataset.global_attributes


def test_selafin_can_open_declines_a_layered_3d_file(tmp_path):
    """can_open answers for readability: a file the read refuses is not claimed.

    Auto-detection trusts can_open, so claiming a prism file and then throwing from read
    would turn a clean "no driver" into a hard failure part-way through dispatch.
    """
    nx = [0.0, 1.0, 1.0, 0.0] * 2
    ny = [0.0, 0.0, 1.0, 1.0] * 2
    path = _write(
        tmp_path,
        node_x=nx,
        node_y=ny,
        ikle=[[1, 2, 3, 5, 6, 7]],
        variable_names=["Z               M               "],
        times=[0.0],
        values=[[list(range(8))]],
    )
    assert SelafinDriver().can_open(path) is False


def test_selafin_can_open_declines_a_parallel_subdomain_file(tmp_path):
    """A non-zero interface-point count means local numbering, so the file is not claimed."""
    path = _write(tmp_path, iparam=[1, 0, 0, 0, 0, 0, 0, 0, 7, 0])

    assert SelafinDriver().can_open(path) is False


def test_selafin_can_open_still_accepts_a_serial_2d_file(tmp_path):
    """The added header checks must not start declining ordinary files."""
    assert SelafinDriver().can_open(_write(tmp_path)) is True
    assert (
        SelafinDriver().can_open(
            _write(
                tmp_path,
                name="dated.slf",
                iparam=[1, 0, 0, 0, 0, 0, 1, 264, 0, 1],
                date=[2000, 1, 1, 0, 0, 0],
            )
        )
        is True
    )


def test_selafin_can_open_declines_a_file_truncated_inside_its_header(tmp_path):
    """A first record that parses does not make the rest of the header readable."""
    source = Path(_write(tmp_path))
    truncated = tmp_path / "truncated.slf"
    truncated.write_bytes(source.read_bytes()[:104])

    assert SelafinDriver().can_open(str(truncated)) is False


def test_selafin_peek_header_refuses_an_empty_file(tmp_path):
    """An empty file has no title record, which the peek reports rather than assuming.

    ``_read_record`` answers ``None`` at a clean end of file, so this is the one shape that
    reaches the peek's own guard; a file cut mid-record raises from the helper instead.
    """
    from pyramids_mesh.drivers.coastal_ocean.telemac import _peek_header

    path = tmp_path / "empty.slf"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="ends before its title record"):
        _peek_header(str(path), "<")


def test_selafin_can_open_declines_a_bare_length_word(tmp_path):
    """A length word announcing a title that is not there is declined, not claimed.

    ``detect_endianness`` only reads that first word, so the probe passes; the sniff has to
    go further before answering, and a failed peek means the file is not readable.
    """
    path = tmp_path / "stub.slf"
    path.write_bytes(np.array([80], dtype="<i4").tobytes())

    assert SelafinDriver().can_open(str(path)) is False


def test_selafin_peek_header_refuses_a_file_cut_between_its_name_records(tmp_path):
    """A header that stops part-way through the variable names is declined, not guessed."""
    from pyramids_mesh.drivers.coastal_ocean.telemac import _peek_header

    source = Path(
        _write(
            tmp_path,
            variable_names=[
                "WATER DEPTH     M               ",
                "VELOCITY U      M/S             ",
            ],
            values=[[[0.0] * 4, [0.0] * 4], [[1.0] * 4, [1.0] * 4]],
        )
    )
    # title record (4+80+4) + nbv record (4+8+4) + one of the two 32-byte name records
    cut = tmp_path / "cut.slf"
    cut.write_bytes(source.read_bytes()[: 88 + 16 + 40])

    with pytest.raises(ValueError, match="ends inside its variable-name records"):
        _peek_header(str(cut), "<")
    assert SelafinDriver().can_open(str(cut)) is False


def test_selafin_an_undefined_element_width_is_not_diagnosed_as_3d(tmp_path):
    """NDP 5 is no Selafin element at all, so it is not reported as a layered 3D file.

    Keying the message on "not 2D" sent the reader looking for a 3D file that is not there;
    only NDP 6 prisms are 3D, and anything else is a corrupt or unknown variant.
    """
    nx = [float(i) for i in range(5)]
    ny = [0.0, 1.0, 2.0, 3.0, 4.0]
    path = _write(
        tmp_path,
        node_x=nx,
        node_y=ny,
        ikle=[[1, 2, 3, 4, 5]],
        variable_names=["Z               M               "],
        times=[0.0],
        values=[[[0.0] * 5]],
    )
    with pytest.raises(FileFormatNotSupportedError, match="no elements for"):
        SelafinDriver().read(path)


def test_selafin_prisms_are_still_diagnosed_as_layered_3d(tmp_path):
    """NDP 6 keeps the 3D diagnosis, naming the prisms and the plane count."""
    nx = [0.0, 1.0, 1.0, 0.0] * 2
    ny = [0.0, 0.0, 1.0, 1.0] * 2
    path = _write(
        tmp_path,
        node_x=nx,
        node_y=ny,
        ikle=[[1, 2, 3, 5, 6, 7]],
        variable_names=["Z               M               "],
        times=[0.0],
        values=[[list(range(8))]],
        iparam=[1, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    )
    with pytest.raises(FileFormatNotSupportedError, match="NDP=6 prisms over 3 plane"):
        SelafinDriver().read(path)


def test_selafin_boundary_numbering_is_integral(tmp_path):
    """IPOBO is an ordering, so it is surfaced as integers rather than floats."""
    dataset = SelafinDriver().read(_write(tmp_path, ipobo=[1, 2, 0, 3]))
    numbering = np.asarray(dataset.get("boundary_node_number").data)

    assert numbering.dtype.kind == "i", (
        f"expected an integer dtype, got {numbering.dtype}"
    )
    assert numbering.tolist() == [1, 2, 0, 3]


@pytest.mark.parametrize(
    "date_record",
    [
        [1993, 0, 0, 12, 0, 0],
        [1993, 7, 14, 99, 99, 99],
        [1993, 13, 1, 0, 0, 0],
        [-5, 1, 1, 0, 0, 0],
        [1993, 2, 30, 0, 0, 0],
    ],
)
def test_selafin_a_date_record_that_is_not_an_instant_warns_and_is_dropped(
    tmp_path, date_record
):
    """A record that is not a calendar instant is dropped, and the drop is announced.

    Formatting is not parsing: ``[1993, 0, 0, 12, 0, 0]`` formats cleanly into
    '1993-00-00T12:00:00', which looks like a timestamp and which no date parser accepts.
    Unlike the all-zero placeholder this is a surprise, so it warns rather than going quiet.
    """
    path = _write(
        tmp_path,
        iparam=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        date=date_record,
    )
    with pytest.warns(UserWarning, match="not a calendar instant"):
        dataset = SelafinDriver().read(path)

    assert "start_date" not in dataset.global_attributes
