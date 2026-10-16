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

from pyramids_mesh.drivers import telemac
from pyramids_mesh.drivers.telemac import (
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
    """The file reads whether or not the optional date record is present."""
    iparam = [1, 0, 0, 0, 0, 0, 0, 0, 0, date_flag]
    dataset = SelafinDriver().read(_write(tmp_path, iparam=iparam))

    assert dataset.mesh.n_node == 4
    assert dataset.get("WATER DEPTH").shape == (2, 4)


@pytest.mark.parametrize("planes", [0, 1])
def test_selafin_plane_count_zero_or_one_is_2d(tmp_path, planes):
    """A plane count of 0 or 1 is a 2D mesh and reads normally.

    Some tools write 1 for a 2D mesh, which MDAL also accepts; neither value is a real
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
    numbering is local to one sub-domain. Keyed on IPARAM[8] alone, matching MDAL's slot
    semantics (mdal_selafin.cpp).
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
    committed = Path(__file__).parent.parent / "data" / "minimal.slf"
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
    from pyramids_mesh.drivers.telemac import _SelafinLoader

    path = tmp_path / "tiny.slf"
    path.write_bytes(b"\x00" * 8)
    loader = _SelafinLoader(str(path), [0], 100, endian="<", double=False)
    with pytest.raises(FileFormatNotSupportedError, match="not a readable Selafin"):
        loader()
