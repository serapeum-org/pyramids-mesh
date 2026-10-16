"""Tests for the SMS ASCII DAT reader (D2).

DAT is the ``needs_mesh`` exemplar: it carries values and no topology, so the tests go
through ``UgridDataset.attach_data`` rather than ``read``. The emphasis is on the
element-count check, which is the only thing tying a results file to its mesh, and on
the two dialects, which differ in ``TS`` arity and so fail silently when conflated.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers import sms
from pyramids_mesh.drivers.sms import DatDriver, build_minimal_fixture

# The 2DM minimal fixture is four nodes and two faces, so a node block is four rows and
# a cell block two. Keeping them unequal is what lets the active-block layout check fire.
_N_NODE = 4
_N_FACE = 2

_NEW_DIALECT = """DATASET
OBJTYPE mesh2d
BEGSCL
ND 4
NC 2
NAME depth
TIMEUNITS SECONDS
TS 0 0.0
0.5
0.6
0.7
0.8
ENDDS
"""


@pytest.fixture
def mesh(tmp_path):
    """A 2DM mesh of four nodes and two faces, for a DAT to attach to."""
    path = tmp_path / "domain.2dm"
    build_minimal_fixture(str(path))
    return UgridDataset.read_file(str(path))


def _write_dat(tmp_path, text, name="results.dat"):
    """Write a DAT fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _attach(mesh, tmp_path, text, name="results.dat"):
    """Attach a DAT written from ``text`` to ``mesh``."""
    return mesh.attach_data(_write_dat(tmp_path, text, name))


def test_dat_read_raises_pointing_at_attach_data():
    """``read`` refuses and names ``attach_data``, rather than failing obscurely."""
    with pytest.raises(ValueError, match="no read\\(\\)") as caught:
        DatDriver().read("depth.dat")

    assert "attach_data" in str(caught.value)


def test_dat_new_format_parses(mesh, tmp_path):
    """The ``DATASET`` dialect yields one node variable carrying the file's values."""
    attached = _attach(mesh, tmp_path, _NEW_DIALECT)

    depth = attached.get("depth")
    assert depth.location == "node"
    assert depth.shape == (1, _N_NODE)
    assert depth.data.tolist() == [[0.5, 0.6, 0.7, 0.8]]


def test_dat_old_format_without_dataset_card_parses(mesh, tmp_path):
    """A file with no ``DATASET`` card is recognised by its other cards and read."""
    text = """BEGSCL
ND 4
NAME stage
TS 0.0
1.0
2.0
3.0
4.0
ENDDS
"""
    attached = _attach(mesh, tmp_path, text)

    assert attached.get("stage").data.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_dat_old_format_ts_arity(mesh, tmp_path):
    """The old dialect's ``TS <time>`` is read as a time, not as an active flag.

    Conflating the arities takes the time for ``istat`` and the first value for the
    time, which yields a plausible one-timestep series instead of an error -- so the
    times themselves are what this asserts.
    """
    text = """BEGSCL
ND 4
NAME stage
TS 10.0
1.0
2.0
3.0
4.0
TS 20.0
5.0
6.0
7.0
8.0
ENDDS
"""
    attached = _attach(mesh, tmp_path, text)

    stage = attached.get("stage")
    assert stage.shape == (2, _N_NODE)
    assert stage.attributes["time_values"].tolist() == [10.0, 20.0]
    assert stage.data.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


@pytest.mark.parametrize(
    "card, declared, noun, actual",
    [("ND", 9, "nodes", _N_NODE), ("NC", 7, "faces", _N_FACE)],
)
def test_dat_element_count_mismatch_raises_reporting_both(
    mesh, tmp_path, card, declared, noun, actual
):
    """A declared count that disagrees with the mesh is refused, reporting both."""
    text = f"""DATASET
BEGSCL
{card} {declared}
NAME depth
TS 0 0.0
ENDDS
"""
    with pytest.raises(ValueError) as caught:
        _attach(mesh, tmp_path, text)

    message = str(caught.value)
    assert f"{card} {declared}" in message
    assert f"{actual} {noun}" in message


def test_dat_missing_nd_and_nc_refuses(mesh, tmp_path):
    """A file declaring neither count is refused rather than attached on hope."""
    text = """DATASET
BEGSCL
NAME depth
TS 0 0.0
0.5
0.6
0.7
0.8
ENDDS
"""
    with pytest.raises(ValueError, match="declares neither"):
        _attach(mesh, tmp_path, text)


def _with_active(flag_rows):
    """A one-timestep DAT whose ``istat`` is set, with the given active rows."""
    rows = "\n".join(flag_rows)
    return f"""DATASET
BEGSCL
ND 4
NC 2
NAME depth
TS 1 0.0
{rows}
0.5
0.6
0.7
0.8
ENDDS
"""


def test_dat_istat_becomes_sibling_active_variable(mesh, tmp_path):
    """``istat`` yields a face-located ``<name>_active`` sibling with the time axis."""
    attached = _attach(mesh, tmp_path, _with_active(["1", "0"]))

    active = attached.get("depth_active")
    assert active.location == "face"
    assert active.shape == (1, _N_FACE)
    assert active.data.tolist() == [[1, 0]]


def test_dat_active_not_folded_into_values(mesh, tmp_path):
    """An inactive cell leaves the values untouched -- no NaN is written over them."""
    attached = _attach(mesh, tmp_path, _with_active(["1", "0"]))

    values = attached.get("depth").data
    assert not bool(np.isnan(values).any())
    assert values.tolist() == [[0.5, 0.6, 0.7, 0.8]]


def test_dat_active_block_with_non_binary_flags_is_refused(mesh, tmp_path):
    """A block of real values where the flags belong is refused, not read as flags.

    This is the only check available on the active block's position, and it is why the
    node and cell counts of the fixture are deliberately unequal.
    """
    with pytest.raises(FileFormatNotSupportedError, match="other than 0 and 1"):
        _attach(mesh, tmp_path, _with_active(["0.5", "0.6"]))


def test_dat_rt_julian_kept_in_variable_attributes(mesh, tmp_path):
    """``RT_JULIAN`` survives on the variable, the only place a driver can put it.

    ``attach_data`` merges the variables a driver returns and keeps the dataset's own
    ``global_attributes``, so there is no route to a file-level attribute.
    """
    text = """DATASET
BEGSCL
ND 4
NAME depth
RT_JULIAN 2451545.0
TIMEUNITS HOURS
TS 0 0.0
0.5
0.6
0.7
0.8
ENDDS
"""
    attached = _attach(mesh, tmp_path, text)

    attributes = attached.get("depth").attributes
    assert attributes["reference_time"] == 2451545.0
    assert attributes["time_units"] == "HOURS"
    assert "reference_time" not in attached.global_attributes


def test_dat_new_group_while_open_raises(mesh, tmp_path):
    """A second ``BEGSCL`` before ``ENDDS`` is refused rather than guessed at."""
    text = """DATASET
BEGSCL
ND 4
NAME depth
BEGSCL
ENDDS
"""
    with pytest.raises(
        FileFormatNotSupportedError, match="while another is still open"
    ):
        _attach(mesh, tmp_path, text)


def test_dat_endds_without_group_raises(mesh, tmp_path):
    """An ``ENDDS`` with nothing open is refused."""
    text = """DATASET
ND 4
ENDDS
"""
    with pytest.raises(FileFormatNotSupportedError, match="no open group"):
        _attach(mesh, tmp_path, text)


@pytest.mark.parametrize("card", ["NAME depth", "TIMEUNITS SECONDS", "TS 0 0.0"])
def test_dat_group_only_card_outside_a_group_raises(mesh, tmp_path, card):
    """``NAME``, ``TIMEUNITS`` and ``TS`` outside a group are refused, as MDAL does."""
    text = f"""DATASET
ND 4
{card}
"""
    with pytest.raises(FileFormatNotSupportedError, match="outside a"):
        _attach(mesh, tmp_path, text)


def test_dat_vector_group_yields_documented_shape(mesh, tmp_path):
    """A ``BEGVEC`` group becomes two scalar variables, ``_x`` and ``_y``."""
    text = """DATASET
BEGVEC
ND 4
NAME velocity
TS 0 0.0
1.0 -1.0
2.0 -2.0
3.0 -3.0
4.0 -4.0
ENDDS
"""
    attached = _attach(mesh, tmp_path, text)

    assert "velocity" not in attached.data_variable_names
    assert attached.get("velocity_x").data.tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert attached.get("velocity_y").data.tolist() == [[-1.0, -2.0, -3.0, -4.0]]
    assert attached.get("velocity_x").shape == (1, _N_NODE)


def test_dat_face_located_when_only_nc_is_declared(mesh, tmp_path):
    """A file declaring only ``NC`` is read as face-located, with a cell per row."""
    text = """DATASET
BEGSCL
NC 2
NAME bed_shear
TS 0 0.0
7.0
8.0
ENDDS
"""
    attached = _attach(mesh, tmp_path, text)

    variable = attached.get("bed_shear")
    assert variable.location == "face"
    assert variable.shape == (1, _N_FACE)


def test_dat_unterminated_group_is_refused(mesh, tmp_path):
    """A group left open at end of file is refused rather than silently closed."""
    text = """DATASET
BEGSCL
ND 4
NAME depth
TS 0 0.0
0.5
0.6
0.7
0.8
"""
    with pytest.raises(FileFormatNotSupportedError, match="still open"):
        _attach(mesh, tmp_path, text)


def test_dat_file_with_no_group_is_refused(mesh, tmp_path):
    """A file holding no group at all carries nothing to attach."""
    with pytest.raises(FileFormatNotSupportedError, match="holds no"):
        _attach(mesh, tmp_path, "DATASET\nND 4\n")


def test_dat_group_without_timestep_is_refused(mesh, tmp_path):
    """A group with no ``TS`` card carries no values, and is refused as such."""
    text = """DATASET
BEGSCL
ND 4
NAME depth
ENDDS
"""
    with pytest.raises(FileFormatNotSupportedError, match="no TS card"):
        _attach(mesh, tmp_path, text)


def test_dat_short_value_block_hits_the_next_card_and_is_refused(mesh, tmp_path):
    """A block short of its declared rows reads the next card and refuses it.

    The row count comes from ``ND``, so a block two rows short consumes ``ENDDS`` where
    a number belongs. Naming the offending token is more use than a row count: it points
    at the card the file actually reached.
    """
    text = """DATASET
BEGSCL
ND 4
NAME depth
TS 0 0.0
0.5
0.6
ENDDS
"""
    with pytest.raises(FileFormatNotSupportedError, match="found 'ENDDS'"):
        _attach(mesh, tmp_path, text)


def test_dat_value_block_truncated_by_end_of_file_is_refused(mesh, tmp_path):
    """A block running past the end of the file is refused, reporting what was found."""
    text = """DATASET
BEGSCL
ND 4
NAME depth
TS 0 0.0
0.5
0.6
"""
    with pytest.raises(FileFormatNotSupportedError, match="ends inside a TS block"):
        _attach(mesh, tmp_path, text)


def test_dat_short_vector_row_is_refused(mesh, tmp_path):
    """A vector row carrying one number instead of two is refused."""
    text = """DATASET
BEGVEC
ND 4
NAME velocity
TS 0 0.0
1.0 -1.0
2.0
3.0 -3.0
4.0 -4.0
ENDDS
"""
    with pytest.raises(FileFormatNotSupportedError, match="short TS row"):
        _attach(mesh, tmp_path, text)


def test_dat_duplicate_group_names_are_refused(mesh, tmp_path):
    """Two groups of the same name are refused, since one would replace the other."""
    group = """BEGSCL
ND 4
NAME depth
TS 0 0.0
0.5
0.6
0.7
0.8
ENDDS
"""
    with pytest.raises(FileFormatNotSupportedError, match="twice"):
        _attach(mesh, tmp_path, "DATASET\n" + group + group)


def test_dat_unnamed_group_gets_a_positional_name(mesh, tmp_path):
    """A group with no ``NAME`` is named by position rather than refused."""
    text = """DATASET
BEGSCL
ND 4
TS 0 0.0
0.5
0.6
0.7
0.8
ENDDS
"""
    attached = _attach(mesh, tmp_path, text)

    assert "dataset_1" in attached.data_variable_names


def test_dat_unknown_card_is_ignored(mesh, tmp_path):
    """A card this driver does not model is skipped, as MDAL skips its unknowns."""
    text = _NEW_DIALECT.replace("NAME depth\n", "NAME depth\nSOMETHINGELSE 1 2 3\n")
    attached = _attach(mesh, tmp_path, text)

    assert attached.get("depth").data.tolist() == [[0.5, 0.6, 0.7, 0.8]]


def test_dat_quoted_name_with_spaces_is_kept(mesh, tmp_path):
    """A quoted multi-word group name keeps its spaces and loses its quotes."""
    text = _NEW_DIALECT.replace("NAME depth", 'NAME "water depth"')
    attached = _attach(mesh, tmp_path, text)

    assert "water depth" in attached.data_variable_names


def test_dat_attaching_keeps_the_mesh_own_variables(mesh, tmp_path):
    """``attach_data`` adds to the dataset's variables rather than replacing them."""
    attached = _attach(mesh, tmp_path, _NEW_DIALECT)

    assert sorted(attached.data_variable_names) == ["depth", "material", "node_z"]


def test_dat_can_open_declines_junk_without_raising(tmp_path):
    """``can_open`` answers False for an empty, foreign or directory path."""
    driver = DatDriver()
    empty = tmp_path / "empty.dat"
    empty.write_bytes(b"")
    foreign = tmp_path / "prose.dat"
    foreign.write_text("this is not a results file\n", encoding="utf-8")
    directory = tmp_path / "folder"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(foreign)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "absent.dat")) is False


@pytest.mark.parametrize(
    "text, expected",
    [
        ("DATASET\nBEGSCL\n", True),
        ("# a comment\nDATASET\n", True),
        ("BEGSCL\nND 4\nTS 0 0.0\n", True),
        ("TIMEUNITS SECONDS\n", True),
        ("MESH2D\nND 1 0.0 0.0\n", False),
        ("nothing to see here\n", False),
    ],
)
def test_dat_sniff_recognises_both_dialects(tmp_path, text, expected):
    """Both dialects are recognised, and a 2DM under a ``.dat`` suffix is not."""
    path = tmp_path / "candidate.dat"
    path.write_text(text, encoding="utf-8")

    assert DatDriver().can_open(str(path)) is expected


def test_dat_suffix_is_required(tmp_path):
    """Right content under a suffix the driver does not claim is still declined."""
    path = tmp_path / "results.txt"
    path.write_text(_NEW_DIALECT, encoding="utf-8")

    assert DatDriver().can_open(str(path)) is False


def test_dat_minimal_fixture_attaches_to_the_2dm_minimal_fixture(mesh, tmp_path):
    """The two minimal fixtures are built to match, so they attach without error."""
    path = tmp_path / "minimal.dat"
    sms.build_minimal_fixture_dat(str(path))
    attached = mesh.attach_data(str(path))

    depth = attached.get("depth")
    assert depth.shape == (2, _N_NODE)
    assert depth.attributes["time_values"].tolist() == [0.0, 3600.0]


def test_dat_committed_fixture_attaches_and_matches_the_builder(mesh, tmp_path):
    """The committed fixture attaches, and matches ``build_minimal_fixture_dat``.

    The harness builds its fixture from the function while the repository ships the
    file, so without this they could drift apart and each keep passing on its own.
    """
    committed = Path(__file__).parent.parent / "data" / "minimal.dat"
    attached = mesh.attach_data(str(committed))

    assert attached.get("depth").shape == (2, _N_NODE)

    generated = tmp_path / "generated.dat"
    sms.build_minimal_fixture_dat(str(generated))
    assert generated.read_text(encoding="utf-8") == committed.read_text(
        encoding="utf-8"
    )
