"""Tests for the Nastran bulk-data reader.

The oracle is a real committed deck, ``tests/data/nastran/cylinder.fem`` (meshio test
data, MIT): 33 ``GRID`` nodes, 4 ``CTRIA3`` and 18 ``CQUAD4`` surface faces, plus
volumetric ``CTETRA``/``CPYRA`` and ``CBAR`` cards that this surface reader drops. Tiny
inline decks in ``tmp_path`` exercise the defensive branches -- small, large and free
field layouts, the id map, and the error paths -- the single committed sample cannot
cover alone.

A key trap: the ``.fem`` suffix is shared with DHI FEFLOW, so ``can_open`` is proved to
decline the committed FEFLOW ``.fem`` as well as to accept the Nastran deck.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.nastran import NastranDriver, _parse_real

_DATA = Path(__file__).parent.parent / "data"
_CYLINDER = _DATA / "nastran" / "cylinder.fem"
# A real DHI FEFLOW mesh under the shared '.fem' suffix, to prove the sniff declines it.
_FEFLOW = _DATA / "feflow" / "disv_nodes.fem"


def _write(tmp_path, text, name="mesh.bdf"):
    """Write an inline deck and return its path as a string."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _read(path):
    """Read a deck, silencing the winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NastranDriver().read(str(path))


def _signed_area(face, node_x, node_y):
    """Twice the signed polygon area of a face's real (non-fill) vertices; CCW is > 0."""
    vertices = face[face != -1]
    xs = node_x[vertices]
    ys = node_y[vertices]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _small(name, *fields):
    """Build a small-field (8-column) fixed-field card line."""
    line = f"{name:<8}"
    for field in fields:
        line += f"{field:>8}"
    return line


def _large_pair(name, *fields):
    """Build a large-field (16-column) card over its name and continuation lines.

    The first four fields sit on the ``name*`` line, the rest on a ``*`` continuation.
    """
    first = f"{name + '*':<8}" + "".join(f"{field:>16}" for field in fields[:4])
    rest = fields[4:]
    if not rest:
        return first
    second = f"{'*':<8}" + "".join(f"{field:>16}" for field in rest)
    return first + "\n" + second


# A minimal small-field surface deck: the unit square split into two triangles.
_MINIMAL = "\n".join(
    [
        "BEGIN BULK",
        _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
        _small("GRID", "2", "0", "1.0", "0.0", "0.0"),
        _small("GRID", "3", "0", "1.0", "1.0", "0.0"),
        _small("GRID", "4", "0", "0.0", "1.0", "0.0"),
        _small("CTRIA3", "1", "0", "1", "2", "3"),
        _small("CTRIA3", "2", "0", "1", "3", "4"),
        "ENDDATA",
        "",
    ]
)


# -- the committed sample -----------------------------------------------------------


def test_committed_sample_reads_surface_counts():
    """The committed deck builds 33 nodes and 22 surface faces, dropping the solids."""
    mesh = _read(_CYLINDER).mesh
    assert mesh.n_node == 33
    # 4 CTRIA3 + 18 CQUAD4 = 22; CTETRA/CPYRA/CBAR are skipped.
    assert mesh.n_face == 22


def test_committed_sample_pads_triangles_to_four_columns():
    """Mixed CTRIA3/CQUAD4 widths pad to the quad's four columns with -1 fill."""
    mesh = _read(_CYLINDER).mesh
    data = mesh.face_node_connectivity.data
    assert data.shape == (22, 4)
    # At least one row (a CTRIA3) carries the -1 pad; CQUAD4 rows do not.
    assert (data == -1).any()


def test_committed_sample_embedded_exponent_coordinates():
    """The fixed-field '-2.45-16' reals parse as -2.45e-16, not a format error."""
    mesh = _read(_CYLINDER).mesh
    # GRID 1 and 2 carry y = -2.45-16; the sign-only exponent must have been read.
    assert float(mesh.node_y[0]) == pytest.approx(-2.45e-16)
    assert float(mesh.node_x[0]) == pytest.approx(1.0)


def test_committed_sample_connectivity_is_zero_based_with_base_recorded():
    """The deck's ids resolve to 0-based indices, the 1-based base kept for a writer."""
    mesh = _read(_CYLINDER).mesh
    connectivity = mesh.face_node_connectivity
    assert int(connectivity.data[connectivity.data != -1].min()) == 0
    assert connectivity.original_start_index == 1


def test_committed_sample_faces_are_not_clockwise():
    """No face is left wound clockwise -- the invariant the normaliser enforces.

    The deck is a cylinder surface, so its vertical wall faces project to zero area in
    the ``xy`` plane and stay degenerate; the normaliser only flips a genuinely clockwise
    face, so the testable guarantee is that none has negative (clockwise) signed area.
    """
    mesh = _read(_CYLINDER).mesh
    node_x = np.asarray(mesh.node_x)
    node_y = np.asarray(mesh.node_y)
    for face in mesh.face_node_connectivity.data:
        assert _signed_area(face, node_x, node_y) >= 0.0


def test_committed_sample_node_z_from_x3_column():
    """The GRID X3 column, non-zero in this deck, becomes a node_z variable."""
    dataset = _read(_CYLINDER)
    assert "node_z" in set(dataset.data_variable_names)
    assert dataset.get("node_z").location == "node"


def test_committed_sample_has_no_crs():
    """Nastran carries no CRS, and none is guessed from the coordinate magnitudes."""
    dataset = _read(_CYLINDER)
    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


# -- field layouts ------------------------------------------------------------------


def test_small_field_minimal_deck_reads(tmp_path):
    """A small-field deck builds the unit square from two triangles."""
    dataset = _read(_write(tmp_path, _MINIMAL))
    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    np.testing.assert_allclose(dataset.mesh.node_x, [0.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(dataset.mesh.node_y, [0.0, 0.0, 1.0, 1.0])


def test_large_field_deck_reads(tmp_path):
    """A large-field (16-column) deck with a GRID* continuation line reads.

    The continuation line carries the X3 column, so node_z is picked up too.
    """
    text = "\n".join(
        [
            _large_pair("GRID", "1", "0", "0.0", "0.0", "2.5"),
            _large_pair("GRID", "2", "0", "1.0", "0.0", "2.5"),
            _large_pair("GRID", "3", "0", "1.0", "1.0", "2.5"),
            _small("CTRIA3", "1", "0", "1", "2", "3"),
            "",
        ]
    )
    dataset = _read(_write(tmp_path, text, name="large.bdf"))
    assert dataset.mesh.n_node == 3
    assert dataset.mesh.n_face == 1
    np.testing.assert_allclose(dataset.mesh.node_x, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(dataset.get("node_z").data, [2.5, 2.5, 2.5])


def test_free_field_deck_reads(tmp_path):
    """A free-field (comma-separated) deck reads, with empty fields defaulting."""
    text = "GRID,1,,0.,0.\n" "GRID,2,,1.,0.\n" "GRID,3,,0.,1.\n" "CTRIA3,10,0,1,2,3\n"
    dataset = _read(_write(tmp_path, text, name="free.nas"))
    assert dataset.mesh.n_node == 3
    assert dataset.mesh.n_face == 1
    np.testing.assert_array_equal(dataset.mesh.face_node_connectivity.data, [[0, 1, 2]])


def test_quad_card_reads_four_corner_nodes(tmp_path):
    """A CQUAD4 yields a four-vertex face in card order."""
    text = "\n".join(
        [
            _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "2", "0", "1.0", "0.0", "0.0"),
            _small("GRID", "3", "0", "1.0", "1.0", "0.0"),
            _small("GRID", "4", "0", "0.0", "1.0", "0.0"),
            _small("CQUAD4", "1", "0", "1", "2", "3", "4"),
            "",
        ]
    )
    dataset = _read(_write(tmp_path, text))
    assert dataset.mesh.face_node_connectivity.data.shape == (1, 4)


# -- real-field parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    "field, expected",
    [
        ("-2.45-16", -2.45e-16),
        ("2.06-15", 2.06e-15),
        ("8.698-15", 8.698e-15),
        (".7071068", 0.7071068),
        ("1.0", 1.0),
        ("1.5D+3", 1500.0),
        ("1.+3", 1000.0),
        ("   ", 0.0),
    ],
)
def test_parse_real_spellings(field, expected):
    """Every Nastran real spelling, including a sign-only exponent, parses."""
    assert _parse_real(field, "deck.bdf") == pytest.approx(expected)


def test_parse_real_rejects_garbage():
    """A field that is not a real names the file as a format error."""
    with pytest.raises(FileFormatNotSupportedError, match="not a Nastran real"):
        _parse_real("clay", "deck.bdf")


# -- id mapping and element handling ------------------------------------------------


def test_gapped_and_out_of_order_ids_resolve(tmp_path):
    """Arbitrary GRID labels wire through the id map, not by id == index + 1."""
    text = "\n".join(
        [
            _small("GRID", "7", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "3", "0", "1.0", "0.0", "0.0"),
            _small("GRID", "90", "0", "1.0", "1.0", "0.0"),
            _small("CTRIA3", "1", "0", "7", "3", "90"),
            "",
        ]
    )
    dataset = _read(_write(tmp_path, text))
    assert dataset.mesh.n_node == 3
    np.testing.assert_array_equal(dataset.mesh.face_node_connectivity.data, [[0, 1, 2]])
    np.testing.assert_allclose(dataset.mesh.node_x, [0.0, 1.0, 1.0])


def test_volumetric_cards_are_skipped(tmp_path):
    """CTETRA solids ride alongside the surface faces but are not read as faces."""
    text = "\n".join(
        [
            _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "2", "0", "1.0", "0.0", "0.0"),
            _small("GRID", "3", "0", "1.0", "1.0", "0.0"),
            _small("GRID", "4", "0", "0.0", "1.0", "1.0"),
            _small("CTRIA3", "1", "0", "1", "2", "3"),
            _small("CTETRA", "2", "0", "1", "2", "3", "4"),
            "",
        ]
    )
    dataset = _read(_write(tmp_path, text))
    # Only the CTRIA3 is a face; the CTETRA solid is dropped.
    assert dataset.mesh.n_face == 1


def test_undefined_node_reference_raises(tmp_path):
    """A face naming a node id no GRID defines is an error, not a guess."""
    text = "\n".join(
        [
            _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "2", "0", "1.0", "0.0", "0.0"),
            _small("GRID", "3", "0", "1.0", "1.0", "0.0"),
            _small("CTRIA3", "1", "0", "1", "2", "99"),
            "",
        ]
    )
    with pytest.raises(ValueError, match="node id 99"):
        _read(_write(tmp_path, text))


def test_duplicate_grid_id_is_refused(tmp_path):
    """Two GRID cards sharing an id make the connectivity ambiguous, so it raises."""
    text = "\n".join(
        [
            _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "1", "0", "1.0", "0.0", "0.0"),
            _small("GRID", "3", "0", "1.0", "1.0", "0.0"),
            _small("CTRIA3", "1", "0", "1", "1", "3"),
            "",
        ]
    )
    with pytest.raises(ValueError, match="duplicate native id"):
        _read(_write(tmp_path, text))


def test_deck_without_surface_elements_is_refused(tmp_path):
    """A deck of only solids has no surface, so it is refused naming the missing card."""
    text = "\n".join(
        [
            _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "2", "0", "1.0", "0.0", "0.0"),
            _small("GRID", "3", "0", "1.0", "1.0", "0.0"),
            _small("GRID", "4", "0", "0.0", "1.0", "1.0"),
            _small("CTETRA", "1", "0", "1", "2", "3", "4"),
            "",
        ]
    )
    with pytest.raises(FileFormatNotSupportedError, match="no CTRIA3 or CQUAD4"):
        _read(_write(tmp_path, text))


def test_deck_without_grid_is_refused(tmp_path):
    """A deck with no GRID card defines no node, so it is refused."""
    text = _small("CTRIA3", "1", "0", "1", "2", "3") + "\n"
    with pytest.raises(FileFormatNotSupportedError, match="no GRID card"):
        _read(_write(tmp_path, text, name="nogrid.bdf"))


def test_flat_deck_carries_no_node_z(tmp_path):
    """An all-zero X3 column gains no meaningless node_z, matching the exodus reader."""
    dataset = _read(_write(tmp_path, _MINIMAL))
    assert "node_z" not in set(dataset.data_variable_names)


# -- can_open and the FEFLOW collision ----------------------------------------------


def test_can_open_accepts_the_nastran_deck():
    """``can_open`` recognises the committed Nastran deck by its GRID and face cards."""
    assert NastranDriver().can_open(str(_CYLINDER)) is True


def test_can_open_declines_the_feflow_fem():
    """A DHI FEFLOW ``.fem`` shares the suffix but is declined by its PROBLEM/CLASS header.

    This is the whole reason the sniff requires the Nastran signature: both drivers claim
    ``.fem``, and misreading a FEFLOW mesh as Nastran must not happen.
    """
    assert NastranDriver().can_open(str(_FEFLOW)) is False


def test_can_open_declines_empty_binary_directory_and_foreign(tmp_path):
    """``can_open`` declines anything that is not a Nastran deck, and never raises."""
    driver = NastranDriver()
    empty = tmp_path / "empty.bdf"
    empty.write_bytes(b"")
    binary = tmp_path / "binary.nas"
    binary.write_bytes(bytes(range(256)) * 8)
    directory = tmp_path / "dir.fem"
    directory.mkdir()
    foreign = tmp_path / "deck.txt"
    foreign.write_text(_MINIMAL, encoding="utf-8")

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(binary)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "missing.bdf")) is False
    # Right content, wrong suffix: the suffix gate still applies.
    assert driver.can_open(str(foreign)) is False


def test_sniff_never_raises_on_binary(tmp_path):
    """``_sniff`` returns a bool on undecodable bytes rather than raising."""
    path = tmp_path / "binary.bdf"
    path.write_bytes(bytes(range(256)) * 8)
    assert NastranDriver()._sniff(str(path)) is False


def test_can_open_requires_both_a_grid_and_a_face_card(tmp_path):
    """A deck with GRID but no surface card is not yet recognisable as this driver's."""
    grid_only = "\n".join(
        [
            _small("GRID", "1", "0", "0.0", "0.0", "0.0"),
            _small("GRID", "2", "0", "1.0", "0.0", "0.0"),
            "",
        ]
    )
    assert NastranDriver().can_open(_write(tmp_path, grid_only)) is False


# -- registration surface -----------------------------------------------------------


def test_read_yields_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a ``UgridDataset``."""
    assert isinstance(_read(_CYLINDER), UgridDataset)


def test_driver_identity_and_capabilities():
    """The driver names itself ``nastran``, claims the deck suffixes, and reads meshes."""
    assert NastranDriver.name == "nastran"
    assert NastranDriver.suffixes == (".bdf", ".nas", ".fem")
    assert set(NastranDriver.capabilities) == {"read_mesh"}


def test_grid_with_a_non_zero_coordinate_frame_is_refused(tmp_path):
    """A GRID card in a non-zero CP frame is refused rather than read as global coords."""
    deck = (
        "GRID,1,1,0.0,0.0,0.0\n"
        "GRID,2,0,1.0,0.0,0.0\n"
        "GRID,3,0,0.0,1.0,0.0\n"
        "CTRIA3,10,1,1,2,3\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="coordinate frame CP="):
        _read(_write(tmp_path, deck))


def test_short_grid_card_is_refused(tmp_path):
    """A GRID card short of its id and both planar coordinates is a format error."""
    text = "GRID,1,0\nCTRIA3,1,0,1,2,3\n"
    with pytest.raises(FileFormatNotSupportedError, match="short GRID card"):
        _read(_write(tmp_path, text, name="shortgrid.bdf"))


def test_short_face_card_is_refused(tmp_path):
    """A CTRIA3 card short of its id, property id and three node ids is refused."""
    text = "GRID,1,0,0.0,0.0\nGRID,2,0,1.0,0.0\nGRID,3,0,0.0,1.0\nCTRIA3,1,0,1\n"
    with pytest.raises(FileFormatNotSupportedError, match="short CTRIA3 record"):
        _read(_write(tmp_path, text, name="shortface.bdf"))


def test_non_integer_node_id_is_refused(tmp_path):
    """A GRID id field that is not an integer names the card as a format error."""
    text = "GRID,x,0,0.0,0.0\nCTRIA3,1,0,1,2,3\n"
    with pytest.raises(FileFormatNotSupportedError, match="not an integer id"):
        _read(_write(tmp_path, text, name="badid.bdf"))


def test_sniff_reads_past_a_large_node_block(tmp_path):
    """A deck whose GRID block exceeds any fixed sniff window still opens (face follows)."""
    grids = "".join(f"GRID,{i},0,0.0,0.0,0.0\n" for i in range(1, 20001))
    deck = (
        grids
        + "GRID,20001,0,1.0,0.0,0.0\nGRID,20002,0,0.0,1.0,0.0\n"
        + "CTRIA3,1,1,1,20001,20002\n"
    )
    assert NastranDriver().can_open(str(_write(tmp_path, deck))) is True
