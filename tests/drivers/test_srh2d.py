"""Tests for the SRH-2D reader (C12), against the real committed Muncie trio.

SRH-2D stores a mesh as an ASCII trio sharing a basename stem. The public-domain HEC-RAS
Muncie example is committed under ``tests/data/srh2d/`` (staged from the MIT pyHMT2D
project): ``Muncie.srhgeom`` (19407 mixed tri/quad/polygon elements, 21520 nodes),
``Muncie.srhhydro`` (control deck) and ``Muncie.srhmat`` (eight material zones). The tests
read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.srh2d import Srh2dDriver, _Srh2dReader

_DATA = Path(__file__).parent.parent / "data" / "srh2d"
_GEOM = _DATA / "Muncie.srhgeom"
_HYDRO = _DATA / "Muncie.srhhydro"

_N_NODE = 21520
_N_ELEM = 19407


def _read(path=_GEOM):
    """Read a committed SRH-2D member file, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Srh2dDriver().read(str(path))


def test_reads_nodes_and_elements():
    """The Muncie mesh reads its 21520 nodes and 19407 mixed-width faces."""
    mesh = _read().mesh
    assert mesh.n_node == _N_NODE
    assert mesh.n_face == _N_ELEM


def test_connectivity_is_mixed_width_fill_padded():
    """Faces span 3 to 8 nodes, so the array is 8 wide with ``-1`` padding short rows."""
    faces = _read().mesh.face_node_connectivity.data
    assert faces.shape == (_N_ELEM, 8)
    # Both padded and full-width rows exist: quads pad to 8, octagons fill every column.
    assert bool((faces == -1).any())
    assert bool((faces[:, -1] != -1).any())
    # The real (non-fill) vertex counts are exactly the SRH-2D element widths.
    widths = (faces != -1).sum(axis=1)
    assert set(np.unique(widths).tolist()) == {3, 4, 5, 6, 7, 8}


def test_indices_are_one_based_resolved_to_zero_based():
    """The 1-based node ids resolve to 0-based, and the base is recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    real = connectivity.data[connectivity.data != -1]
    assert int(real.min()) == 0
    assert int(real.max()) == _N_NODE - 1


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise after validation, fill columns ignored."""
    mesh = _read().mesh
    node_x = np.asarray(mesh.node_x)
    node_y = np.asarray(mesh.node_y)
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = node_x[vertices]
        ys = node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_bed_elevation_becomes_node_z():
    """The ``Node`` z column becomes a node-located ``node_z`` variable."""
    node_z = _read().get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (_N_NODE,)


def test_material_zones_are_a_clean_face_variable():
    """The .srhmat assigns every element one zone, so ``material`` is a face variable."""
    material = _read().get("material")
    assert material.location == "face"
    assert material.shape == (_N_ELEM,)
    # The Muncie zones are the ids 2..7 named in the .srhmat.
    assert set(np.unique(material.data).astype(int).tolist()) == {2, 3, 4, 5, 6, 7}


def test_manning_n_is_carried_from_the_hydro_deck():
    """Each element's Manning's n is looked up from its zone's ``ManningsN`` card."""
    dataset = _read()
    manning = dataset.get("manning_n")
    material = dataset.get("material")
    assert manning.location == "face"
    assert manning.shape == (_N_ELEM,)
    # The deck's ManningsN cards map zone 2->0.03, 3->0.04, 4->0.045, 5->0.05, 6->0.055,
    # 7->0.06.
    expected = {2: 0.03, 3: 0.04, 4: 0.045, 5: 0.05, 6: 0.055, 7: 0.06}
    for zone, n in expected.items():
        selected = manning.data[material.data == zone]
        assert selected.size > 0
        assert np.allclose(selected, n)


def test_read_via_srhhydro_matches_read_via_srhgeom():
    """Reading through the control deck resolves the same geometry as the .srhgeom."""
    by_geom = _read(_GEOM).mesh
    by_hydro = _read(_HYDRO).mesh
    assert by_hydro.n_node == by_geom.n_node
    assert by_hydro.n_face == by_geom.n_face
    assert np.array_equal(
        by_hydro.face_node_connectivity.data, by_geom.face_node_connectivity.data
    )


def test_read_via_srhhydro_still_carries_the_metadata():
    """The .srhmat / ManningsN metadata is found from the stem when entering via hydro."""
    dataset = _read(_HYDRO)
    assert "material" in dataset.data_variable_names
    assert "manning_n" in dataset.data_variable_names


def test_can_open_accepts_both_member_files():
    """can_open accepts the .srhgeom and the .srhhydro of a resolvable trio."""
    driver = Srh2dDriver()
    assert driver.can_open(str(_GEOM)) is True
    assert driver.can_open(str(_HYDRO)) is True


def test_can_open_declines_junk(tmp_path):
    """can_open declines a wrong suffix, a wrong header and an orphaned control deck."""
    driver = Srh2dDriver()
    # right suffix, wrong content
    not_geom = tmp_path / "bogus.srhgeom"
    not_geom.write_text("GARBAGE 1\nnot a mesh\n")
    assert driver.can_open(str(not_geom)) is False
    # a control deck whose geometry is nowhere to be found
    orphan = tmp_path / "orphan.srhhydro"
    orphan.write_text('SRHHYDRO 30\nGrid "missing.srhgeom"\n')
    assert driver.can_open(str(orphan)) is False
    # an unrelated suffix
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False


def test_dispatch_resolves_the_project_from_either_entry():
    """The driver's own dispatch resolves the trio from the geom or the hydro file.

    The entry point is registered as ``srh2d``; here the driver is exercised directly (the
    installed package need not yet carry the entry point), which is the same resolution
    ``UgridDataset.read_file`` performs once the driver is discovered.
    """
    driver = Srh2dDriver()
    assert driver._geom_path(str(_HYDRO)) == str(_GEOM)
    assert driver._geom_path(str(_GEOM)) == str(_GEOM)
    assert driver._geom_path(str(_DATA / "Muncie.srhmat")) is None


def test_read_of_a_non_srh2d_path_is_a_format_error(tmp_path):
    """read() on a path that is neither member file is refused with a clear message."""
    other = tmp_path / "notes.txt"
    other.write_text("not srh-2d")
    with pytest.raises(FileFormatNotSupportedError, match="not an SRH-2D"):
        Srh2dDriver().read(str(other))


def test_empty_geometry_is_a_format_error(tmp_path):
    """A .srhgeom with a header but no elements is a clean format error."""
    geom = tmp_path / "empty.srhgeom"
    geom.write_text("SRHGEOM 30\nNode 1 0.0 0.0 1.0\nNode 2 1.0 0.0 1.0\n")
    with pytest.raises(FileFormatNotSupportedError, match="no Elem records"):
        Srh2dDriver().read(str(geom))


# A minimal valid geometry: one counter-clockwise triangle, no GridUnit card. The three
# nodes carry a z column, so bed elevation becomes a node_z variable.
_GEOM_BODY = (
    "SRHGEOM 30\n"
    "Node 1 0.0 0.0 1.0\n"
    "Node 2 1.0 0.0 1.0\n"
    "Node 3 0.0 1.0 1.0\n"
    "Elem 1 1 2 3\n"
)

# Two counter-clockwise triangles sharing an edge, for the incomplete-material-map path.
_GEOM_TWO_ELEM = (
    "SRHGEOM 30\n"
    "Node 1 0.0 0.0 1.0\n"
    "Node 2 1.0 0.0 1.0\n"
    "Node 3 0.0 1.0 1.0\n"
    "Node 4 1.0 1.0 1.0\n"
    "Elem 1 1 2 3\n"
    "Elem 2 2 4 3\n"
)


def test_can_open_declines_a_srhhydro_with_the_wrong_header(tmp_path):
    """A .srhhydro whose header is not SRHHYDRO is declined before resolving geometry."""
    hydro = tmp_path / "x.srhhydro"
    hydro.write_text('NOTHYDRO 30\nGrid "x.srhgeom"\n')
    assert Srh2dDriver().can_open(str(hydro)) is False


def test_can_open_warns_and_declines_when_sniffing_raises(tmp_path, monkeypatch):
    """A sniff that raises is reported as a warning and the file declined, never raised."""
    geom = tmp_path / "x.srhgeom"
    geom.write_text("SRHGEOM 30\n")
    driver = Srh2dDriver()

    def boom(_path):
        raise RuntimeError("sniff exploded")

    monkeypatch.setattr(driver, "_first_token", boom)
    with pytest.warns(UserWarning, match="failed to sniff"):
        assert driver.can_open(str(geom)) is False


def test_geom_path_returns_none_for_a_missing_file(tmp_path):
    """_geom_path declines a path that is not a file on disk."""
    assert Srh2dDriver()._geom_path(str(tmp_path / "nope.srhgeom")) is None


def test_geom_from_hydro_falls_back_to_the_stem_sibling(tmp_path):
    """With no Grid card, the geometry resolves from the <stem>.srhgeom sibling."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    hydro = tmp_path / "m.srhhydro"
    hydro.write_text("SRHHYDRO 30\n")  # no Grid card
    resolved = Srh2dDriver()._geom_from_hydro(str(hydro))
    assert resolved == str(tmp_path / "m.srhgeom")


def test_mat_path_without_a_hydromat_card_uses_the_stem(tmp_path):
    """With no HydroMat card, the .srhmat resolves from the shared stem."""
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\n")
    hydro = tmp_path / "m.srhhydro"
    hydro.write_text("SRHHYDRO 30\n")  # no HydroMat card
    resolved = Srh2dDriver()._mat_path(str(tmp_path / "m.srhgeom"), str(hydro))
    assert resolved == str(tmp_path / "m.srhmat")


def test_mat_path_ignores_a_hydromat_pointing_nowhere(tmp_path):
    """A HydroMat card naming a non-existent file, with no stem sibling, yields None."""
    hydro = tmp_path / "m.srhhydro"
    hydro.write_text('SRHHYDRO 30\nHydroMat "ghost.srhmat"\n')
    resolved = Srh2dDriver()._mat_path(str(tmp_path / "m.srhgeom"), str(hydro))
    assert resolved is None


def test_first_token_of_an_empty_file_is_none(tmp_path):
    """_first_token returns None for a file with no significant records."""
    empty = tmp_path / "empty.srhgeom"
    empty.write_text("")
    assert Srh2dDriver()._first_token(str(empty)) is None


def test_quoted_value_returns_none_for_an_unquoted_card(tmp_path):
    """A matching card with no quoted value yields None rather than the bare token."""
    deck = tmp_path / "m.srhhydro"
    deck.write_text("SRHHYDRO 30\nGrid unquoted_value\n")
    assert Srh2dDriver()._quoted_value(str(deck), "Grid") is None


def test_build_refuses_a_geometry_with_the_wrong_header(tmp_path):
    """A .srhgeom whose first record is not SRHGEOM is a clean format error."""
    geom = tmp_path / "b.srhgeom"
    geom.write_text("NOTGEOM 30\nNode 1 0.0 0.0 1.0\nElem 1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="expected a SRHGEOM header"):
        Srh2dDriver().read(str(geom))


def test_build_refuses_nodes_that_disagree_about_z(tmp_path):
    """Some nodes with a z column and some without is refused, not invented."""
    geom = tmp_path / "z.srhgeom"
    geom.write_text(
        "SRHGEOM 30\n"
        "Node 1 0.0 0.0 1.0\n"
        "Node 2 1.0 0.0\n"
        "Node 3 0.0 1.0 1.0\n"
        "Elem 1 1 2 3\n"
    )
    with pytest.raises(
        FileFormatNotSupportedError, match="disagree about the z column"
    ):
        Srh2dDriver().read(str(geom))


def test_build_refuses_a_short_node_record(tmp_path):
    """A Node record with fewer than three fields is a clean format error."""
    geom = tmp_path / "s.srhgeom"
    geom.write_text("SRHGEOM 30\nNode 1 0.0\nElem 1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="short Node record"):
        Srh2dDriver().read(str(geom))


def test_nodes_without_z_build_a_mesh_with_no_node_z(tmp_path):
    """Nodes carrying no z column build a mesh with no node_z and no grid_unit."""
    geom = tmp_path / "noz.srhgeom"
    geom.write_text(
        "SRHGEOM 30\n"
        "Node 1 0.0 0.0\n"
        "Node 2 1.0 0.0\n"
        "Node 3 0.0 1.0\n"
        "Elem 1 1 2 3\n"
    )
    dataset = Srh2dDriver().read(str(geom))
    assert "node_z" not in dataset.data_variable_names


def test_build_refuses_a_short_elem_record(tmp_path):
    """An Elem record with fewer than three node ids is a clean format error."""
    geom = tmp_path / "e.srhgeom"
    geom.write_text(
        "SRHGEOM 30\n"
        "Node 1 0.0 0.0 1.0\n"
        "Node 2 1.0 0.0 1.0\n"
        "Node 3 0.0 1.0 1.0\n"
        "Elem 1 1 2\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="short Elem record"):
        Srh2dDriver().read(str(geom))


def test_build_refuses_an_element_referencing_an_undefined_node(tmp_path):
    """An element citing a node id no Node record defines is a ValueError."""
    geom = tmp_path / "u.srhgeom"
    geom.write_text(
        "SRHGEOM 30\n"
        "Node 1 0.0 0.0 1.0\n"
        "Node 2 1.0 0.0 1.0\n"
        "Node 3 0.0 1.0 1.0\n"
        "Elem 1 1 2 9\n"
    )
    with pytest.raises(ValueError, match="references node id 9"):
        Srh2dDriver().read(str(geom))


def test_build_refuses_a_non_integer_node_id(tmp_path):
    """A Node record with a non-integer id is a clean format error."""
    geom = tmp_path / "ni.srhgeom"
    geom.write_text("SRHGEOM 30\nNode x 0.0 0.0 1.0\nElem 1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="non-integer field"):
        Srh2dDriver().read(str(geom))


def test_build_refuses_a_non_numeric_node_coordinate(tmp_path):
    """A Node record with a non-numeric coordinate is a clean format error."""
    geom = tmp_path / "nf.srhgeom"
    geom.write_text("SRHGEOM 30\nNode 1 foo 0.0 1.0\nElem 1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="non-numeric field"):
        Srh2dDriver().read(str(geom))


def test_geometry_without_a_srhmat_carries_no_material(tmp_path):
    """With neither a .srhmat nor a .srhhydro beside it, no material variable attaches."""
    geom = tmp_path / "m.srhgeom"
    geom.write_text(_GEOM_BODY)
    dataset = Srh2dDriver().read(str(geom))
    assert "material" not in dataset.data_variable_names


def test_material_without_a_hydro_deck_has_no_manning(tmp_path):
    """A clean material map with no ManningsN source attaches material but not manning_n."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\nMaterial 2 1\n")
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" in dataset.data_variable_names
    assert "manning_n" not in dataset.data_variable_names


def test_manning_dropped_when_a_zone_lacks_a_value(tmp_path):
    """manning_n is dropped when an assigned zone has no ManningsN card."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\nMaterial 9 1\n")
    (tmp_path / "m.srhhydro").write_text("SRHHYDRO 30\nManningsN 2 0.03\n")
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" in dataset.data_variable_names
    assert "manning_n" not in dataset.data_variable_names


def test_material_block_without_an_id_drops_the_map(tmp_path):
    """A Material card with no id makes the whole map ambiguous, so it is dropped."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\nMaterial\n")
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" not in dataset.data_variable_names


def test_material_continuation_before_any_block_is_ignored(tmp_path):
    """An integer line before any Material card has no open block and is skipped."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\n10 11\nMaterial 2 1\n")
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" in dataset.data_variable_names


def test_material_with_a_non_integer_element_drops_the_map(tmp_path):
    """A non-integer element token in a Material block drops the whole map."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\nMaterial 2 1 foo\n")
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" not in dataset.data_variable_names


def test_element_assigned_to_two_zones_drops_the_map(tmp_path):
    """An element assigned to two zones is ambiguous, so the material map is dropped."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_BODY)
    (tmp_path / "m.srhmat").write_text("SRHMAT 30\nMaterial 2 1\nMaterial 3 1\n")
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" not in dataset.data_variable_names


def test_incomplete_material_map_drops_the_variable(tmp_path):
    """A map leaving an element unassigned is incomplete, so no material attaches."""
    (tmp_path / "m.srhgeom").write_text(_GEOM_TWO_ELEM)
    (tmp_path / "m.srhmat").write_text(
        "SRHMAT 30\nMaterial 2 1\n"
    )  # element 2 unassigned
    dataset = Srh2dDriver().read(str(tmp_path / "m.srhgeom"))
    assert "material" not in dataset.data_variable_names


def test_manning_by_material_skips_malformed_cards(tmp_path):
    """ManningsN cards with a non-integer zone or non-numeric value are skipped."""
    hydro = tmp_path / "m.srhhydro"
    hydro.write_text(
        "SRHHYDRO 30\nManningsN foo 0.03\nManningsN 2 abc\nManningsN 3 0.04\n"
    )
    reader = _Srh2dReader(str(tmp_path / "m.srhgeom"), hydro=str(hydro), material=None)
    assert reader._manning_by_material() == {3: 0.04}


def test_manning_by_material_is_empty_without_a_hydro_deck():
    """With no .srhhydro, the ManningsN map is empty."""
    reader = _Srh2dReader("m.srhgeom", hydro=None, material=None)
    assert reader._manning_by_material() == {}


def test_quoted_or_first_returns_none_for_no_fields():
    """A card with no fields has no value to report."""
    assert _Srh2dReader._quoted_or_first([]) is None


def test_quoted_or_first_returns_the_first_unquoted_field():
    """An unquoted card value is taken from its first field."""
    assert _Srh2dReader._quoted_or_first(["Meters"]) == "Meters"
