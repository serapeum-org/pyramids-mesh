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
from pyramids_mesh.drivers.srh2d import Srh2dDriver

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
