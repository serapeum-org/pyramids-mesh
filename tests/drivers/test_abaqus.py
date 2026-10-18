"""Tests for the Abaqus keyword-deck reader, against a real committed sample.

The fixture ``tests/data/abaqus/nle1xf3c.inp`` is a genuine Abaqus deck (meshio test
data, MIT): the NAFEMS LE1 plane-stress membrane, ``*NODE`` plus an ``*ELEMENT,
TYPE=CPS3`` block of twelve triangles over twelve nodes. The co-located ``.inp``
AVS-UCD sample is reused to prove the two drivers never both accept one file.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import pytest

from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.abaqus import AbaqusDriver
from pyramids_mesh.drivers.avsucd import AvsUcdDriver

_DATA = Path(__file__).parent.parent / "data"
_ABAQUS = _DATA / "abaqus" / "nle1xf3c.inp"
# The other .inp format's committed sample, for the cross-decline assertion.
_AVSUCD = _DATA / "avsucd" / "02_tet_mesh.inp"


def _read(path):
    """Read a committed deck, silencing the winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AbaqusDriver().read(str(path))


def test_reads_nodes_and_triangular_faces():
    """The CPS3 deck builds a 12-node, 12-triangle mesh."""
    mesh = _read(_ABAQUS).mesh
    assert mesh.n_node == 12
    assert mesh.n_face == 12
    assert mesh.face_node_connectivity.data.shape == (12, 3)


def test_connectivity_is_label_resolved_to_zero_based():
    """Node labels resolve to 0-based indices, with the ``1`` base recorded."""
    parsed = AbaqusDriver()._parse(str(_ABAQUS), lazy=False)
    # `*ELEMENT ... 1, 5, 2, 1` -> labels (5, 2, 1) -> 0-based (4, 1, 0).
    assert parsed.faces[0].tolist() == [4, 1, 0]
    assert parsed.start_index == 0
    assert parsed.original_start_index == 1

    connectivity = _read(_ABAQUS).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1


def test_planar_deck_has_no_node_z():
    """Every node's z is zero, so no ``node_z`` variable is built."""
    assert AbaqusDriver()._parse(str(_ABAQUS), lazy=False).node_z is None
    assert "node_z" not in set(_read(_ABAQUS).data_variable_names)


def test_volumetric_element_maps_to_a_face(tmp_path):
    """A ``C3D4`` tet has its node list read into a face row, not an empty mesh."""
    deck = tmp_path / "solid.inp"
    deck.write_text(
        "*HEADING\n"
        "a tiny solid deck\n"
        "*NODE\n"
        "1, 0.0, 0.0, 0.0\n"
        "2, 1.0, 0.0, 0.0\n"
        "3, 0.0, 1.0, 0.0\n"
        "4, 0.0, 0.0, 1.0\n"
        "*ELEMENT, TYPE=C3D4\n"
        "1, 1, 2, 3, 4\n"
        "*END\n",
        encoding="utf-8",
    )
    parsed = AbaqusDriver()._parse(str(deck), lazy=False)
    assert parsed.faces.shape == (1, 4)
    assert parsed.faces[0].tolist() == [0, 1, 2, 3]
    # A non-zero z column makes it a 3D deck, so node_z is carried.
    assert parsed.node_z is not None
    assert _read(deck).mesh.n_face == 1


def test_read_dispatches_to_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a ``UgridDataset``."""
    assert isinstance(_read(_ABAQUS), UgridDataset)


def test_can_open_accepts_abaqus_and_declines_others(tmp_path):
    """``can_open`` accepts the deck and declines the AVS-UCD ``.inp``, junk and a dir."""
    driver = AbaqusDriver()
    assert driver.can_open(str(_ABAQUS)) is True

    # The co-located AVS-UCD sample shares the .inp suffix; its first significant line
    # is five integers, not a `*` keyword, so this driver must decline it.
    assert driver.can_open(str(_AVSUCD)) is False

    junk = tmp_path / "junk.inp"
    junk.write_text("not a keyword deck at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.inp"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_the_other_driver_declines_this_sample():
    """The AVS-UCD driver declines the Abaqus deck, so neither claims both."""
    assert AvsUcdDriver().can_open(str(_ABAQUS)) is False


def test_declares_read_mesh_only():
    """The driver advertises exactly the ``read_mesh`` capability."""
    assert set(AbaqusDriver.capabilities) == {"read_mesh"}


def test_sniff_never_raises_on_binary(tmp_path):
    """A binary file under the claimed suffix is declined, never raised over."""
    binary = tmp_path / "binary.inp"
    binary.write_bytes(bytes(range(256)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert AbaqusDriver().can_open(str(binary)) is False


def test_deck_without_nodes_is_refused(tmp_path):
    """A deck with elements but no ``*NODE`` is not a readable mesh."""
    deck = tmp_path / "nonodes.inp"
    deck.write_text("*ELEMENT, TYPE=CPS3\n1, 1, 2, 3\n", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AbaqusDriver()._parse(str(deck), lazy=False)
