"""Tests for the Kratos Multiphysics ``.mdpa`` reader, against the committed sample.

The fixture is a genuine Kratos model part committed under ``tests/data/kratos/``
(``small_cube.mdpa``, KratosMultiphysics/Kratos, BSD-3-Clause): 8 ``Begin Nodes`` and 6
``Element3D4N`` tetrahedra. The tests read the committed bytes, so they cover the traps a
real ``.mdpa`` carries: paired ``Begin``/``End`` blocks, the element/property ids that
lead each element row, and volume elements mapped verbatim to face rows (the Exodus
precedent). A tiny inline ``Element2D3N`` snippet covers the 2D-surface preference.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.kratos import KratosDriver

_DATA = Path(__file__).parent.parent / "data" / "kratos"
_CUBE = _DATA / "small_cube.mdpa"


def _read(path):
    """Read a committed Kratos model part, silencing winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KratosDriver().read(str(path))


def _write(tmp_path, name, text):
    """Write an ``.mdpa`` snippet to tmp_path and return its path as a string."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_reads_nodes_and_tetra_elements():
    """small_cube.mdpa builds an 8-node mesh of 6 four-node (tetra) faces."""
    mesh = _read(_CUBE).mesh

    assert mesh.n_node == 8
    assert mesh.n_face == 6
    assert mesh.face_node_connectivity.data.shape == (6, 4)


def test_connectivity_is_resolved_to_zero_based():
    """The 1-based node ids arrive 0-based, with the base recorded for a writer."""
    parsed = KratosDriver()._parse(str(_CUBE), lazy=False)
    assert int(parsed.faces.min()) == 0
    assert parsed.start_index == 0
    assert parsed.original_start_index == 1

    connectivity = _read(_CUBE).mesh.face_node_connectivity
    assert int(connectivity.data.min()) == 0
    assert connectivity.original_start_index == 1


def test_first_element_drops_the_leading_ids():
    """`1 1 2 6 8 1` drops elem_id and prop_id, mapping nodes 2,6,8,1 to 0-based."""
    faces = _read(_CUBE).mesh.face_node_connectivity.data
    real = faces[0][faces[0] != -1]

    # 1-based nodes [2, 6, 8, 1] -> 0-based [1, 5, 7, 0]; winding may reorder, so
    # compare the vertex set.
    assert sorted(int(v) for v in real) == [0, 1, 5, 7]


def test_node_z_is_carried_for_the_three_dimensional_cube():
    """The cube spans z 0..1, so the node z column becomes node_z."""
    dataset = _read(_CUBE)
    parsed = KratosDriver()._parse(str(_CUBE), lazy=False)

    assert parsed.node_z is not None
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (8,)


def test_element_type_is_recorded_as_metadata():
    """The element type rides along as metadata."""
    attributes = _read(_CUBE).global_attributes

    assert attributes["source"] == "Kratos"
    assert attributes["element_types"] == ["Element3D4N"]


def test_read_dispatches_to_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a UgridDataset."""
    assert isinstance(_read(_CUBE), UgridDataset)


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else."""
    assert set(KratosDriver.capabilities) == {"read_mesh"}
    assert KratosDriver.name == "kratos"
    assert KratosDriver.suffixes == (".mdpa",)


_SURFACE_AND_VOLUME = """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 1.0
End Nodes

Begin Elements Element2D3N
1 1 1 2 3
2 1 2 4 3
End Elements

Begin Elements Element3D4N
1 1 1 2 3 4
End Elements
"""


def test_two_dimensional_surface_elements_are_preferred(tmp_path):
    """When 2D surface elements are present they build the mesh, not the 3D volumes."""
    path = _write(tmp_path, "mixed.mdpa", _SURFACE_AND_VOLUME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = KratosDriver().read(path)

    mesh = dataset.mesh
    # Two Element2D3N triangles, three nodes wide -- the lone Element3D4N tet is left out.
    assert mesh.n_face == 2
    assert mesh.face_node_connectivity.data.shape == (2, 3)
    assert dataset.global_attributes["element_types"] == ["Element2D3N"]


def test_submodelpart_block_is_read_past(tmp_path):
    """A SubModelPart block's node-id list is not mistaken for Nodes or Elements."""
    text = (
        "Begin Nodes\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "End Nodes\n"
        "Begin Elements Element2D3N\n"
        "1 1 1 2 3\n"
        "End Elements\n"
        "Begin SubModelPart Inlet\n"
        "Begin SubModelPartNodes\n"
        "1\n"
        "2\n"
        "End SubModelPartNodes\n"
        "End SubModelPart\n"
    )
    mesh = KratosDriver().read(_write(tmp_path, "sub.mdpa", text)).mesh

    # The SubModelPart lists nodes 1 and 2 again -- they must not inflate the node count.
    assert mesh.n_node == 3
    assert mesh.n_face == 1


def test_can_open_accepts_kratos_and_declines_others(tmp_path):
    """can_open accepts the sample and declines foreign content, junk, binary and a dir."""
    driver = KratosDriver()
    assert driver.can_open(str(_CUBE)) is True

    # FLAC3D content under a claimed suffix: no Begin Nodes/Elements, so it is declined.
    flac3d = tmp_path / "foreign.mdpa"
    flac3d.write_text("G 1 0 0 0\nZ T4 1 1 2 3 4\n", encoding="utf-8")
    assert driver.can_open(str(flac3d)) is False

    # Nodes but no Elements block: both are required.
    only_nodes = tmp_path / "noelem.mdpa"
    only_nodes.write_text("Begin Nodes\n1 0 0 0\nEnd Nodes\n", encoding="utf-8")
    assert driver.can_open(str(only_nodes)) is False

    junk = tmp_path / "junk.mdpa"
    junk.write_text("not a model part at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    binary = tmp_path / "binary.mdpa"
    binary.write_bytes(bytes(range(256)) * 4)
    assert driver.can_open(str(binary)) is False

    empty = tmp_path / "empty.mdpa"
    empty.write_bytes(b"")
    assert driver.can_open(str(empty)) is False

    assert driver.can_open(str(tmp_path / "missing.mdpa")) is False

    directory = tmp_path / "folder.mdpa"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False

    # Right content, wrong suffix: the suffix gate still applies.
    foreign_suffix = tmp_path / "model.txt"
    foreign_suffix.write_text(
        "Begin Nodes\n1 0 0 0\nEnd Nodes\nBegin Elements E2N\nEnd Elements\n",
        encoding="utf-8",
    )
    assert driver.can_open(str(foreign_suffix)) is False


def test_element_type_without_node_count_is_refused(tmp_path):
    """An Elements block whose type name carries no <count>N is a format error."""
    text = (
        "Begin Nodes\n1 0 0 0\n2 1 0 0\n3 0 1 0\nEnd Nodes\n"
        "Begin Elements SomeElement\n1 1 1 2 3\nEnd Elements\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="carries no node count"):
        KratosDriver().read(_write(tmp_path, "notype.mdpa", text))


def test_short_element_row_is_refused(tmp_path):
    """An element row short of its node ids is a format error, not an IndexError."""
    text = (
        "Begin Nodes\n1 0 0 0\n2 1 0 0\n3 0 1 0\nEnd Nodes\n"
        "Begin Elements Element2D3N\n1 1 1 2\nEnd Elements\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="short Element2D3N row"):
        KratosDriver().read(_write(tmp_path, "short.mdpa", text))


def test_file_without_nodes_is_refused(tmp_path):
    """A file carrying elements but no nodes is not a readable mesh."""
    text = "Begin Elements Element2D3N\n1 1 1 2 3\nEnd Elements\n"
    with pytest.raises(FileFormatNotSupportedError, match="declares no Nodes"):
        KratosDriver().read(_write(tmp_path, "nonodes.mdpa", text))


def test_file_without_elements_is_refused(tmp_path):
    """A file with nodes but no elements has no connectivity to build from."""
    text = "Begin Nodes\n1 0 0 0\n2 1 0 0\nEnd Nodes\n"
    with pytest.raises(FileFormatNotSupportedError, match="declares no Elements"):
        KratosDriver().read(_write(tmp_path, "noelements.mdpa", text))


def test_non_numeric_coordinate_is_refused(tmp_path):
    """A non-numeric node coordinate is a format error naming the block."""
    text = "Begin Nodes\n1 x 0 0\nEnd Nodes\nBegin Elements E3N\nEnd Elements\n"
    with pytest.raises(FileFormatNotSupportedError, match="malformed NODES field"):
        KratosDriver().read(_write(tmp_path, "badcoord.mdpa", text))


def test_short_nodes_row_is_refused(tmp_path):
    """A Nodes row short of its id and three coordinates is a format error."""
    text = (
        "Begin Nodes\n1 0.0 0.0\nEnd Nodes\n"
        "Begin Elements Element2D3N\n1 1 1 2 3\nEnd Elements\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="short Nodes row"):
        KratosDriver().read(_write(tmp_path, "shortnode.mdpa", text))


def test_undefined_node_reference_is_refused(tmp_path):
    """An element naming a node id no Nodes row defines is refused, not guessed."""
    text = (
        "Begin Nodes\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\nEnd Nodes\n"
        "Begin Elements Element2D3N\n1 1 1 2 99\nEnd Elements\n"
    )
    with pytest.raises(ValueError, match="node id 99"):
        KratosDriver().read(_write(tmp_path, "badref.mdpa", text))


def test_non_integer_element_node_id_is_refused(tmp_path):
    """A non-integer node id in an element row names the block as a format error."""
    text = (
        "Begin Nodes\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\nEnd Nodes\n"
        "Begin Elements Element2D3N\n1 1 1 2 x\nEnd Elements\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="malformed ELEMENTS field"):
        KratosDriver().read(_write(tmp_path, "badint.mdpa", text))


def test_gapped_node_ids_are_resolved_by_position(tmp_path):
    """Gapped, out-of-order node ids resolve by position, not by value."""
    text = (
        "Begin Nodes\n"
        "7 0.0 0.0 0.0\n"
        "3 1.0 0.0 0.0\n"
        "5 0.0 1.0 0.0\n"
        "End Nodes\n"
        "Begin Elements Element2D3N\n"
        "1 1 7 3 5\n"
        "End Elements\n"
    )
    parsed = KratosDriver()._parse(_write(tmp_path, "gapped.mdpa", text), lazy=False)

    # Ids 7, 3, 5 map to their positions 0, 1, 2 -- not to their values.
    assert sorted(int(v) for v in parsed.faces[0]) == [0, 1, 2]
