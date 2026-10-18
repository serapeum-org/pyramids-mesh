"""Tests for the AVS-UCD reader, against a real committed sample.

The fixture ``tests/data/avsucd/02_tet_mesh.inp`` is a genuine AVS-UCD ASCII mesh
(LANL/LaGriT, BSD-3-Clause): 1122 nodes and 4800 ``tet`` cells -- a solid mesh, read
as four-node faces exactly as the ``exodus`` reader maps ``TETRA``. The co-located
``.inp`` Abaqus sample is reused to prove the two drivers never both accept one file.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import pytest

from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.abaqus import AbaqusDriver
from pyramids_mesh.drivers.avsucd import AvsUcdDriver

_DATA = Path(__file__).parent.parent / "data"
_AVSUCD = _DATA / "avsucd" / "02_tet_mesh.inp"
# The other .inp format's committed sample, for the cross-decline assertion.
_ABAQUS = _DATA / "abaqus" / "nle1xf3c.inp"


def _read(path):
    """Read a committed file, silencing the winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AvsUcdDriver().read(str(path))


def test_reads_nodes_and_tet_faces():
    """The tet mesh builds 1122 nodes and 4800 four-node faces."""
    mesh = _read(_AVSUCD).mesh
    assert mesh.n_node == 1122
    assert mesh.n_face == 4800
    assert mesh.face_node_connectivity.data.shape == (4800, 4)


def test_connectivity_is_label_resolved_to_zero_based():
    """Cell node labels resolve to 0-based indices, with the ``1`` base recorded."""
    parsed = AvsUcdDriver()._parse(str(_AVSUCD), lazy=False)
    # First cell `00001 1 tet 626 682 681 693` -> 0-based (625, 681, 680, 692).
    assert parsed.faces[0].tolist() == [625, 681, 680, 692]
    assert parsed.start_index == 0
    assert parsed.original_start_index == 1

    connectivity = _read(_AVSUCD).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1


def test_solid_mesh_carries_node_z():
    """The 3D sample has a non-zero z column, so ``node_z`` is carried."""
    assert AvsUcdDriver()._parse(str(_AVSUCD), lazy=False).node_z is not None


def test_surface_quad_cell(tmp_path):
    """A ``quad`` surface cell reads as a single four-node, planar face."""
    mesh_file = tmp_path / "quad.inp"
    mesh_file.write_text(
        "4 1 0 0 0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 1.0 1.0 0.0\n"
        "4 0.0 1.0 0.0\n"
        "1 1 quad 1 2 3 4\n",
        encoding="utf-8",
    )
    parsed = AvsUcdDriver()._parse(str(mesh_file), lazy=False)
    assert parsed.faces.shape == (1, 4)
    assert parsed.faces[0].tolist() == [0, 1, 2, 3]
    # An all-zero z column is dropped rather than kept as a meaningless node_z.
    assert parsed.node_z is None

    dataset = _read(mesh_file)
    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 1


def test_point_and_line_cells_are_dropped(tmp_path):
    """``pt`` and ``line`` cells are skipped; only the face-forming cell survives."""
    mesh_file = tmp_path / "mixed.inp"
    mesh_file.write_text(
        "3 3 0 0 0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "1 1 pt 1\n"
        "2 1 line 1 2\n"
        "3 1 tri 1 2 3\n",
        encoding="utf-8",
    )
    parsed = AvsUcdDriver()._parse(str(mesh_file), lazy=False)
    assert parsed.faces.shape == (1, 3)
    assert parsed.faces[0].tolist() == [0, 1, 2]


def test_read_dispatches_to_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a ``UgridDataset``."""
    assert isinstance(_read(_AVSUCD), UgridDataset)


def test_can_open_accepts_avsucd_and_declines_others(tmp_path):
    """``can_open`` accepts the mesh and declines the Abaqus ``.inp``, junk and a dir."""
    driver = AvsUcdDriver()
    assert driver.can_open(str(_AVSUCD)) is True

    # The co-located Abaqus sample shares the .inp suffix; its first significant line
    # is a `*` keyword, not five integers, so this driver must decline it.
    assert driver.can_open(str(_ABAQUS)) is False

    junk = tmp_path / "junk.inp"
    junk.write_text("not an avs mesh at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.inp"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_the_other_driver_declines_this_sample():
    """The Abaqus driver declines the AVS-UCD mesh, so neither claims both."""
    assert AbaqusDriver().can_open(str(_AVSUCD)) is False


def test_avs_suffix_is_claimed():
    """The driver claims both the ``.inp`` and ``.avs`` suffixes."""
    assert set(AvsUcdDriver.suffixes) == {".inp", ".avs"}


def test_declares_read_mesh_only():
    """The driver advertises exactly the ``read_mesh`` capability."""
    assert set(AvsUcdDriver.capabilities) == {"read_mesh"}


def test_sniff_never_raises_on_binary(tmp_path):
    """A binary file under a claimed suffix is declined, never raised over."""
    binary = tmp_path / "binary.avs"
    binary.write_bytes(bytes(range(256)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert AvsUcdDriver().can_open(str(binary)) is False


def test_cells_only_pt_line_is_refused(tmp_path):
    """A file whose only cells are 0-D/1-D builds no face and is refused."""
    mesh_file = tmp_path / "nofaces.inp"
    mesh_file.write_text(
        "2 1 0 0 0\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n1 1 line 1 2\n",
        encoding="utf-8",
    )
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_sniff_declines_a_comment_only_file(tmp_path):
    """A file of only comment lines yields no header record, so it is declined."""
    mesh_file = tmp_path / "comments.inp"
    mesh_file.write_text("# just a comment\n# another\n", encoding="utf-8")
    assert AvsUcdDriver().can_open(str(mesh_file)) is False


def test_empty_file_is_refused(tmp_path):
    """An empty file has no five-integer header and is refused."""
    mesh_file = tmp_path / "empty.inp"
    mesh_file.write_text("", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_malformed_header_is_refused(tmp_path):
    """A first record that is not five integers is refused as a bad header."""
    mesh_file = tmp_path / "badheader.inp"
    mesh_file.write_text("1 2 3\n1 0.0 0.0 0.0\n", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_short_node_record_is_refused(tmp_path):
    """A node row missing its z coordinate is refused."""
    mesh_file = tmp_path / "shortnode.inp"
    mesh_file.write_text("1 1 0 0 0\n1 0.0 0.0\n1 1 tri 1 2 3\n", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_short_cell_record_is_refused(tmp_path):
    """A cell row short of a type and one node is refused."""
    mesh_file = tmp_path / "shortcell.inp"
    mesh_file.write_text("1 1 0 0 0\n1 0.0 0.0 0.0\n1 1 tri\n", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_unknown_cell_type_is_refused(tmp_path):
    """A cell naming a type that is neither face-forming nor dropped is refused."""
    mesh_file = tmp_path / "unknowncell.inp"
    mesh_file.write_text(
        "3 1 0 0 0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "1 1 banana 1 2 3\n",
        encoding="utf-8",
    )
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_truncated_file_is_refused(tmp_path):
    """A header promising more nodes than the file carries is refused."""
    mesh_file = tmp_path / "truncated.inp"
    mesh_file.write_text("3 1 0 0 0\n1 0.0 0.0 0.0\n", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_cell_referencing_an_undefined_node_is_refused(tmp_path):
    """A cell naming a node label no node row defines raises ``ValueError``."""
    mesh_file = tmp_path / "dangling.inp"
    mesh_file.write_text(
        "3 1 0 0 0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "1 1 tri 1 2 99\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_non_integer_cell_node_is_refused(tmp_path):
    """A non-integer node value in a cell row is refused."""
    mesh_file = tmp_path / "badcellnode.inp"
    mesh_file.write_text(
        "3 1 0 0 0\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "1 1 tri 1 2 x\n",
        encoding="utf-8",
    )
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)


def test_non_numeric_node_coordinate_is_refused(tmp_path):
    """A non-numeric coordinate in a node row is refused."""
    mesh_file = tmp_path / "badcoord.inp"
    mesh_file.write_text("1 1 0 0 0\n1 x 0.0 0.0\n1 1 tri 1 2 3\n", encoding="utf-8")
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError):
        AvsUcdDriver()._parse(str(mesh_file), lazy=False)
