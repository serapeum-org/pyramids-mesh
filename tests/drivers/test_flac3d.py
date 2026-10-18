"""Tests for the FLAC3D grid reader, against the committed meshio sample.

The fixture is a genuine Itasca ``.f3grid`` committed under ``tests/data/flac3d/``
(``flac3d_mesh_ex.f3grid``, meshio test data, MIT): 160 ``G`` gridpoints and 108 ``Z``
zones mixing bricks (``B8``), wedges (``W6``), pyramids (``P5``) and tetrahedra (``T4``).
The tests read the committed bytes, so they cover the traps a real FLAC3D grid carries:
mixed-width zones, the ``ZGROUP``/``FGROUP`` groupings and the ``F`` face table that must
be ignored, and zones mapped verbatim to fill-padded face rows (the Exodus precedent).
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.flac3d import Flac3dDriver

_DATA = Path(__file__).parent.parent / "data" / "flac3d"
_MESH = _DATA / "flac3d_mesh_ex.f3grid"


def _read(path):
    """Read a committed FLAC3D grid, silencing the winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Flac3dDriver().read(str(path))


def _write(tmp_path, name, text):
    """Write a FLAC3D snippet to tmp_path and return its path as a string."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_reads_gridpoints_and_zones():
    """The sample builds a 160-gridpoint mesh of 108 zone faces, padded to width 8."""
    mesh = _read(_MESH).mesh

    assert mesh.n_node == 160
    assert mesh.n_face == 108
    # The widest zone is the eight-node B8 brick, so the table is padded to width 8.
    assert mesh.face_node_connectivity.data.shape == (108, 8)


def test_gridpoint_coordinates():
    """The first gridpoint is the origin, as the committed G records give it."""
    mesh = _read(_MESH).mesh

    assert float(mesh.node_x[0]) == pytest.approx(0.0)
    assert float(mesh.node_y[0]) == pytest.approx(0.0)


def test_connectivity_is_resolved_to_zero_based():
    """The 1-based gridpoint ids arrive 0-based, with the base recorded for a writer."""
    parsed = Flac3dDriver()._parse(str(_MESH), lazy=False)
    # As parsed, the connectivity is already 0-based (resolved through the id map);
    # the -1 entries are padding on the short, non-brick zones.
    real = parsed.faces[parsed.faces != -1]
    assert int(real.min()) == 0
    assert parsed.start_index == 0
    assert parsed.original_start_index == 1

    connectivity = _read(_MESH).mesh.face_node_connectivity
    data = connectivity.data
    assert int(data[data != -1].min()) == 0
    assert connectivity.original_start_index == 1


def test_first_brick_zone_maps_eight_gridpoints():
    """``Z B8 1 1 2 3 4 5 6 7 8`` becomes an eight-node face of 0-based [0..7]."""
    faces = _read(_MESH).mesh.face_node_connectivity.data
    real = faces[0][faces[0] != -1]

    # Winding may reverse the row, so compare the vertex set, not the order.
    assert sorted(int(v) for v in real) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_short_zones_are_fill_padded():
    """A four-node T4 tetrahedron pads to the brick width 8 with four -1 entries."""
    faces = _read(_MESH).mesh.face_node_connectivity.data
    # The final zone is `Z T4 108 160 89 93 97` -- four gridpoints.
    last = faces[-1]

    assert int((last == -1).sum()) == 4
    real = last[last != -1]
    # 1-based [160, 89, 93, 97] -> 0-based [159, 88, 92, 96].
    assert sorted(int(v) for v in real) == [88, 92, 96, 159]


def test_node_z_is_carried_for_the_three_dimensional_grid():
    """A FLAC3D grid is 3D, so the gridpoint z column becomes node_z."""
    dataset = _read(_MESH)
    parsed = Flac3dDriver()._parse(str(_MESH), lazy=False)

    assert parsed.node_z is not None
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (160,)


def test_zone_types_are_recorded_as_metadata():
    """The per-zone type codes ride along as metadata, in zone order."""
    attributes = _read(_MESH).global_attributes

    assert attributes["source"] == "FLAC3D"
    assert len(attributes["zone_types"]) == 108
    assert attributes["zone_types"][0] == "B8"
    assert attributes["zone_types"][-1] == "T4"
    assert set(attributes["zone_types"]) == {"B8", "W6", "P5", "T4"}


def test_group_and_face_tables_are_ignored():
    """ZGROUP/FGROUP and the F face table do not add nodes or faces."""
    # The sample carries ZGROUP, an F face table and FGROUP after the zones; none of it
    # must change the node or face counts, which come from G and Z alone.
    mesh = _read(_MESH).mesh

    assert mesh.n_node == 160
    assert mesh.n_face == 108


def test_read_dispatches_to_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a UgridDataset."""
    assert isinstance(_read(_MESH), UgridDataset)


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else."""
    assert set(Flac3dDriver.capabilities) == {"read_mesh"}
    assert Flac3dDriver.name == "flac3d"
    assert Flac3dDriver.suffixes == (".f3grid",)


def test_can_open_accepts_flac3d_and_declines_others(tmp_path):
    """can_open accepts the sample and declines foreign content, junk, binary and a dir."""
    driver = Flac3dDriver()
    assert driver.can_open(str(_MESH)) is True

    # Kratos content under a claimed suffix: it has no G/Z lines, so the sniff declines.
    kratos = tmp_path / "foreign.f3grid"
    kratos.write_text(
        "Begin Nodes\n1 0 0 0\nEnd Nodes\nBegin Elements E\nEnd Elements\n",
        encoding="utf-8",
    )
    assert driver.can_open(str(kratos)) is False

    # A gridpoint but no zone line: both are required, so this is declined.
    only_gridpoints = tmp_path / "nozone.f3grid"
    only_gridpoints.write_text("G 1 0.0 0.0 0.0\n", encoding="utf-8")
    assert driver.can_open(str(only_gridpoints)) is False

    junk = tmp_path / "junk.f3grid"
    junk.write_text("not a grid at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    binary = tmp_path / "binary.f3grid"
    binary.write_bytes(bytes(range(256)) * 4)
    assert driver.can_open(str(binary)) is False

    empty = tmp_path / "empty.f3grid"
    empty.write_bytes(b"")
    assert driver.can_open(str(empty)) is False

    assert driver.can_open(str(tmp_path / "missing.f3grid")) is False

    directory = tmp_path / "folder.f3grid"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False

    # Right content, wrong suffix: the suffix gate still applies.
    foreign_suffix = tmp_path / "grid.txt"
    foreign_suffix.write_text("G 1 0 0 0\nZ T4 1 1 1 1 1\n", encoding="utf-8")
    assert driver.can_open(str(foreign_suffix)) is False


def test_unknown_zone_type_is_refused(tmp_path):
    """A zone type outside the FLAC3D table is refused, naming the code."""
    text = "G 1 0 0 0\nG 2 1 0 0\nG 3 0 1 0\nZ H20 1 1 2 3\n"
    with pytest.raises(FileFormatNotSupportedError, match="zone type 'H20'"):
        Flac3dDriver().read(_write(tmp_path, "badzone.f3grid", text))


def test_short_zone_row_is_refused(tmp_path):
    """A zone row short of the gridpoints its type declares is a format error."""
    text = "G 1 0 0 0\nG 2 1 0 0\nG 3 0 1 0\nZ T4 1 1 2 3\n"
    with pytest.raises(FileFormatNotSupportedError, match="short T4 zone"):
        Flac3dDriver().read(_write(tmp_path, "shortzone.f3grid", text))


def test_file_without_gridpoints_is_refused(tmp_path):
    """A file carrying zones but no gridpoints is not a readable grid."""
    text = "Z T4 1 1 2 3 4\n"
    with pytest.raises(FileFormatNotSupportedError, match="no G gridpoints"):
        Flac3dDriver().read(_write(tmp_path, "nogrid.f3grid", text))


def test_file_without_zones_is_refused(tmp_path):
    """A file carrying gridpoints but no zones has no connectivity."""
    text = "G 1 0 0 0\nG 2 1 0 0\n"
    with pytest.raises(FileFormatNotSupportedError, match="no Z zones"):
        Flac3dDriver().read(_write(tmp_path, "nozones.f3grid", text))


def test_non_numeric_coordinate_is_refused(tmp_path):
    """A non-numeric gridpoint coordinate is a format error naming the keyword."""
    text = "G 1 x 0 0\nZ T4 1 1 1 1 1\n"
    with pytest.raises(FileFormatNotSupportedError, match="malformed G field"):
        Flac3dDriver().read(_write(tmp_path, "badcoord.f3grid", text))


def test_gapped_gridpoint_ids_are_resolved_by_position(tmp_path):
    """Gapped, out-of-order ids resolve by position, not by value."""
    text = (
        "G 7 0.0 0.0 0.0\n"
        "G 3 1.0 0.0 0.0\n"
        "G 5 0.0 1.0 0.0\n"
        "G 9 0.0 0.0 1.0\n"
        "Z T4 1 7 3 5 9\n"
    )
    parsed = Flac3dDriver()._parse(_write(tmp_path, "gapped.f3grid", text), lazy=False)

    # Ids 7, 3, 5, 9 map to their positions 0, 1, 2, 3 -- not to their values.
    assert sorted(int(v) for v in parsed.faces[0]) == [0, 1, 2, 3]
