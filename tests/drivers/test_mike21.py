"""Tests for the DHI Mike21 ``.mesh`` reader (D4).

Mike21 is a frame-only text format: a header, node records, an element header, element
records. The tests cover the index base, the projection-to-CRS mapping (a documented
trap, since a DHI name is not a WKT), and the record-shape errors. Two format details --
the projection field and the node-column order -- could not be checked against a vendor
file, so the tests pin the behaviour this driver actually implements.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers import misc
from pyramids_mesh.drivers.misc import Mike21Driver, build_minimal_fixture_mike21

_UNIT_SQUARE = """100079 1000 4 LONG/LAT
1 0.0 0.0 0.0 1
2 1.0 0.0 0.0 1
3 1.0 1.0 0.0 1
4 0.0 1.0 0.0 1
2 3 21
1 1 2 3
2 1 3 4
"""


def _write(tmp_path, text, name="domain.mesh"):
    """Write a Mike21 fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_mike21_reads_frame(tmp_path):
    """A unit-square mesh reads into four nodes and two triangular faces."""
    dataset = Mike21Driver().read(_write(tmp_path, _UNIT_SQUARE))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_mike21_index_base_verified_against_fixture(tmp_path):
    """Connectivity arrives 0-based, with the file's 1-based numbering recorded."""
    dataset = Mike21Driver().read(_write(tmp_path, _UNIT_SQUARE))

    connectivity = dataset.mesh.face_node_connectivity
    assert int(connectivity.data.min()) == 0
    assert connectivity.original_start_index == 1


def test_mike21_elevation_becomes_node_variable(tmp_path):
    """The node z column becomes a node-located ``node_z`` variable, not a coordinate."""
    text = _UNIT_SQUARE.replace("1 0.0 0.0 0.0 1", "1 0.0 0.0 2.5 1")
    dataset = Mike21Driver().read(_write(tmp_path, text))

    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (4,)
    assert node_z.data[0] == 2.5


def test_mike21_non_contiguous_node_ids_resolve_correctly(tmp_path):
    """Gapped node ids are resolved by position, so a renumbered mesh is wired right."""
    text = """100079 1000 4 LONG/LAT
10 0.0 0.0 0.0 1
20 1.0 0.0 0.0 1
30 1.0 1.0 0.0 1
40 0.0 1.0 0.0 1
2 3 21
1 10 20 30
2 10 30 40
"""
    dataset = Mike21Driver().read(_write(tmp_path, text))

    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_mike21_quad_padding_zero_becomes_fill(tmp_path):
    """A triangle padded with a trailing ``0`` in a quad mesh gets ``-1`` fill."""
    text = """100079 1000 4 LONG/LAT
1 0.0 0.0 0.0 1
2 1.0 0.0 0.0 1
3 1.0 1.0 0.0 1
4 0.0 1.0 0.0 1
2 4 21
1 1 2 3 4
2 1 3 4 0
"""
    dataset = Mike21Driver().read(_write(tmp_path, text))

    rows = dataset.mesh.face_node_connectivity.data.tolist()
    assert rows[0] == [0, 1, 2, 3]
    assert rows[1] == [0, 2, 3, -1]


def test_mike21_unknown_projection_leaves_crs_none(tmp_path):
    """A DHI-internal projection name carries no datum, so the CRS is left unset."""
    dataset = Mike21Driver().read(_write(tmp_path, _UNIT_SQUARE))

    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


def test_mike21_epsg_projection_is_carried(tmp_path):
    """A numeric projection the PROJ database validates is carried as an EPSG code."""
    text = _UNIT_SQUARE.replace("LONG/LAT", "32632")
    dataset = Mike21Driver().read(_write(tmp_path, text))

    assert dataset.mesh.crs == 32632


def test_mike21_unknown_epsg_projection_leaves_crs_none(tmp_path):
    """A numeric projection PROJ does not recognise is refused, not carried blindly."""
    text = _UNIT_SQUARE.replace("LONG/LAT", "999999")
    dataset = Mike21Driver().read(_write(tmp_path, text))

    assert dataset.mesh.crs is None


def test_mike21_wkt_projection_is_carried(tmp_path):
    """A real WKT projection string is validated and carried."""
    from osgeo import osr

    reference = osr.SpatialReference()
    reference.ImportFromEPSG(32632)
    wkt = reference.ExportToWkt()
    text = _UNIT_SQUARE.replace("LONG/LAT", wkt)
    dataset = Mike21Driver().read(_write(tmp_path, text))

    assert dataset.crs_wkt is not None
    assert "UTM" in dataset.crs_wkt


def test_mike21_garbage_wkt_projection_leaves_crs_none(tmp_path):
    """A WKT-looking string that does not validate is left as None, not carried."""
    text = _UNIT_SQUARE.replace("LONG/LAT", 'PROJCS["not a real projection"]')
    dataset = Mike21Driver().read(_write(tmp_path, text))

    assert dataset.crs_wkt is None


def test_mike21_boundary_code_is_not_carried(tmp_path):
    """The node boundary code is read for width but not turned into a variable."""
    dataset = Mike21Driver().read(_write(tmp_path, _UNIT_SQUARE))

    assert dataset.data_variable_names == ["node_z"]


def test_mike21_clockwise_face_is_reversed(tmp_path):
    """A clockwise element is wound counter-clockwise by the base validator."""
    text = """100079 1000 3 LONG/LAT
1 0.0 0.0 0.0 1
2 1.0 1.0 0.0 1
3 1.0 0.0 0.0 1
1 3 21
1 1 2 3
"""
    dataset = Mike21Driver().read(_write(tmp_path, text))

    row = dataset.mesh.face_node_connectivity.data[0]
    vertices = row[row != -1]
    xs = dataset.mesh.node_x[vertices]
    ys = dataset.mesh.node_y[vertices]
    signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
    assert signed_area2 > 0


def test_mike21_empty_file_is_refused(tmp_path):
    """An empty file is refused rather than read as an empty mesh."""
    with pytest.raises(FileFormatNotSupportedError, match="is empty"):
        Mike21Driver().read(_write(tmp_path, ""))


def test_mike21_short_node_record_is_refused(tmp_path):
    """A node record missing its z and boundary columns is refused."""
    text = """100079 1000 1 LONG/LAT
1 0.0 0.0
2 3 21
1 1 2 3
"""
    with pytest.raises(FileFormatNotSupportedError, match="short node record"):
        Mike21Driver().read(_write(tmp_path, text))


def test_mike21_truncated_before_elements_is_refused(tmp_path):
    """A file whose node count outruns its records is refused, naming what is missing."""
    text = """100079 1000 9 LONG/LAT
1 0.0 0.0 0.0 1
2 1.0 0.0 0.0 1
"""
    with pytest.raises(FileFormatNotSupportedError, match="ends before"):
        Mike21Driver().read(_write(tmp_path, text))


def test_mike21_non_numeric_coordinate_names_the_record(tmp_path):
    """A non-numeric coordinate is refused with the record and file named."""
    text = _UNIT_SQUARE.replace("1 0.0 0.0 0.0 1", "1 east 0.0 0.0 1")
    with pytest.raises(FileFormatNotSupportedError, match="malformed node record"):
        Mike21Driver().read(_write(tmp_path, text))


def test_mike21_undefined_node_reference_is_refused(tmp_path):
    """An element referencing a node id no record defines is refused."""
    text = _UNIT_SQUARE.replace("1 1 2 3", "1 1 2 99")
    with pytest.raises(ValueError, match="node id 99"):
        Mike21Driver().read(_write(tmp_path, text))


def test_mike21_can_open_false_on_junk(tmp_path):
    """``can_open`` declines empty, foreign and directory paths without raising."""
    driver = Mike21Driver()
    empty = tmp_path / "empty.mesh"
    empty.write_bytes(b"")
    foreign = tmp_path / "other.mesh"
    foreign.write_text("not a header line at all\n", encoding="utf-8")
    directory = tmp_path / "folder"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(foreign)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "absent.mesh")) is False


def test_mike21_can_open_true_on_a_real_header(tmp_path):
    """A file whose first line parses as a header is recognised."""
    assert Mike21Driver().can_open(_write(tmp_path, _UNIT_SQUARE)) is True


def test_mike21_suffix_is_required(tmp_path):
    """Right content under a suffix the driver does not claim is still declined."""
    path = tmp_path / "domain.txt"
    path.write_text(_UNIT_SQUARE, encoding="utf-8")

    assert Mike21Driver().can_open(str(path)) is False


def test_mike21_committed_fixture_reads_and_matches_the_builder(tmp_path):
    """The committed fixture reads, and matches ``build_minimal_fixture_mike21``."""
    committed = Path(__file__).parent.parent / "data" / "minimal.mesh"
    dataset = Mike21Driver().read(str(committed))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2

    generated = tmp_path / "generated.mesh"
    build_minimal_fixture_mike21(str(generated))
    assert generated.read_text(encoding="utf-8") == committed.read_text(
        encoding="utf-8"
    )
