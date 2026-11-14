"""Tests for the SWAN Easymesh ``.n`` + ``.e`` reader (C32).

Easymesh writes a triangulation across sibling ASCII files that share a stem; the driver
reads the ``.n`` nodes and ``.e`` elements. A genuine pair produced by EasyMesh from an
original unit-square domain is committed under ``tests/data/easymesh/`` -- ``square.n``
(12 nodes) and ``square.e`` (14 triangles) -- and the tests read the committed bytes,
trailer comment lines and all. Malformed and foreign inputs are synthesised in ``tmp_path``.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MeshDataset

from pyramids_mesh.drivers.easymesh import EasymeshDriver

_DATA = Path(__file__).parent.parent / "data" / "easymesh"
_NODE = _DATA / "square.n"
_ELEMENT = _DATA / "square.e"


def _read(path):
    """Read a committed pair, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EasymeshDriver().read(str(path))


def _signed_area_2(xs, ys):
    """Twice the signed polygon area (positive counter-clockwise)."""
    return float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def _pair(tmp_path, node_text, element_text):
    """Write a ``.n``/``.e`` pair in tmp_path and return the ``.n`` path."""
    (tmp_path / "m.n").write_text(node_text)
    (tmp_path / "m.e").write_text(element_text)
    return str(tmp_path / "m.n")


# A minimal single-triangle pair in the exact Easymesh layout (colon ids, trailer lines).
_ONE_N = (
    "3\n"
    "   0:  0.0 0.0  1\n"
    "   1:  1.0 0.0  1\n"
    "   2:  0.0 1.0  1\n"
    "-----------------\n"
    "   n:  x    y   mark\n"
)
_ONE_E = (
    "1\n"
    "   0:  0 1 2  -1 -1 -1  0 1 2  0.5 0.5  -1\n"
    "----------------------------------------\n"
    "   e:  i, j, k, ...\n"
)


def test_reads_nodes_and_triangular_faces():
    """square.n/square.e read 12 nodes and 14 three-node elements."""
    mesh = _read(_NODE).mesh
    assert mesh.n_node == 12
    assert mesh.n_face == 14
    assert mesh.face_node_connectivity.data.shape == (14, 3)


def test_index_base_zero_is_detected_from_the_committed_pair():
    """The committed pair is 0-based: the base is detected and recorded as 0."""
    connectivity = _read(_NODE).mesh.face_node_connectivity
    assert connectivity.original_start_index == 0
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 11


def test_node_coordinates_are_read_from_scientific_notation():
    """Coordinates written in scientific notation are parsed; node 0 is the origin."""
    mesh = _read(_NODE).mesh
    assert float(mesh.node_x[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(mesh.node_y[0]) == pytest.approx(0.0, abs=1e-12)
    # Node 1 of square.n is the midpoint of the bottom edge at x = 0.5.
    assert float(mesh.node_x[1]) == pytest.approx(0.5, abs=1e-12)


def test_committed_face_connectivity_matches_square_e():
    """The parsed faces carry exactly the node triples square.e declares (first 3 cols).

    Compared as sorted sets so the check is independent of winding normalisation (which may
    reverse a face's node order) and of face ordering.
    """
    expected = [
        {8, 0, 1},
        {9, 1, 2},
        {10, 11, 8},
        {10, 3, 4},
        {10, 8, 9},
        {11, 5, 6},
        {8, 11, 7},
        {8, 7, 0},
        {9, 2, 3},
        {8, 1, 9},
        {10, 4, 5},
        {9, 3, 10},
        {11, 6, 7},
        {10, 5, 11},
    ]
    data = _read(_NODE).mesh.face_node_connectivity.data
    got = sorted(sorted(row) for row in data.tolist())
    assert got == sorted(sorted(face) for face in expected)


def test_faces_are_counter_clockwise_after_validation():
    """Every triangle comes back non-clockwise on the xy projection."""
    mesh = _read(_NODE).mesh
    for row in mesh.face_node_connectivity.data:
        area2 = _signed_area_2(mesh.node_x[row], mesh.node_y[row])
        assert area2 >= -1e-9


def test_read_dispatches_from_either_file_of_the_pair():
    """read() reaches the same mesh whether pointed at the .n or the .e file."""
    from_node = _read(_NODE)
    from_element = _read(_ELEMENT)
    assert isinstance(from_node, MeshDataset)
    assert isinstance(from_element, MeshDataset)
    assert from_node.mesh.n_node == from_element.mesh.n_node == 12
    assert from_node.mesh.n_face == from_element.mesh.n_face == 14


def test_trailer_comment_lines_are_ignored():
    """A fixture that really ends with Easymesh's trailer lines still reads to its counts."""
    node_raw = _NODE.read_text()
    element_raw = _ELEMENT.read_text()
    assert "----" in node_raw and "mark" in node_raw
    assert "----" in element_raw
    mesh = _read(_NODE).mesh
    assert mesh.n_node == 12
    assert mesh.n_face == 14


def test_a_synthesised_single_triangle_pair_reads(tmp_path):
    """A minimal one-triangle pair in the Easymesh layout reads to one face."""
    node = _pair(tmp_path, _ONE_N, _ONE_E)
    mesh = _read(node).mesh
    assert mesh.n_node == 3
    assert mesh.n_face == 1
    assert sorted(mesh.face_node_connectivity.data[0].tolist()) == [0, 1, 2]


def test_can_open_accepts_either_file_of_a_pair():
    """can_open accepts the .n and the .e file when both are present."""
    driver = EasymeshDriver()
    assert driver.can_open(str(_NODE)) is True
    assert driver.can_open(str(_ELEMENT)) is True


def test_dot_e_dispatches_to_easymesh_not_exodus():
    """The .e suffix is shared with exodus, but content/sibling sniffing resolves it here.

    Exodus `.e` is binary NetCDF (its sniff declines an ASCII file) and this driver needs a
    sibling `.n`, so a real EasyMesh `.e` resolves to easymesh and an Exodus `.e` does not.
    """
    from pyramids.netcdf.ugrid import MeshDataset, mesh_drivers

    mesh = MeshDataset.read_file(str(_ELEMENT)).mesh
    assert mesh.n_node == 12
    assert mesh.n_face == 14
    drivers = mesh_drivers()
    assert drivers["easymesh"].can_open(str(_ELEMENT)) is True
    exodus = drivers.get("exodus")
    if exodus is not None:
        assert exodus.can_open(str(_ELEMENT)) is False


def test_can_open_declines_a_stray_file(tmp_path):
    """can_open declines a lone .n with no sibling, and a foreign file."""
    driver = EasymeshDriver()
    solo = tmp_path / "solo.n"
    solo.write_text("1\n   0:  0.0 0.0  1\n")
    assert driver.can_open(str(solo)) is False  # no sibling .e
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False


def test_can_open_declines_a_triangle_style_header(tmp_path):
    """A .node-style header (count plus dim/attr fields) is not sniffed as Easymesh."""
    node = _pair(tmp_path, "3 2 0 0\n1 0.0 0.0\n", "1 3 0\n1 1 2 3\n")
    assert EasymeshDriver().can_open(node) is False


def test_can_open_declines_without_a_colon_tagged_id(tmp_path):
    """A lone-integer header whose records are not colon-tagged is declined."""
    node = _pair(tmp_path, "1\n0 0.0 0.0 1\n", "1\n0 0 1 2\n")
    assert EasymeshDriver().can_open(node) is False


def test_can_open_declines_a_file_with_fewer_than_two_records(tmp_path):
    """can_open declines a paired file holding only its count line (no records yet)."""
    node = _pair(tmp_path, "1\n", "1\n   0:  0 1 2  -1 -1 -1  0 1 2  0.5 0.5  -1\n")
    assert EasymeshDriver().can_open(node) is False


def test_can_open_declines_a_non_integer_count(tmp_path):
    """can_open declines a paired file whose first line is not a bare integer count."""
    node = _pair(tmp_path, "foo\n   0:  0.0 0.0  1\n", _ONE_E)
    assert EasymeshDriver().can_open(node) is False


def test_can_open_swallows_a_sniff_that_raises(tmp_path):
    """can_open declines and warns when _sniff raises, never letting it escape."""
    node = _pair(tmp_path, _ONE_N, _ONE_E)

    class _BrokenSniff(EasymeshDriver):
        """A driver whose sniff always raises, to exercise the never-raise guard."""

        def _sniff(self, path):
            """Raise so can_open's guard is the thing under test."""
            raise RuntimeError("sniff is wrong")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = _BrokenSniff().can_open(node)

    assert verdict is False
    assert any("treated as declining it" in str(w.message) for w in caught)


def test_can_open_declines_a_missing_path(tmp_path):
    """can_open declines a path that is not a file, pairing nothing."""
    assert EasymeshDriver().can_open(str(tmp_path / "gone.n")) is False


def test_read_of_a_non_pair_path_is_a_format_error(tmp_path):
    """read() on a lone .n with no .e sibling is a clean format error."""
    solo = tmp_path / "solo.n"
    solo.write_text(_ONE_N)
    with pytest.raises(FileFormatNotSupportedError, match="Easymesh"):
        EasymeshDriver().read(str(solo))


def test_read_rejects_a_node_file_with_no_header(tmp_path):
    """An empty .n file reports that it has no node-count header."""
    node = _pair(tmp_path, "", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="number of nodes"):
        EasymeshDriver().read(node)


def test_read_rejects_a_non_integer_node_count(tmp_path):
    """A .n file whose count line is not an integer is a format error."""
    node = _pair(tmp_path, "foo\n   0:  0.0 0.0  1\n", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="non-integer nodes count"):
        EasymeshDriver().read(node)


def test_read_rejects_a_header_with_extra_fields(tmp_path):
    """A .n header with fields beside the count is not a bare count."""
    node = _pair(tmp_path, "3 2\n   0:  0.0 0.0  1\n", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="not a bare nodes count"):
        EasymeshDriver().read(node)


def test_read_rejects_a_zero_node_count(tmp_path):
    """A .n declaring zero nodes is a clean format error, not an IndexError."""
    node = _pair(tmp_path, "0\n----\n   n:  x y mark\n", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="non-positive nodes count 0"):
        EasymeshDriver().read(node)


def test_read_rejects_a_negative_node_count(tmp_path):
    """A .n declaring a negative node count is a clean format error."""
    node = _pair(tmp_path, "-5\n   0:  0.0 0.0  1\n", _ONE_E)
    with pytest.raises(
        FileFormatNotSupportedError, match="non-positive nodes count -5"
    ):
        EasymeshDriver().read(node)


def test_read_rejects_a_zero_element_count(tmp_path):
    """An .e declaring zero elements is a clean format error, not an empty mesh."""
    (tmp_path / "m.n").write_text(_ONE_N)
    (tmp_path / "m.e").write_text("0\n----\n   e:  i j k\n")
    with pytest.raises(
        FileFormatNotSupportedError, match="non-positive elements count 0"
    ):
        EasymeshDriver().read(str(tmp_path / "m.n"))


def test_read_rejects_too_few_node_lines(tmp_path):
    """A .n file declaring more nodes than it holds is a format error."""
    node = _pair(tmp_path, "3\n   0:  0.0 0.0  1\n", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="declares 3 nodes"):
        EasymeshDriver().read(node)


def test_read_rejects_a_non_integer_node_id(tmp_path):
    """A .n file whose first node id is not an integer is a format error."""
    node = _pair(tmp_path, "2\n   X:  0.0 0.0  1\n   1:  1.0 0.0  1\n", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="non-integer node id"):
        EasymeshDriver().read(node)


def test_read_rejects_a_node_line_without_two_coordinates(tmp_path):
    """A .n node line missing a coordinate is a clean format error."""
    node = _pair(tmp_path, "1\n   0:  0.0\n", _ONE_E)
    with pytest.raises(FileFormatNotSupportedError, match="two numeric coordinates"):
        EasymeshDriver().read(node)


def test_read_rejects_too_few_element_lines(tmp_path):
    """An .e file declaring more elements than it holds is a format error."""
    element = "2\n   0:  0 1 2  -1 -1 -1  0 1 2  0.5 0.5  -1\n"
    (tmp_path / "m.n").write_text(_ONE_N)
    (tmp_path / "m.e").write_text(element)
    with pytest.raises(FileFormatNotSupportedError, match="declares 2 elements"):
        EasymeshDriver().read(str(tmp_path / "m.n"))


def test_read_rejects_a_non_integer_element_node_id(tmp_path):
    """An .e element line with a non-integer node id is a format error."""
    element = "1\n   0:  0 1 X\n"
    (tmp_path / "m.n").write_text(_ONE_N)
    (tmp_path / "m.e").write_text(element)
    with pytest.raises(FileFormatNotSupportedError, match="three integer node ids"):
        EasymeshDriver().read(str(tmp_path / "m.n"))


def test_read_rejects_two_field_element_rows(tmp_path):
    """Uniformly two-id element rows are declined, not read as an (n, 2) face table."""
    (tmp_path / "m.n").write_text(_ONE_N)
    (tmp_path / "m.e").write_text("1\n   0:  0 1\n")
    with pytest.raises(FileFormatNotSupportedError, match="fewer than three node ids"):
        EasymeshDriver().read(str(tmp_path / "m.n"))
