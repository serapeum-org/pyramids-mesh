"""Tests for the Triangle / TetGen ``.node`` + ``.ele`` reader (C9).

Triangle (2-D) and TetGen (3-D) share an ASCII mesh split across a ``.node`` vertex file
and an ``.ele`` element file. A genuine 3-D (TetGen) pair is committed under
``tests/data/triangle/`` -- ``mesh.node`` (89 vertices, ``dim=3``) and ``mesh.ele`` (304
tetrahedra, 4 nodes each) -- and the tests read the committed bytes. The index base is
0 or 1 depending on the tool, so a small 1-based pair is synthesised to prove detection.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset
from pyramids_mesh.drivers.triangle import TriangleDriver

_DATA = Path(__file__).parent.parent / "data" / "triangle"
_NODE = _DATA / "mesh.node"
_ELE = _DATA / "mesh.ele"


def _read(path):
    """Read a committed pair, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TriangleDriver().read(str(path))


def _signed_area_2(xs, ys):
    """Twice the signed polygon area (positive counter-clockwise)."""
    return float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def test_reads_nodes_and_faces():
    """mesh.node/mesh.ele read 89 vertices and 304 four-node elements."""
    mesh = _read(_NODE).mesh
    assert mesh.n_node == 89
    assert mesh.n_face == 304
    assert mesh.face_node_connectivity.data.shape == (304, 4)


def test_index_base_zero_is_detected_from_the_committed_pair():
    """The committed pair is 0-based: the base is detected and recorded as 0."""
    connectivity = _read(_NODE).mesh.face_node_connectivity
    assert connectivity.original_start_index == 0
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 88


def test_index_base_one_is_detected_from_the_first_vertex_id(tmp_path):
    """A 1-based pair is detected from its first vertex id and re-based to 0."""
    (tmp_path / "tri.node").write_text("3 2 0 0\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n")
    (tmp_path / "tri.ele").write_text("1 3 0\n1 1 2 3\n")
    mesh = _read(tmp_path / "tri.node").mesh
    assert mesh.n_node == 3
    assert mesh.n_face == 1
    connectivity = mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert connectivity.data.tolist() == [[0, 1, 2]]


def test_node_z_comes_from_the_third_coordinate():
    """A 3-D (TetGen) pair carries elevation as a node-located node_z variable."""
    node_z = _read(_NODE).get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (89,)
    # The first committed vertex has z = 6.9999998807907104e-01.
    assert float(node_z.data[0]) == pytest.approx(0.7, abs=1e-6)


def test_faces_are_counter_clockwise_after_validation():
    """Every element's node list comes back non-clockwise on the xy projection."""
    mesh = _read(_NODE).mesh
    for row in mesh.face_node_connectivity.data:
        area2 = _signed_area_2(mesh.node_x[row], mesh.node_y[row])
        # A clockwise face is reversed; a degenerate projection is left alone within
        # tolerance -- neither leaves a face more than negligibly clockwise.
        assert area2 >= -1e-9


def test_read_dispatches_from_either_file_of_the_pair():
    """read() reaches the same mesh whether pointed at the .node or the .ele file."""
    from_node = _read(_NODE)
    from_ele = _read(_ELE)
    assert isinstance(from_node, UgridDataset)
    assert isinstance(from_ele, UgridDataset)
    assert from_node.mesh.n_node == from_ele.mesh.n_node == 89
    assert from_node.mesh.n_face == from_ele.mesh.n_face == 304


def test_can_open_accepts_either_file_of_a_pair():
    """can_open accepts the .node and the .ele file when both are present."""
    driver = TriangleDriver()
    assert driver.can_open(str(_NODE)) is True
    assert driver.can_open(str(_ELE)) is True


def test_can_open_declines_a_stray_file(tmp_path):
    """can_open declines a lone .node with no sibling, and a foreign file."""
    driver = TriangleDriver()
    solo = tmp_path / "solo.node"
    solo.write_text("3 2 0 0\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n")
    assert driver.can_open(str(solo)) is False  # no sibling .ele
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False


def test_read_of_a_non_pair_path_is_a_format_error(tmp_path):
    """read() on a lone .node with no .ele sibling is a clean format error."""
    solo = tmp_path / "solo.node"
    solo.write_text("3 2 0 0\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n")
    with pytest.raises(FileFormatNotSupportedError, match="Triangle/TetGen"):
        TriangleDriver().read(str(solo))


def _pair(tmp_path, node_text, ele_text):
    """Write a .node/.ele pair in tmp_path and return the .node path."""
    (tmp_path / "m.node").write_text(node_text)
    (tmp_path / "m.ele").write_text(ele_text)
    return str(tmp_path / "m.node")


def test_can_open_swallows_a_sniff_that_raises(tmp_path):
    """can_open declines and warns when _sniff raises, never letting it escape."""
    node = _pair(
        tmp_path, "3 2 0 0\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n", "1 3 0\n1 1 2 3\n"
    )

    class _BrokenSniff(TriangleDriver):
        """A driver whose header sniff always raises, to exercise the guard."""

        def _sniff(self, path):
            """Raise so can_open's never-raise guard is the thing under test."""
            raise RuntimeError("header parser is wrong")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = _BrokenSniff().can_open(node)

    assert verdict is False
    assert any("treated as declining it" in str(w.message) for w in caught)


def test_can_open_declines_a_single_field_header(tmp_path):
    """can_open declines a pair whose header has fewer than two fields."""
    node = _pair(tmp_path, "3\n1 0.0 0.0\n", "1 3 0\n1 1 2 3\n")
    assert TriangleDriver().can_open(node) is False


def test_can_open_declines_a_missing_path(tmp_path):
    """can_open declines a path that is not a file, pairing nothing."""
    assert TriangleDriver().can_open(str(tmp_path / "gone.node")) is False


def test_read_rejects_a_non_integer_node_header(tmp_path):
    """A .node header whose first token is not an integer is a format error."""
    node = _pair(tmp_path, "foo 2 0 0\n1 0.0 0.0\n", "1 3 0\n1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="header"):
        TriangleDriver().read(node)


def test_read_rejects_a_node_file_with_no_header(tmp_path):
    """An empty .node file reports that it has no header line."""
    node = _pair(tmp_path, "", "1 3 0\n1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="no header line"):
        TriangleDriver().read(node)


def test_read_rejects_an_unsupported_dimension(tmp_path):
    """A .node header declaring dimension 5 is refused: only 2 or 3 is supported."""
    node = _pair(tmp_path, "3 5 0 0\n1 0.0 0.0 0.0\n", "1 3 0\n1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="only 2 or 3 is supported"):
        TriangleDriver().read(node)


def test_read_rejects_too_few_vertex_lines(tmp_path):
    """A .node file declaring more vertices than it holds is a format error."""
    node = _pair(tmp_path, "3 2 0 0\n1 0.0 0.0\n", "1 3 0\n1 1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="declares 3 vertices"):
        TriangleDriver().read(node)


def test_read_rejects_a_non_integer_vertex_id(tmp_path):
    """A .node file whose first vertex id is not an integer is a format error."""
    node = _pair(
        tmp_path, "3 2 0 0\nX 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n", "1 3 0\n1 1 2 3\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="non-integer vertex id"):
        TriangleDriver().read(node)


def test_read_rejects_an_ele_file_with_no_header(tmp_path):
    """An empty .ele file reports that it has no element header."""
    node = _pair(tmp_path, "3 2 0 0\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n", "")
    with pytest.raises(FileFormatNotSupportedError, match="no '<#elts>"):
        TriangleDriver().read(node)


def test_read_rejects_too_few_element_lines(tmp_path):
    """An .ele file declaring more elements than it holds is a format error."""
    node = _pair(
        tmp_path, "3 2 0 0\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n", "2 3 0\n1 1 2 3\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="declares 2 elements"):
        TriangleDriver().read(node)
