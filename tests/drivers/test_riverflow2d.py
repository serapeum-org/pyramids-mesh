"""Tests for the RiverFlow2D ``.FED`` reader (C29, issue #67).

The committed fixture ``tests/data/riverflow2d/minimal.fed`` is an authored 5-node,
4-triangle mesh (a square split into a fan about a centre node). It is hand-written from
the published ``.FED`` layout and was cross-checked to be structurally identical to the
real meshes in Hydronia's ExampleProjects package -- the same validator passes on both --
so it carries no vendor data while exercising every column: 1-based ids, counter-clockwise
triangles, the ``-9999`` sentinel, per-node ``ZB`` and per-element ``MANNINGN``.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest
from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.riverflow2d import RiverFlow2dDriver

_DATA = Path(__file__).parent.parent / "data" / "riverflow2d"
_MESH = _DATA / "minimal.fed"


def _read(path=_MESH):
    """Read the committed RiverFlow2D mesh."""
    return RiverFlow2dDriver().read(str(path))


def test_reads_nodes_and_triangles():
    """The .FED builds a 5-node, 4-triangle mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 5
    assert mesh.n_face == 4
    assert mesh.face_node_connectivity.data.shape == (4, 3)


def test_connectivity_is_one_based_resolved_to_zero_based():
    """The 1-based element table arrives 0-based, with the base recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 4


def test_node_coordinates_match_the_mesh():
    """Node 1's x/y match the committed .FED (projected metres)."""
    mesh = _read().mesh
    assert float(mesh.node_x[0]) == pytest.approx(500000.0)
    assert float(mesh.node_y[0]) == pytest.approx(4500000.0)


def test_zb_becomes_node_z():
    """The ZB column becomes a node-located node_z variable."""
    node_z = _read().get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (5,)
    assert float(node_z.data[0]) == pytest.approx(10.0)
    assert float(node_z.data[4]) == pytest.approx(9.5)  # the centre node


def test_manning_is_face_located():
    """MANNINGN becomes a face-located manning_n variable."""
    manning = _read().get("manning_n")
    assert manning.location == "face"
    assert manning.shape == (4,)
    assert np.allclose(manning.data, 0.035)


def test_init_wse_is_face_located():
    """ELINITWSE becomes a face-located init_wse variable."""
    init_wse = _read().get("init_wse")
    assert init_wse.location == "face"
    assert init_wse.shape == (4,)
    assert np.allclose(init_wse.data, 0.0)


def test_min_erode_elev_carries_the_sentinel_as_nodata():
    """ELMINERODELEV is face-located and records the -9999 sentinel as nodata."""
    min_erode = _read().get("min_erode_elev")
    assert min_erode.location == "face"
    assert min_erode.nodata == -9999.0
    assert np.allclose(min_erode.data, -9999.0)


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise after validation."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        xs = mesh.node_x[row]
        ys = mesh.node_y[row]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_declares_read_mesh():
    """The driver advertises read_mesh."""
    assert set(RiverFlow2dDriver.capabilities) == {"read_mesh"}


def test_can_open_accepts_the_fed_file():
    """can_open accepts the committed .fed file."""
    assert RiverFlow2dDriver().can_open(str(_MESH)) is True


def test_can_open_declines_non_fed(tmp_path):
    """can_open declines a wrong-suffix file, a directory, and a bad .fed."""
    driver = RiverFlow2dDriver()
    assert driver.can_open(str(tmp_path)) is False
    plain = tmp_path / "notes.txt"
    plain.write_text("hello")
    assert driver.can_open(str(plain)) is False
    bad = tmp_path / "bad.fed"
    bad.write_text("not a mesh\nat all\n", encoding="utf-8")
    assert driver.can_open(str(bad)) is False


@pytest.mark.parametrize(
    "content",
    [
        "abc def 5 5\n1 0.0 0.0 1.0 0 -9999 0 0\n",  # non-integer header counts
        "0 5 5 5\n1 0.0 0.0 1.0 0 -9999 0 0\n",  # non-positive element count
        "4 5 5 5\n1 2 3\n",  # first node line too short
        "4 5 5 5\nx y z w v 0 0\n",  # non-numeric node fields
    ],
)
def test_sniff_declines_malformed_fed(tmp_path, content):
    """can_open declines a .fed whose header or first node line is not FED-shaped."""
    bad = tmp_path / "f.fed"
    bad.write_text(content, encoding="utf-8")
    assert RiverFlow2dDriver().can_open(str(bad)) is False


def test_non_consecutive_node_ids_are_refused(tmp_path):
    """A gapped/renumbered node block is refused rather than silently mis-wired."""
    mesh = tmp_path / "f.fed"
    mesh.write_text(
        "1 3 5 5\n"
        "1 0.0 0.0 1.0 0 -9999 0 0\n"
        "2 1.0 0.0 1.0 0 -9999 0 0\n"
        "9 1.0 1.0 1.0 0 -9999 0 0\n"  # id 9, not 3
        "1 1 2 3 0.03 1.0 0.0 -9999 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="not consecutive"):
        RiverFlow2dDriver().read(str(mesh))


def test_element_referencing_unknown_node_is_refused(tmp_path):
    """An element vertex outside 1..NNODES is a ValueError."""
    mesh = tmp_path / "f.fed"
    mesh.write_text(
        "1 3 5 5\n"
        "1 0.0 0.0 1.0 0 -9999 0 0\n"
        "2 1.0 0.0 1.0 0 -9999 0 0\n"
        "3 1.0 1.0 1.0 0 -9999 0 0\n"
        "1 1 2 7 0.03 1.0 0.0 -9999 0.0\n",  # node 7 does not exist
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside 1..3"):
        RiverFlow2dDriver().read(str(mesh))


def test_read_of_a_non_fed_path_is_a_format_error(tmp_path):
    """read() on a file that is not a .FED is refused with a format error."""
    other = tmp_path / "empty.fed"
    other.write_text("", encoding="utf-8")
    with pytest.raises(FileFormatNotSupportedError):
        RiverFlow2dDriver().read(str(other))
