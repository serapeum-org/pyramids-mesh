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

from pyramids_mesh.drivers.river_flood.riverflow2d import RiverFlow2dDriver

_DATA = Path(__file__).parent.parent.parent / "data" / "riverflow2d"
_MESH = _DATA / "minimal.fed"
# The docs-facing example sample: a realistic 80-node / 120-triangle channel with two
# Manning zones, an upstream pool and a mixed sentinel/real erodible floor.
_CHANNEL = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "data"
    / "riverflow2d"
    / "channel.fed"
)


def _read(path=_MESH):
    """Read the committed RiverFlow2D mesh."""
    return RiverFlow2dDriver().read(str(path))


def test_reads_nodes_and_triangles():
    """The .FED builds a 5-node, 4-triangle mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 5
    assert mesh.n_face == 4
    assert mesh.face_node_connectivity.data.shape == (4, 3)


def test_reads_the_realistic_channel_example():
    """The committed example channel.fed reads with its known shape and face values.

    Test scenario:
        The docs-facing 80-node / 120-triangle sample should parse to that size with the
        connectivity re-based to 0..79, two Manning zones, and ``min_erode_elev`` carrying
        the -9999 sentinel on its floodplain faces. Guards the example asset against the
        reader or the committed data drifting out of sync.
    """
    ds = _read(_CHANNEL)
    conn = ds.mesh.face_node_connectivity.data
    assert ds.mesh.n_node == 80, f"expected 80 nodes, got {ds.mesh.n_node}"
    assert ds.mesh.n_face == 120, f"expected 120 faces, got {ds.mesh.n_face}"
    assert (int(conn.min()), int(conn.max())) == (0, 79), "connectivity not 0..79"
    manning = ds.get("manning_n").data
    assert set(np.round(manning, 3)) == {0.03, 0.075}, "expected two Manning zones"
    min_erode = ds.get("min_erode_elev")
    assert min_erode.nodata == -9999.0, "min_erode_elev nodata should be -9999"
    sentinel = int((min_erode.data == -9999.0).sum())
    assert sentinel == 60, f"expected 60 sentinel faces, got {sentinel}"
    assert int((ds.get("init_wse").data > 0).sum()) == 32, "expected 32 ponded faces"


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


# -- parse-level error branches ---------------------------------------------------------
# can_open/_sniff short-circuits before _parse in normal dispatch, so these drive read()
# directly to exercise the _FedReader raise paths. A minimal valid 3-node, 1-triangle
# file whose blocks are mutated one at a time.

_VALID = (
    "1 3 5 5\n"
    "1 0.0 0.0 1.0 0.0 -9999.0 0 0\n"
    "2 1.0 0.0 1.0 0.0 -9999.0 0 0\n"
    "3 0.0 1.0 1.0 0.0 -9999.0 0 0\n"
    "1 1 2 3 0.03 1.0 0.0 -9999.0 0.0\n"
)


def _write(tmp_path, text):
    """Write ``text`` to a ``.fed`` under ``tmp_path`` and return its path string."""
    path = tmp_path / "f.fed"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_read_valid_three_node_file(tmp_path):
    """The minimal 3-node, 1-triangle template reads as a one-face mesh.

    Test scenario:
        Confirms the template the error-path tests mutate is itself valid, so each later
        failure is attributable to the single field it changes.
    """
    ds = RiverFlow2dDriver().read(_write(tmp_path, _VALID))
    assert ds.mesh.n_face == 1, f"expected 1 face, got {ds.mesh.n_face}"


def test_read_rejects_short_header(tmp_path):
    """A header with fewer than two integers is a format error.

    Test scenario:
        A single-token header line cannot give NELEM and NNODES, so ``_header`` refuses it.
    """
    path = _write(tmp_path, "4\n1 0.0 0.0 1.0 0 -9999 0 0\n")
    with pytest.raises(FileFormatNotSupportedError, match="short header"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_non_integer_header(tmp_path):
    """A header whose first two fields are not integers is a format error.

    Test scenario:
        ``NELEM``/``NNODES`` must parse as ints; alphabetic counts are refused.
    """
    path = _write(tmp_path, "a b 5 5\n1 0.0 0.0 1.0 0 -9999 0 0\n")
    with pytest.raises(FileFormatNotSupportedError, match="not two integers"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_too_few_lines(tmp_path):
    """A header promising more nodes/elements than the file holds is a format error.

    Test scenario:
        Header declares 3 nodes and 1 element but only one node line follows, so the
        declared counts cannot be satisfied.
    """
    text = "1 3 5 5\n1 0.0 0.0 1.0 0.0 -9999.0 0 0\n"
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="header declares"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_zero_elements(tmp_path):
    """A header declaring zero elements is a format error (no mesh).

    Test scenario:
        ``NELEM`` of 0 with its node block present carries no faces, so it is refused.
    """
    text = (
        "0 3 5 5\n"
        "1 0.0 0.0 1.0 0.0 -9999.0 0 0\n"
        "2 1.0 0.0 1.0 0.0 -9999.0 0 0\n"
        "3 0.0 1.0 1.0 0.0 -9999.0 0 0\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="no elements"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_non_positive_node_count(tmp_path):
    """A header declaring a non-positive NNODES is a format error, not a slicing crash.

    Test scenario:
        A negative node count would otherwise drive negative-index block slicing and a
        misleading downstream "outside 1..-3" error; it is rejected up front, symmetric
        with the zero-element guard.
    """
    path = _write(tmp_path, "1 -3 5 5\n1 0.0 0.0 1.0 0 -9999 0 0\n")
    with pytest.raises(FileFormatNotSupportedError, match="non-positive node count"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_short_node_line(tmp_path):
    """A node line with fewer than four fields is a format error.

    Test scenario:
        ``IN X Y ZB`` are the minimum a node line must carry; a three-token line is short.
    """
    text = _VALID.replace("1 0.0 0.0 1.0 0.0 -9999.0 0 0\n", "1 0.0 0.0\n")
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="short node line"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_non_numeric_node_field(tmp_path):
    """A non-numeric node coordinate is a format error.

    Test scenario:
        The X column must parse as a float; a word in its place is refused.
    """
    text = _VALID.replace(
        "1 0.0 0.0 1.0 0.0 -9999.0 0 0\n", "1 x 0.0 1.0 0.0 -9999.0 0 0\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="non-numeric field"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_short_element_line(tmp_path):
    """An element line with fewer than eight fields is a format error.

    Test scenario:
        An element needs its id, three node ids and the four named float columns; a
        truncated line is refused.
    """
    text = _VALID.replace("1 1 2 3 0.03 1.0 0.0 -9999.0 0.0\n", "1 1 2 3\n")
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="short element line"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_non_numeric_element_field(tmp_path):
    """A non-numeric element field (Manning's n) is a format error.

    Test scenario:
        ``MANNINGN`` must parse as a float; a word in its place is refused, naming the
        element record.
    """
    text = _VALID.replace(
        "1 1 2 3 0.03 1.0 0.0 -9999.0 0.0\n", "1 1 2 3 rough 1.0 0.0 -9999.0 0.0\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(
        FileFormatNotSupportedError, match="element line with a non-numeric"
    ):
        RiverFlow2dDriver().read(path)


def test_read_rejects_non_integer_element_vertex(tmp_path):
    """A non-integer element vertex token is a FileFormatNotSupportedError.

    Test scenario:
        A garbage vertex token is a syntactic corruption, refused the same way as a
        non-numeric coordinate -- distinct from a well-formed id that is merely out of
        range (which is a referential ValueError).
    """
    text = _VALID.replace(
        "1 1 2 3 0.03 1.0 0.0 -9999.0 0.0\n", "1 1 2 x 0.03 1.0 0.0 -9999.0 0.0\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="not an integer"):
        RiverFlow2dDriver().read(path)


def test_read_rejects_degenerate_face(tmp_path):
    """An element that repeats a vertex id is refused as a degenerate face.

    Test scenario:
        A triangle whose three node ids are not distinct (e.g. ``1 1 2``) has zero area;
        it passes the per-vertex range check but is rejected as malformed topology.
    """
    text = _VALID.replace(
        "1 1 2 3 0.03 1.0 0.0 -9999.0 0.0\n", "1 1 1 2 0.03 1.0 0.0 -9999.0 0.0\n"
    )
    path = _write(tmp_path, text)
    with pytest.raises(FileFormatNotSupportedError, match="repeats a vertex"):
        RiverFlow2dDriver().read(path)


def test_sniff_declines_single_line_file(tmp_path):
    """can_open declines a one-line .fed (a header with no node block).

    Test scenario:
        The sniff needs a header and a first node line; a file with only one significant
        line cannot be a mesh.
    """
    assert RiverFlow2dDriver().can_open(_write(tmp_path, "4 5 5 5\n")) is False, (
        "a single-line .fed should be declined"
    )


def test_sniff_declines_non_numeric_node_coordinate(tmp_path):
    """can_open declines a .fed whose first node line has a non-float coordinate.

    Test scenario:
        Header and node id are fine, but the X/Y/ZB triple must be floats; a word there
        fails the sniff.
    """
    text = "4 5 5 5\n1 a b c 0.0 -9999.0 0 0\n"
    assert RiverFlow2dDriver().can_open(_write(tmp_path, text)) is False, (
        "a non-numeric node coordinate should be declined"
    )


def test_sniff_tolerates_leading_blank_line(tmp_path):
    """can_open accepts a valid .fed whose first line(s) are blank.

    Test scenario:
        The sniff skips blank lines when locating the header and first node line, so a
        leading blank line does not make a valid file look malformed.
    """
    text = "\n4 5 5 5\n1 0.0 0.0 1.0 0.0 -9999.0 0 0\n"
    assert RiverFlow2dDriver().can_open(_write(tmp_path, text)) is True, (
        "a leading blank line should not break the sniff"
    )


def test_read_tolerates_blank_lines(tmp_path):
    """Blank lines between records are skipped, not counted as nodes or elements.

    Test scenario:
        A valid file with blank lines inserted between the header and blocks still reads
        as the same one-face mesh — only non-blank lines are significant.
    """
    text = "\n" + _VALID.replace("\n", "\n\n")
    ds = RiverFlow2dDriver().read(_write(tmp_path, text))
    assert ds.mesh.n_face == 1, (
        f"blank lines should be ignored, got {ds.mesh.n_face} faces"
    )


def test_carries_nonsentinel_face_values(tmp_path):
    """Non-sentinel ELINITWSE / ELMINERODELEV are carried onto the face variables.

    Test scenario:
        An element with a ponded initial WSE and a real erodible floor should surface
        those exact values on ``init_wse`` and ``min_erode_elev`` (the minimal committed
        fixture is all-sentinel, so this exercises the value-carrying path).
    """
    text = (
        "1 3 5 5\n"
        "1 0.0 0.0 1.0 0.0 -9999.0 0 0\n"
        "2 1.0 0.0 1.0 0.0 -9999.0 0 0\n"
        "3 0.0 1.0 1.0 0.0 -9999.0 0 0\n"
        "1 1 2 3 0.03 1.0 5.5 0.25 0.0\n"
    )
    ds = RiverFlow2dDriver().read(_write(tmp_path, text))
    assert float(ds.get("init_wse").data[0]) == pytest.approx(5.5), (
        "ELINITWSE not carried"
    )
    assert float(ds.get("min_erode_elev").data[0]) == pytest.approx(0.25), (
        "ELMINERODELEV not carried"
    )
