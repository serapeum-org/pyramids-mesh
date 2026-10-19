"""Tests for the OFF (Object File Format) surface-mesh reader.

OFF is a hand-parsed geometry format -- no time axis, elevation per node. The happy path
runs against a committed vendor sample (an 8-vertex / 6-quad cube carrying inline ``#``
comments and per-face RGBA colour), proving the parser skips comments and ignores the
trailing colour floats. The remaining cases are tiny inline snippets written into
``tmp_path`` for the defensive branches.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.off import OffDriver

_COMMENTS_OFF = Path(__file__).parent.parent / "data" / "off" / "comments.off"

# The cube's six quad faces, as vertex *sets* -- face winding is normalised on read, so a
# row may come back reversed, but the set of vertices per face is invariant.
_CUBE_FACE_SETS = {
    frozenset({0, 1, 2, 3}),
    frozenset({0, 3, 4, 7}),
    frozenset({0, 1, 4, 5}),
    frozenset({1, 2, 5, 6}),
    frozenset({2, 3, 6, 7}),
    frozenset({4, 5, 6, 7}),
}


def _write(tmp_path, text, name="surface.off"):
    """Write a text fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestOffSample:
    """The committed trimesh cube: comments skipped, colour ignored."""

    def test_reads_committed_cube(self):
        """The 8-vertex / 6-quad cube reads with its real counts."""
        dataset = OffDriver().read(str(_COMMENTS_OFF))

        assert isinstance(dataset, UgridDataset)
        assert dataset.mesh.n_node == 8
        assert dataset.mesh.n_face == 6

    def test_connectivity_ignores_trailing_colour(self):
        """Per-face RGBA colour is ignored; each quad keeps exactly its four ids."""
        dataset = OffDriver().read(str(_COMMENTS_OFF))

        rows = dataset.mesh.face_node_connectivity.data.tolist()
        assert len(rows) == 6
        assert all(len(row) == 4 for row in rows)
        assert {frozenset(row) for row in rows} == _CUBE_FACE_SETS

    def test_node_z_carries_the_elevation(self):
        """The vertex z becomes a node-located ``node_z`` variable of eight values."""
        dataset = OffDriver().read(str(_COMMENTS_OFF))

        node_z = dataset.get("node_z")
        assert node_z.location == "node"
        assert len(node_z.data.tolist()) == 8

    def test_crs_is_none(self):
        """OFF carries no georeferencing, so ``crs`` is never set."""
        dataset = OffDriver().read(str(_COMMENTS_OFF))

        assert dataset.mesh.crs is None


class TestOffInlineSnippets:
    """Tiny hand-written OFF snippets for the defensive branches."""

    def test_triangle_face_reads_zero_based(self, tmp_path):
        """A single counter-clockwise triangle reads 0-based and unchanged."""
        text = "OFF\n3 1 0\n0.0 0.0 2.0\n1.0 0.0 2.5\n0.0 1.0 3.0\n3 0 1 2\n"
        dataset = OffDriver().read(_write(tmp_path, text))

        assert dataset.mesh.n_node == 3
        assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2]]
        assert dataset.mesh.face_node_connectivity.original_start_index == 0

    def test_face_line_with_trailing_colour_is_ignored(self, tmp_path):
        """RGBA floats after a face's ids are read past, not folded into connectivity."""
        text = (
            "OFF\n"
            "3 1 0\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "3 0 1 2  1.000 0.000 0.000 0.75\n"
        )
        dataset = OffDriver().read(_write(tmp_path, text))

        assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2]]

    def test_inline_and_whole_line_comments_are_skipped(self, tmp_path):
        """A ``#`` comment -- whole-line or trailing a vertex -- is stripped."""
        text = (
            "OFF\n"
            "# a leading comment\n"
            "3 1 0\n"
            "0.0 0.0 0.0 # a weird vertex\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "3 0 1 2\n"
        )
        dataset = OffDriver().read(_write(tmp_path, text))

        assert dataset.mesh.n_node == 3
        assert dataset.mesh.n_face == 1

    def test_counts_appended_to_magic_line(self, tmp_path):
        """Counts may ride on the magic line (``OFF 3 1 0``) rather than the next line."""
        text = "OFF 3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n"
        dataset = OffDriver().read(_write(tmp_path, text))

        assert dataset.mesh.n_node == 3
        assert dataset.mesh.n_face == 1

    def test_missing_magic_is_refused(self, tmp_path):
        """A file whose first significant line is not ``OFF`` is refused."""
        text = "MESH\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n"
        with pytest.raises(FileFormatNotSupportedError, match="not an OFF file"):
            OffDriver().read(_write(tmp_path, text))

    def test_truncated_vertex_block_is_refused(self, tmp_path):
        """A header promising more vertices than the file holds is refused."""
        text = "OFF\n4 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n"
        with pytest.raises(FileFormatNotSupportedError, match="ends inside its vertex"):
            OffDriver().read(_write(tmp_path, text))

    def test_short_face_record_is_refused(self, tmp_path):
        """A face whose leading count promises more ids than the line holds is refused."""
        text = "OFF\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n4 0 1 2\n"
        with pytest.raises(FileFormatNotSupportedError, match="short face record"):
            OffDriver().read(_write(tmp_path, text))

    def test_malformed_header_is_refused(self, tmp_path):
        """A count line with fewer than two numbers is refused."""
        text = "OFF\n3\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n"
        with pytest.raises(FileFormatNotSupportedError, match="malformed OFF header"):
            OffDriver().read(_write(tmp_path, text))


class TestOffCanOpen:
    """``can_open`` recognises the magic and never raises on junk."""

    def test_accepts_the_committed_sample(self):
        """The committed cube is recognised by its ``OFF`` magic."""
        assert OffDriver().can_open(str(_COMMENTS_OFF)) is True

    def test_declines_empty_foreign_and_directory(self, tmp_path):
        """Empty, foreign and directory paths are declined without raising."""
        driver = OffDriver()
        empty = tmp_path / "empty.off"
        empty.write_bytes(b"")
        foreign = tmp_path / "other.off"
        foreign.write_text("not an off header\n", encoding="utf-8")
        directory = tmp_path / "folder"
        directory.mkdir()

        assert driver.can_open(str(empty)) is False
        assert driver.can_open(str(foreign)) is False
        assert driver.can_open(str(directory)) is False
