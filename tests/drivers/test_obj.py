"""Tests for the Wavefront OBJ surface-mesh reader.

A hand-parsed geometry format: ``v x y z`` vertices and 1-based ``f`` faces, no time axis
and no CRS. The committed ``elephav.obj`` sample (623 v / 1148 f) exercises the real file;
inline fixtures cover the n-gon, ``f v/vt/vn`` triple and error branches.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.obj import ObjDriver, _pad_faces

_SAMPLE = Path(__file__).parent.parent / "data" / "obj" / "elephav.obj"


def _write(tmp_path, text, name="surface.obj"):
    """Write a text fixture and return its path as a string."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestObjSample:
    """Reading the committed elephant OBJ sample."""

    def test_reads_expected_node_and_face_counts(self):
        """The sample reads into its 623 vertices and 1148 triangular faces."""
        dataset = ObjDriver().read(str(_SAMPLE))

        assert dataset.mesh.n_node == 623
        assert dataset.mesh.n_face == 1148

    def test_connectivity_is_zero_based_with_recorded_base(self):
        """The 1-based OBJ indices arrive 0-based, with the base recorded for a writer."""
        connectivity = ObjDriver().read(str(_SAMPLE)).mesh.face_node_connectivity

        assert int(connectivity.data.min()) == 0
        assert connectivity.original_start_index == 1

    def test_elevation_becomes_node_z(self):
        """The vertex z is carried as a node-located ``node_z`` variable."""
        dataset = ObjDriver().read(str(_SAMPLE))

        node_z = dataset.get("node_z")
        assert node_z.location == "node"
        assert node_z.data.shape == (623,)

    def test_crs_is_none(self):
        """OBJ carries no CRS, so it is never set."""
        assert ObjDriver().read(str(_SAMPLE)).mesh.crs is None


class TestObjInline:
    """Hand-built fixtures for the geometry and parsing branches."""

    def test_triangle_faces_arrive_zero_based(self, tmp_path):
        """A minimal two-triangle OBJ reads its 1-based faces into 0-based rows."""
        text = (
            "v 0.0 0.0 2.0\n"
            "v 1.0 0.0 2.5\n"
            "v 1.0 1.0 3.0\n"
            "v 0.0 1.0 3.5\n"
            "f 1 2 3\n"
            "f 1 3 4\n"
        )
        dataset = ObjDriver().read(_write(tmp_path, text))

        assert dataset.mesh.face_node_connectivity.data.tolist() == [
            [0, 1, 2],
            [0, 2, 3],
        ]
        assert dataset.get("node_z").data.tolist() == [2.0, 2.5, 3.0, 3.5]

    def test_face_vertex_texture_normal_triples_take_vertex_index(self, tmp_path):
        """``f v/vt/vn`` and ``f v//vn`` tokens use the index before the first slash."""
        text = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 1.0 1.0 0.0\n"
            "vt 0.0 0.0\n"
            "vn 0.0 0.0 1.0\n"
            "f 1/1/1 2/1/1 3//1\n"
        )
        dataset = ObjDriver().read(_write(tmp_path, text))

        assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2]]

    def test_mixed_triangle_and_quad_pads_to_four(self, tmp_path):
        """A triangle beside a quad pads the short row to the widest with -1."""
        text = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 1.0 1.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "v 2.0 0.5 0.0\n"
            "f 1 2 3 4\n"
            "f 2 5 3\n"
        )
        rows = ObjDriver().read(_write(tmp_path, text)).mesh.face_node_connectivity.data
        rows = rows.tolist()

        assert rows[0] == [0, 1, 2, 3]
        assert rows[1] == [1, 4, 2, -1]

    def test_comment_lines_are_ignored(self, tmp_path):
        """A ``#`` comment is skipped rather than parsed as a record."""
        text = (
            "# elephant, decimated\n"
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 1.0 1.0 0.0\n"
            "f 1 2 3\n"
        )
        assert ObjDriver().read(_write(tmp_path, text)).mesh.n_node == 3


class TestObjCanOpen:
    """``can_open`` is a content check over a suffix match, and never raises."""

    def test_accepts_the_sample(self):
        """The committed OBJ is recognised."""
        assert ObjDriver().can_open(str(_SAMPLE)) is True

    def test_declines_empty_foreign_and_directory(self, tmp_path):
        """Empty, non-OBJ and directory paths are declined without raising."""
        driver = ObjDriver()
        empty = tmp_path / "empty.obj"
        empty.write_bytes(b"")
        foreign = tmp_path / "other.obj"
        foreign.write_text("hello world\n", encoding="utf-8")
        directory = tmp_path / "folder"
        directory.mkdir()

        assert driver.can_open(str(empty)) is False
        assert driver.can_open(str(foreign)) is False
        assert driver.can_open(str(directory)) is False

    def test_declines_a_point_cloud_without_faces(self, tmp_path):
        """A ``v``-only file has no face records and is not recognised as a surface."""
        text = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\n"
        assert ObjDriver().can_open(_write(tmp_path, text)) is False

    def test_declines_stl_content(self, tmp_path):
        """ASCII STL content under an .obj suffix carries no v/f records, so it declines."""
        text = (
            "solid cube\n"
            "facet normal 0 0 1\n outer loop\n"
            "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 1 1 0\n"
            " endloop\n endfacet\n"
            "endsolid cube\n"
        )
        assert ObjDriver().can_open(_write(tmp_path, text)) is False


class TestObjErrors:
    """Malformed surfaces raise a named ``FileFormatNotSupportedError``."""

    def test_no_vertices_is_refused(self, tmp_path):
        """A file with faces but no vertices is not a readable surface."""
        with pytest.raises(FileFormatNotSupportedError, match="no 'v' vertex"):
            ObjDriver().read(_write(tmp_path, "f 1 2 3\n"))

    def test_no_faces_is_refused(self, tmp_path):
        """A file with vertices but no faces is a point cloud, not a mesh."""
        text = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\n"
        with pytest.raises(FileFormatNotSupportedError, match="no 'f' faces"):
            ObjDriver().read(_write(tmp_path, text))

    def test_short_vertex_is_refused(self, tmp_path):
        """A vertex line missing a coordinate is refused."""
        text = "v 0.0 0.0\nf 1 1 1\n"
        with pytest.raises(FileFormatNotSupportedError, match="short vertex"):
            ObjDriver().read(_write(tmp_path, text))

    def test_face_with_two_vertices_is_refused(self, tmp_path):
        """A face of fewer than three vertices is an edge, not a surface element."""
        text = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\nf 1 2\n"
        with pytest.raises(FileFormatNotSupportedError, match="fewer than three"):
            ObjDriver().read(_write(tmp_path, text))

    def test_pad_faces_guards_a_short_row_directly(self):
        """`_pad_faces` defends its own width guard, reached directly past `build`."""
        with pytest.raises(FileFormatNotSupportedError, match="fewer than three"):
            _pad_faces([[0, 1, 2], [0, 1]], "surface.obj")

    def test_non_numeric_coordinate_is_refused(self, tmp_path):
        """A non-numeric vertex coordinate is refused, naming it."""
        text = "v east 0.0 0.0\nf 1 1 1\n"
        with pytest.raises(FileFormatNotSupportedError, match="expected a number"):
            ObjDriver().read(_write(tmp_path, text))

    def test_non_numeric_face_index_is_refused(self, tmp_path):
        """A non-numeric face index is refused by the integer parser."""
        text = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\nf a b c\n"
        with pytest.raises(FileFormatNotSupportedError, match="expected an integer"):
            ObjDriver().read(_write(tmp_path, text))


def test_declares_read_capabilities():
    """The driver advertises read_mesh only (it emits geometry, no field data)."""
    assert ObjDriver.name == "obj"
    assert ObjDriver.suffixes == (".obj",)
    assert set(ObjDriver.capabilities) == {"read_mesh"}


def test_negative_relative_face_index_is_refused(tmp_path):
    """OBJ relative (negative) face indices are unsupported and refused cleanly."""
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -1 -2 -3\n"
    with pytest.raises(
        FileFormatNotSupportedError, match="non-positive OBJ face index"
    ):
        ObjDriver().read(_write(tmp_path, text))
