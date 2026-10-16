"""Tests for the geometry interchange readers: PLY and XMS TIN (D5).

Both are hand-parsed geometry formats -- no time axis, elevation per node. The PLY tests
cover ASCII and both binary byte orders, extra vertex properties becoming node variables,
and mixed-width faces; the TIN tests cover the 1-based index base. Esri TIN is not tested
because it is not implemented: an undocumented directory format with no vendor sample.
"""

import pyramids  # noqa: F401  # isort:skip

import struct
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers import interchange
from pyramids_mesh.drivers.interchange import (
    PlyDriver,
    XmsTinDriver,
    build_minimal_fixture_ply,
    build_minimal_fixture_xmstin,
)

_ASCII_PLY = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 2.0
1.0 0.0 2.5
1.0 1.0 3.0
0.0 1.0 3.5
3 0 1 2
3 0 2 3
"""


def _write(tmp_path, text, name="surface.ply"):
    """Write a text fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _binary_ply(tmp_path, endian, name="surface.ply"):
    """Write a binary PLY (``endian`` is ``"<"`` or ``">"``) and return its path."""
    marker = "binary_little_endian" if endian == "<" else "binary_big_endian"
    header = (
        f"ply\nformat {marker} 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.array(
        [[0, 0, 2.0], [1, 0, 2.5], [1, 1, 3.0], [0, 1, 3.5]], dtype=f"{endian}f4"
    )
    body = verts.tobytes()
    for face in ([0, 1, 2], [0, 2, 3]):
        body += struct.pack(f"{endian}B", 3)
        body += np.array(face, dtype=f"{endian}i4").tobytes()
    path = tmp_path / name
    path.write_bytes(header + body)
    return str(path)


class TestPly:
    """Tests for the Stanford PLY reader."""

    def test_ply_reads_triangles_and_node_z(self, tmp_path):
        """An ASCII PLY reads into nodes, faces and a ``node_z`` elevation variable."""
        dataset = PlyDriver().read(_write(tmp_path, _ASCII_PLY))

        assert dataset.mesh.n_node == 4
        assert dataset.mesh.face_node_connectivity.data.tolist() == [
            [0, 1, 2],
            [0, 2, 3],
        ]
        assert dataset.get("node_z").location == "node"
        assert dataset.get("node_z").data.tolist() == [2.0, 2.5, 3.0, 3.5]

    def test_ply_connectivity_is_zero_based(self, tmp_path):
        """PLY indices are 0-based and stay 0-based."""
        dataset = PlyDriver().read(_write(tmp_path, _ASCII_PLY))

        connectivity = dataset.mesh.face_node_connectivity
        assert int(connectivity.data.min()) == 0
        assert connectivity.original_start_index == 0

    @pytest.mark.parametrize("endian", ["<", ">"])
    def test_ply_binary_both_byte_orders_read(self, tmp_path, endian):
        """A binary PLY reads identically in either declared byte order."""
        dataset = PlyDriver().read(_binary_ply(tmp_path, endian))

        assert dataset.mesh.n_node == 4
        assert dataset.get("node_z").data.tolist() == [2.0, 2.5, 3.0, 3.5]

    def test_ply_vertex_properties_become_node_variables(self, tmp_path):
        """A non-coordinate vertex property is carried as its own node variable."""
        text = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 1.0 10
1.0 0.0 1.0 20
1.0 1.0 1.0 30
3 0 1 2
"""
        dataset = PlyDriver().read(_write(tmp_path, text))

        red = dataset.get("red")
        assert red.location == "node"
        assert red.data.tolist() == [10.0, 20.0, 30.0]

    def test_ply_mixed_triangle_and_quad_pads_to_four(self, tmp_path):
        """A mix of a triangle and a quad pads the short row to the widest with -1."""
        text = """ply
format ascii 1.0
element vertex 5
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
2.0 0.5 0.0
4 0 1 2 3
3 1 4 2
"""
        dataset = PlyDriver().read(_write(tmp_path, text))

        rows = dataset.mesh.face_node_connectivity.data.tolist()
        assert rows[0] == [0, 1, 2, 3]
        assert rows[1] == [1, 4, 2, -1]

    def test_ply_round_trip_documents_z_as_variable(self, tmp_path):
        """The geometry is planar; z survives only as the ``node_z`` variable."""
        dataset = PlyDriver().read(_write(tmp_path, _ASCII_PLY))

        assert "node_z" in dataset.data_variable_names
        assert float(dataset.mesh.node_x[2]) == 1.0

    def test_ply_face_with_two_vertices_is_refused(self, tmp_path):
        """An edge (two-vertex face) is refused -- it is not a surface element."""
        text = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
2 0 1
"""
        with pytest.raises(FileFormatNotSupportedError, match="fewer than three"):
            PlyDriver().read(_write(tmp_path, text))

    def test_ply_point_cloud_without_faces_is_refused(self, tmp_path):
        """A PLY with no faces is a point cloud, not a mesh, and is refused."""
        text = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
element face 0
property list uchar int vertex_indices
end_header
0.0 0.0 1.0
1.0 1.0 2.0
"""
        with pytest.raises(FileFormatNotSupportedError, match="no faces"):
            PlyDriver().read(_write(tmp_path, text))

    def test_ply_unknown_property_type_is_refused(self, tmp_path):
        """A property of an unknown scalar type is refused rather than guessed."""
        text = _ASCII_PLY.replace("property float x", "property quadruple x")
        with pytest.raises(FileFormatNotSupportedError, match="unknown PLY property"):
            PlyDriver().read(_write(tmp_path, text))

    def test_ply_truncated_body_is_refused(self, tmp_path):
        """A body with fewer values than the header promises is refused."""
        text = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 1.0
1.0 0.0 1.0
"""
        with pytest.raises(FileFormatNotSupportedError, match="ends inside its data"):
            PlyDriver().read(_write(tmp_path, text))

    def test_ply_crs_is_none(self, tmp_path):
        """PLY carries no CRS, so it is never set."""
        dataset = PlyDriver().read(_write(tmp_path, _ASCII_PLY))

        assert dataset.mesh.crs is None

    def test_ply_can_open_false_on_junk(self, tmp_path):
        """``can_open`` declines empty, foreign and directory paths without raising."""
        driver = PlyDriver()
        empty = tmp_path / "empty.ply"
        empty.write_bytes(b"")
        foreign = tmp_path / "other.ply"
        foreign.write_text("not a ply header\n", encoding="utf-8")
        directory = tmp_path / "folder"
        directory.mkdir()

        assert driver.can_open(str(empty)) is False
        assert driver.can_open(str(foreign)) is False
        assert driver.can_open(str(directory)) is False

    def test_ply_committed_fixture_matches_the_builder(self, tmp_path):
        """The committed fixture reads and matches ``build_minimal_fixture_ply``."""
        committed = Path(__file__).parent.parent / "data" / "minimal.ply"
        dataset = PlyDriver().read(str(committed))

        assert dataset.mesh.n_node == 4
        generated = tmp_path / "generated.ply"
        build_minimal_fixture_ply(str(generated))
        assert generated.read_text(encoding="utf-8") == committed.read_text(
            encoding="utf-8"
        )


class TestXmsTin:
    """Tests for the XMS / SMS TIN reader."""

    _TIN = """TIN
BEGT
TNAM surface
MAT 1
VERT 4
0.0 0.0 2.0 0
1.0 0.0 2.5 0
1.0 1.0 3.0 0
0.0 1.0 3.5 0
TRI 2
1 2 3
1 3 4
ENDT
"""

    def test_tin_reads_triangulation(self, tmp_path):
        """A TIN reads into nodes, triangles and the ``node_z`` elevation."""
        dataset = XmsTinDriver().read(_write(tmp_path, self._TIN, "s.tin"))

        assert dataset.mesh.n_node == 4
        assert dataset.mesh.n_face == 2
        assert dataset.get("node_z").data.tolist() == [2.0, 2.5, 3.0, 3.5]

    def test_tin_indices_are_one_based(self, tmp_path):
        """The 1-based triangle indices arrive 0-based, with the base recorded."""
        dataset = XmsTinDriver().read(_write(tmp_path, self._TIN, "s.tin"))

        connectivity = dataset.mesh.face_node_connectivity
        assert connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]
        assert connectivity.original_start_index == 1

    def test_tin_not_a_tin_header_is_refused(self, tmp_path):
        """A file not starting with the TIN header is refused."""
        text = self._TIN.replace("TIN\n", "MESH\n", 1)
        with pytest.raises(FileFormatNotSupportedError, match="not an XMS TIN"):
            XmsTinDriver().read(_write(tmp_path, text, "s.tin"))

    def test_tin_short_vertex_record_is_refused(self, tmp_path):
        """A vertex record missing a coordinate is refused."""
        text = self._TIN.replace("0.0 0.0 2.0 0", "0.0 0.0")
        with pytest.raises(FileFormatNotSupportedError, match="short vertex record"):
            XmsTinDriver().read(_write(tmp_path, text, "s.tin"))

    def test_tin_truncated_triangle_block_is_refused(self, tmp_path):
        """A TRI block promising more rows than the file holds is refused."""
        text = """TIN
BEGT
VERT 3
0.0 0.0 1.0 0
1.0 0.0 1.0 0
1.0 1.0 1.0 0
TRI 5
1 2 3
"""
        with pytest.raises(FileFormatNotSupportedError, match="ends inside its TRI"):
            XmsTinDriver().read(_write(tmp_path, text, "s.tin"))

    def test_tin_crs_is_none(self, tmp_path):
        """XMS TIN carries no CRS, so it is never set."""
        dataset = XmsTinDriver().read(_write(tmp_path, self._TIN, "s.tin"))

        assert dataset.mesh.crs is None

    def test_tin_can_open_false_on_junk(self, tmp_path):
        """``can_open`` declines empty, foreign and directory paths without raising."""
        driver = XmsTinDriver()
        empty = tmp_path / "empty.tin"
        empty.write_bytes(b"")
        foreign = tmp_path / "other.tin"
        foreign.write_text("not a tin\n", encoding="utf-8")
        directory = tmp_path / "folder"
        directory.mkdir()

        assert driver.can_open(str(empty)) is False
        assert driver.can_open(str(foreign)) is False
        assert driver.can_open(str(directory)) is False

    def test_tin_committed_fixture_matches_the_builder(self, tmp_path):
        """The committed fixture reads and matches ``build_minimal_fixture_xmstin``."""
        committed = Path(__file__).parent.parent / "data" / "minimal.tin"
        dataset = XmsTinDriver().read(str(committed))

        assert dataset.mesh.n_node == 4
        generated = tmp_path / "generated.tin"
        build_minimal_fixture_xmstin(str(generated))
        assert generated.read_text(encoding="utf-8") == committed.read_text(
            encoding="utf-8"
        )
