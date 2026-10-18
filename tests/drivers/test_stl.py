"""Tests for the STL surface-mesh reader.

STL stores three inline vertex coordinates per triangle and has no vertex index, so the
reader welds identical vertices into a node list. The committed binary ``Cube.stl``
(12 triangles) exercises the binary path and the weld; inline fixtures cover the ASCII
path, the ``solid``-prefixed-binary trap and the error branches.
"""

import pyramids  # noqa: F401  # isort:skip

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.stl import StlDriver

_SAMPLE = Path(__file__).parent.parent / "data" / "stl" / "Cube.stl"

_ASCII_STL = (
    "solid square\n"
    "facet normal 0 0 1\n outer loop\n"
    "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 1 1 0\n"
    " endloop\n endfacet\n"
    "facet normal 0 0 1\n outer loop\n"
    "  vertex 0 0 0\n  vertex 1 1 0\n  vertex 0 1 0\n"
    " endloop\n endfacet\n"
    "endsolid square\n"
)


def _read(path):
    """Read an STL, ignoring the planar-projection winding/degeneracy warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StlDriver().read(str(path))


def _write_text(tmp_path, text, name="surface.stl"):
    """Write an ASCII fixture and return its path as a string."""
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return str(path)


def _binary_stl(tmp_path, triangles, header=b"binary sample", name="surface.stl"):
    """Write a binary STL from ``(n, 3, 3)`` vertex coordinates and return its path."""
    tris = np.asarray(triangles, dtype="<f4").reshape(-1, 3, 3)
    raw = header.ljust(80, b"\x00")
    raw += struct.pack("<I", len(tris))
    for triangle in tris:
        raw += np.zeros(3, dtype="<f4").tobytes()  # unit normal, unused
        raw += triangle.astype("<f4").tobytes()
        raw += struct.pack("<H", 0)  # attribute byte count
    path = tmp_path / name
    path.write_bytes(raw)
    return str(path)


class TestStlBinarySample:
    """Reading the committed binary cube sample."""

    def test_welds_to_eight_nodes_and_twelve_faces(self):
        """The 12 triangles weld their shared corners into a cube of 8 nodes."""
        dataset = _read(_SAMPLE)

        assert dataset.mesh.n_node == 8
        assert dataset.mesh.n_face == 12

    def test_solid_prefixed_binary_is_not_read_as_ascii(self):
        """The sample header begins with ' solid' yet is detected as binary by size."""
        assert _SAMPLE.read_bytes().lstrip().startswith(b"solid")
        # If it were mis-read as ASCII it would have no complete facets and raise.
        assert _read(_SAMPLE).mesh.n_face == 12

    def test_elevation_becomes_node_z(self):
        """The third coordinate of each welded vertex is carried as ``node_z``."""
        node_z = _read(_SAMPLE).get("node_z")

        assert node_z.location == "node"
        assert node_z.data.shape == (8,)

    def test_crs_is_none(self):
        """STL carries no CRS, so it is never set."""
        assert _read(_SAMPLE).mesh.crs is None

    def test_connectivity_is_zero_based(self):
        """Welded vertices are 0-based and stay so."""
        connectivity = _read(_SAMPLE).mesh.face_node_connectivity

        assert int(connectivity.data.min()) == 0
        assert connectivity.original_start_index == 0


class TestStlAscii:
    """The ASCII encoding and the welding behaviour."""

    def test_ascii_welds_shared_vertices(self, tmp_path):
        """Two triangles sharing two corners weld to four nodes, not six."""
        dataset = _read(_write_text(tmp_path, _ASCII_STL))

        assert dataset.mesh.n_node == 4
        assert dataset.mesh.n_face == 2

    def test_ascii_face_connectivity_references_welded_nodes(self, tmp_path):
        """The shared corner (0, 0, 0) is one node referenced by both triangles."""
        dataset = _read(_write_text(tmp_path, _ASCII_STL))

        rows = dataset.mesh.face_node_connectivity.data.tolist()
        # Node 0 is the first vertex of both facets (the shared (0, 0, 0) corner).
        assert rows[0][0] == 0
        assert rows[1][0] == 0

    def test_binary_and_ascii_of_the_same_square_agree(self, tmp_path):
        """A binary write of the ASCII square's triangles welds to the same mesh."""
        triangles = [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        ]
        dataset = _read(_binary_stl(tmp_path, triangles))

        assert dataset.mesh.n_node == 4
        assert dataset.mesh.n_face == 2


class TestStlCanOpen:
    """``can_open`` detects both encodings and never raises."""

    def test_accepts_binary_sample(self):
        """The committed binary STL is recognised."""
        assert StlDriver().can_open(str(_SAMPLE)) is True

    def test_accepts_ascii(self, tmp_path):
        """An ASCII ``solid``/``facet`` file is recognised."""
        assert StlDriver().can_open(_write_text(tmp_path, _ASCII_STL)) is True

    def test_declines_empty_foreign_and_directory(self, tmp_path):
        """Empty, non-STL and directory paths are declined without raising."""
        driver = StlDriver()
        empty = tmp_path / "empty.stl"
        empty.write_bytes(b"")
        foreign = tmp_path / "other.stl"
        foreign.write_text("not an stl at all\n", encoding="ascii")
        directory = tmp_path / "folder"
        directory.mkdir()

        assert driver.can_open(str(empty)) is False
        assert driver.can_open(str(foreign)) is False
        assert driver.can_open(str(directory)) is False

    def test_declines_obj_content(self, tmp_path):
        """OBJ content under a .stl suffix is neither binary-sized nor ASCII STL."""
        text = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\nf 1 2 3\n"
        assert StlDriver().can_open(_write_text(tmp_path, text)) is False


class TestStlErrors:
    """Malformed or unrecognised STL files raise a named error."""

    def test_neither_binary_nor_ascii_is_refused(self, tmp_path):
        """A file matching neither layout is refused with a clear message."""
        path = tmp_path / "junk.stl"
        path.write_bytes(b"this is not a readable stl file at all, really\n")
        with pytest.raises(FileFormatNotSupportedError, match="not a readable STL"):
            StlDriver().read(str(path))

    def test_truncated_binary_falls_through_and_is_refused(self, tmp_path):
        """A binary STL cut short of its declared triangle count is refused."""
        path = _binary_stl(
            tmp_path, [[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], name="cut.stl"
        )
        data = Path(path).read_bytes()
        Path(path).write_bytes(data[:-10])
        with pytest.raises(FileFormatNotSupportedError, match="not a readable STL"):
            StlDriver().read(path)

    def test_ascii_facet_without_three_vertices_is_refused(self, tmp_path):
        """A facet carrying other than three vertices is refused."""
        text = (
            "solid bad\n"
            "facet normal 0 0 1\n outer loop\n"
            "  vertex 0 0 0\n  vertex 1 0 0\n"
            " endloop\n endfacet\n"
            "endsolid bad\n"
        )
        with pytest.raises(FileFormatNotSupportedError, match="exactly three"):
            StlDriver().read(_write_text(tmp_path, text))

    def test_ascii_short_vertex_line_is_refused(self, tmp_path):
        """A ``vertex`` line missing a coordinate is refused."""
        text = (
            "solid bad\n"
            "facet normal 0 0 1\n outer loop\n"
            "  vertex 0 0\n  vertex 1 0 0\n  vertex 1 1 0\n"
            " endloop\n endfacet\n"
            "endsolid bad\n"
        )
        with pytest.raises(FileFormatNotSupportedError, match="short vertex"):
            StlDriver().read(_write_text(tmp_path, text))

    def test_ascii_non_numeric_coordinate_is_refused(self, tmp_path):
        """A non-numeric vertex coordinate is refused, naming it."""
        text = _ASCII_STL.replace("vertex 0 0 0", "vertex east 0 0", 1)
        with pytest.raises(FileFormatNotSupportedError, match="expected a number"):
            StlDriver().read(_write_text(tmp_path, text))

    def test_ascii_solid_without_any_facet_is_refused(self, tmp_path):
        """A ``solid`` file with no complete facets has no surface to read."""
        # A lone 'facet' keyword keeps the ASCII sniff happy, but no 'endfacet'
        # closes it, so no triangle is ever collected.
        text = "solid empty\nfacet normal 0 0 1\nendsolid empty\n"
        with pytest.raises(FileFormatNotSupportedError, match="no complete facets"):
            StlDriver().read(_write_text(tmp_path, text))


def test_declares_read_capabilities():
    """The driver advertises read_mesh only (it emits geometry, no field data)."""
    assert StlDriver.name == "stl"
    assert StlDriver.suffixes == (".stl",)
    assert set(StlDriver.capabilities) == {"read_mesh"}


def test_empty_binary_stl_is_refused(tmp_path):
    """An 84-byte binary STL (zero triangles) is refused cleanly, not an IndexError."""
    path = tmp_path / "empty.stl"
    path.write_bytes(b"\x00" * 80 + (0).to_bytes(4, "little"))
    with pytest.raises(FileFormatNotSupportedError, match="empty binary STL"):
        StlDriver().read(str(path))
