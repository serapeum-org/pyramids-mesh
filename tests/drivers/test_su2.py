"""Tests for the SU2 reader (C11), against the committed su2code samples.

The fixtures are genuine SU2 ``.su2`` meshes committed under ``tests/data/su2/``
(LGPL-2.1, su2code/TestCases): ``5x5Mesh.su2`` (25 quadrilaterals, 36 nodes) and
``square.su2`` (128 triangles, 81 nodes). The tests read the committed bytes, so they
cover the traps a real SU2 file carries: VTK-typed rows of mixed arity, 0-based
connectivity, the trailing element/point index columns, and the boundary markers.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.su2 import Su2Driver

_DATA = Path(__file__).parent.parent / "data" / "su2"
_QUADS = _DATA / "5x5Mesh.su2"
_TRIANGLES = _DATA / "square.su2"


def _read(path):
    """Read a committed SU2 sample."""
    return Su2Driver().read(str(path))


def _signed_area2(mesh, row):
    """Twice the signed area of one face of a built mesh (positive = CCW)."""
    real = row[row != -1]
    xs = np.asarray(mesh.node_x)[real]
    ys = np.asarray(mesh.node_y)[real]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def test_quad_file_reads_nodes_and_quad_faces():
    """5x5Mesh.su2 builds a 36-node mesh of 25 four-node faces."""
    mesh = _read(_QUADS).mesh

    assert mesh.n_node == 36
    assert mesh.n_face == 25
    assert mesh.face_node_connectivity.data.shape == (25, 4)


def test_triangle_file_reads_nodes_and_tri_faces():
    """square.su2 builds an 81-node mesh of 128 three-node faces."""
    mesh = _read(_TRIANGLES).mesh

    assert mesh.n_node == 81
    assert mesh.n_face == 128
    assert mesh.face_node_connectivity.data.shape == (128, 3)


@pytest.mark.parametrize(
    "path, first_face",
    [(_QUADS, [0, 1, 7, 6]), (_TRIANGLES, [0, 1, 9])],
)
def test_connectivity_is_zero_based_and_not_shifted(path, first_face):
    """SU2 is 0-based: the node indices arrive unshifted, base 0 recorded for a writer.

    The first element of each file is a known 0-based row, so a stray ``-1`` shift (the
    1-based re-basing the 2DM driver applies) would move every index and be caught here.
    """
    connectivity = _read(path).mesh.face_node_connectivity

    assert int(connectivity.data.min()) == 0
    assert connectivity.original_start_index == 0
    assert connectivity.data[0].tolist() == first_face


@pytest.mark.parametrize("path", [_QUADS, _TRIANGLES])
def test_every_face_comes_back_counter_clockwise(path):
    """Validation orients each face counter-clockwise, whatever the file's winding."""
    mesh = _read(path).mesh

    for row in mesh.face_node_connectivity.data:
        assert _signed_area2(mesh, row) > 0


@pytest.mark.parametrize(
    "path, expected",
    [
        (_QUADS, {"bottom": 5, "left": 5, "right": 5, "top": 5}),
        (_TRIANGLES, {"lower": 8, "right": 8, "upper": 8, "left": 8}),
    ],
)
def test_boundary_markers_are_recorded_as_metadata(path, expected):
    """Each MARKER_TAG group is kept as su2_markers (tag -> boundary-element count)."""
    dataset = _read(path)

    assert dataset.global_attributes["su2_markers"] == expected
    assert dataset.global_attributes["source"] == "SU2"
    assert dataset.global_attributes["NDIME"] == 2


def test_mixed_triangle_and_quad_file_pads_to_the_widest(tmp_path):
    """A file mixing a triangle and a quad pads the triangle row to width 4 with -1."""
    text = """NDIME= 2
NELEM= 2
5 0 1 2 0
9 0 1 2 3 1
NPOIN= 4
0.0 0.0 0
1.0 0.0 1
1.0 1.0 2
0.0 1.0 3
NMARK= 0
"""
    path = tmp_path / "mixed.su2"
    path.write_text(text, encoding="utf-8")

    connectivity = Su2Driver().read(str(path)).mesh.face_node_connectivity

    assert connectivity.data.shape == (2, 4)
    assert connectivity.data[0, 3] == -1
    assert sorted(int(v) for v in connectivity.data[1]) == [0, 1, 2, 3]


def test_glued_card_value_is_accepted(tmp_path):
    """A card written NDIME=2 (no space after =) reads as NDIME= 2 does."""
    text = "NDIME=2\nNELEM=1\n5 0 1 2 0\nNPOIN=3\n0 0 0\n1 0 1\n0 1 2\nNMARK=0\n"
    path = tmp_path / "glued.su2"
    path.write_text(text, encoding="utf-8")

    assert Su2Driver().read(str(path)).mesh.n_face == 1


def test_volume_element_type_is_refused(tmp_path):
    """A VTK tetra (code 10) is refused: this reader builds surface meshes only."""
    text = "NDIME= 3\nNELEM= 1\n10 0 1 2 3 0\nNPOIN= 4\n"
    text += "0 0 0 0\n1 0 0 1\n0 1 0 2\n0 0 1 3\nNMARK= 0\n"
    path = tmp_path / "tet.su2"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        FileFormatNotSupportedError, match="volume/line element code 10"
    ):
        Su2Driver().read(str(path))


def test_unknown_vtk_code_is_refused(tmp_path):
    """An element code outside the SU2 table is refused, naming the code."""
    text = "NDIME= 2\nNELEM= 1\n7 0 1 2 0\nNPOIN= 3\n0 0 0\n1 0 1\n0 1 2\nNMARK= 0\n"
    path = tmp_path / "bad.su2"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(FileFormatNotSupportedError, match="VTK element code 7"):
        Su2Driver().read(str(path))


def test_short_element_row_is_refused(tmp_path):
    """A quad row short of its four node ids is a format error, not an IndexError."""
    text = "NDIME= 2\nNELEM= 1\n9 0 1 2\nNPOIN= 3\n0 0 0\n1 0 1\n0 1 2\nNMARK= 0\n"
    path = tmp_path / "short.su2"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(FileFormatNotSupportedError, match="short NELEM row"):
        Su2Driver().read(str(path))


def test_can_open_declines_junk_and_foreign_content(tmp_path):
    """can_open declines non-SU2 content, an empty or binary file, and never raises."""
    driver = Su2Driver()
    junk = tmp_path / "junk.su2"
    junk.write_text("hello world\nnothing here\n", encoding="utf-8")
    empty = tmp_path / "empty.su2"
    empty.write_bytes(b"")
    binary = tmp_path / "binary.su2"
    binary.write_bytes(bytes(range(256)) * 4)
    foreign = tmp_path / "mesh.txt"
    foreign.write_text("NDIME= 2\nNELEM= 1\n", encoding="utf-8")

    assert driver.can_open(str(junk)) is False
    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(binary)) is False
    assert driver.can_open(str(tmp_path / "missing.su2")) is False
    # Right content, wrong suffix: the suffix gate still applies.
    assert driver.can_open(str(foreign)) is False


def test_dispatch_accepts_the_committed_files_and_returns_a_dataset():
    """The dispatch gate (can_open) selects both samples, and read yields a dataset.

    can_open is the discriminator pyramids uses to route a path to a driver, so a True
    here on each committed file, paired with a built UgridDataset, is the dispatch path
    this driver exposes once its entry point is registered.
    """
    driver = Su2Driver()

    for path in (_QUADS, _TRIANGLES):
        assert driver.can_open(str(path)) is True
        assert isinstance(driver.read(str(path)), UgridDataset)


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else."""
    assert set(Su2Driver.capabilities) == {"read_mesh"}
    assert Su2Driver.name == "su2"
    assert Su2Driver.suffixes == (".su2",)
