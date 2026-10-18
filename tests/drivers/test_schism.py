"""Tests for the SCHISM ``hgrid.gr3`` reader (C2), against a real committed sample.

The fixture is the genuine SCHISM ``hgrid.gr3`` committed under ``tests/data/schism/``
(Apache-2.0, CADWRDeltaModeling/HelloSCHISM): 2639 nodes, 4636 triangles and three open
plus three land boundary sections. The tests read the committed bytes.

The committed grid is triangle-only, so the ``i34 == 4`` quad branch is covered by a
small constructed (white-box) ``.gr3`` written inline, rather than by fabricating a large
binary.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers._gr3 import read_gr3_grid
from pyramids_mesh.drivers.schism import SchismDriver

_DATA = Path(__file__).parent.parent / "data" / "schism"
_GRID = _DATA / "hgrid.gr3"

_QUAD_GRID = (
    "tiny quad grid\n"
    "2 5\n"
    "1 0.0 0.0 1.0\n"
    "2 1.0 0.0 1.0\n"
    "3 1.0 1.0 1.0\n"
    "4 0.0 1.0 1.0\n"
    "5 2.0 0.0 1.0\n"
    "1 4 1 2 3 4\n"
    "2 3 2 5 3\n"
)


def _read(path=_GRID):
    """Read the committed SCHISM grid."""
    return SchismDriver().read(str(path))


def test_reads_nodes_and_triangles():
    """hgrid.gr3 builds a 2639-node, 4636-triangle mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 2639
    assert mesh.n_face == 4636
    assert mesh.face_node_connectivity.data.shape == (4636, 3)


def test_connectivity_is_one_based_resolved_to_zero_based():
    """The 1-based element table arrives 0-based, with the base recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 2638


def test_node_depth_becomes_node_z():
    """The gr3 node depth column becomes a node-located node_z variable."""
    dataset = _read()
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (2639,)
    # node 1 of the committed grid: depth 0.5
    assert float(node_z.data[0]) == pytest.approx(0.5, rel=1e-9)


def test_node_coordinates_match_the_grid():
    """Node 1's coordinates match the committed hgrid.gr3."""
    mesh = _read().mesh
    assert float(mesh.node_x[0]) == pytest.approx(56000.0, rel=1e-9)
    assert float(mesh.node_y[0]) == pytest.approx(-10400.0, rel=1e-9)


def test_faces_are_counter_clockwise():
    """Every sampled face comes back counter-clockwise after validation."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data[:300]:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_boundaries_are_parsed_into_metadata():
    """The open and land boundary node lists are carried as dataset metadata."""
    attributes = _read().global_attributes
    assert attributes["n_open_boundaries"] == 3
    assert attributes["n_land_boundaries"] == 3
    open_nodes = attributes["open_boundary_nodes"]
    land_nodes = attributes["land_boundary_nodes"]
    assert [len(group) for group in open_nodes] == [4, 4, 17]
    assert sum(len(group) for group in open_nodes) == 25
    assert [len(group) for group in land_nodes] == [180, 261, 180]
    assert sum(len(group) for group in land_nodes) == 621


def test_boundary_node_ids_are_zero_based():
    """Boundary node ids are resolved to 0-based so they index into the mesh."""
    dataset = _read()
    first_open = dataset.global_attributes["open_boundary_nodes"][0]
    # file node ids 2630, 2634, 2637, 2639 -> 0-based 2629, 2633, 2636, 2638
    assert first_open.tolist() == [2629, 2633, 2636, 2638]
    # every id is a valid 0-based node index
    assert int(first_open.max()) < dataset.mesh.n_node


def test_reads_quad_and_triangle_mesh(tmp_path):
    """A mixed quad/triangle grid builds a quad-width, fill-padded face table."""
    grid = tmp_path / "quad.gr3"
    grid.write_text(_QUAD_GRID, encoding="utf-8")
    mesh = SchismDriver().read(str(grid)).mesh
    assert mesh.n_face == 2
    faces = mesh.face_node_connectivity.data
    assert faces.shape == (2, 4)
    # the triangle row is padded in its trailing column
    assert int((faces == -1).sum()) == 1
    assert int(faces[1, -1]) == -1


def test_read_gr3_grid_pads_quad_rows(tmp_path):
    """White-box: the shared parser returns raw 1-based ids with triangles pad-filled."""
    grid = tmp_path / "quad.gr3"
    grid.write_text(_QUAD_GRID, encoding="utf-8")
    with open(grid, "r", encoding="latin-1") as handle:
        node_x, node_y, node_z, faces = read_gr3_grid(
            handle, allow_quads=True, source=str(grid)
        )
    assert node_x.shape == (5,)
    assert faces.shape == (2, 4)
    assert faces.tolist() == [[1, 2, 3, 4], [2, 5, 3, -1]]


def test_quad_is_refused_when_quads_disallowed(tmp_path):
    """White-box: allow_quads=False refuses a type-4 element (the ADCIRC contract)."""
    grid = tmp_path / "quad.gr3"
    grid.write_text(_QUAD_GRID, encoding="utf-8")
    with open(grid, "r", encoding="latin-1") as handle:
        with pytest.raises(FileFormatNotSupportedError, match="only reads triangular"):
            read_gr3_grid(handle, allow_quads=False, source=str(grid))


def test_grid_without_boundaries_reads_as_a_bare_mesh(tmp_path):
    """A .gr3 with no boundary section reads cleanly with empty boundary metadata."""
    grid = tmp_path / "bathy.gr3"
    grid.write_text(_QUAD_GRID, encoding="utf-8")
    dataset = SchismDriver().read(str(grid))
    assert dataset.global_attributes["n_open_boundaries"] == 0
    assert dataset.global_attributes["n_land_boundaries"] == 0
    assert dataset.global_attributes["open_boundary_nodes"] == []


def test_can_open_accepts_the_grid():
    """can_open accepts the committed .gr3 grid."""
    assert SchismDriver().can_open(str(_GRID)) is True


def test_can_open_declines_non_gr3(tmp_path):
    """can_open declines a wrong-suffix file, a malformed .gr3, and a directory."""
    driver = SchismDriver()
    plain = tmp_path / "notes.txt"
    plain.write_text("1 2 3\n", encoding="utf-8")
    assert driver.can_open(str(plain)) is False
    bad = tmp_path / "bad.gr3"
    bad.write_text("header\nabc def\n1 2\n", encoding="utf-8")
    assert driver.can_open(str(bad)) is False
    assert driver.can_open(str(tmp_path)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else."""
    assert set(SchismDriver.capabilities) == {"read_mesh"}
    assert SchismDriver.suffixes == (".gr3",)
    assert SchismDriver.name == "schism"
