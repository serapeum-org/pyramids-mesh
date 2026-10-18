"""Tests for the VTK / VTU reader (C4), against the real committed samples.

Three fixtures live under ``tests/data/vtk/`` (see ``SOURCE.txt`` for provenance):
``gh-935.vtk`` is a classic v3.0 ASCII file of two triangles, ``06_unstructured.vtk`` is a
VTK 5.1 ``OFFSETS``/``CONNECTIVITY`` file of twelve hexahedra, and ``tetra.vtu`` is an XML
file of one quadratic tetrahedron and two quadratic triangles. The tests read the
committed bytes; the two synthetic cases (an end-to-end point+cell-data round trip and an
``appended`` payload) are written inline because no committed sample pairs a surface mesh
with cell data or uses a binary payload -- the same inline-fixture pattern the XDMF tests
use for branches the real samples do not cover.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.vtk import (
    VtkDriver,
    _cells_from_offsets,
    _LegacyReader,
    _read_vtu,
)

_DATA = Path(__file__).parent.parent / "data" / "vtk"
_CLASSIC = _DATA / "gh-935.vtk"
_OFFSETS = _DATA / "06_unstructured.vtk"
_VTU = _DATA / "tetra.vtu"


def _read(path):
    """Read a committed sample, ignoring the expected winding/degeneracy warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VtkDriver().read(str(path))


def test_legacy_classic_reads_two_triangles():
    """The classic v3.0 ASCII file reads a 6-node, 2-triangle surface mesh."""
    mesh = _read(_CLASSIC).mesh
    assert mesh.n_node == 6
    assert mesh.n_face == 2
    assert mesh.face_node_connectivity.data.shape == (2, 3)
    assert mesh.face_node_connectivity.original_start_index == 0


def test_legacy_connectivity_is_zero_based():
    """VTK connectivity is already 0-based, so nothing is subtracted on validation."""
    connectivity = _read(_CLASSIC).mesh.face_node_connectivity
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 5


def test_faces_are_counter_clockwise():
    """Every built face comes back counter-clockwise after validation."""
    mesh = _read(_CLASSIC).mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 >= 0


def test_multicomponent_scalar_and_flat_z_are_dropped():
    """The 2-component ``sol`` scalar is skipped, and a flat z gives no ``node_z``."""
    dataset = _read(_CLASSIC)
    assert "sol" not in dataset.data_variable_names
    assert "node_z" not in dataset.data_variable_names


def test_offsets_form_parses():
    """The VTK 5.1 OFFSETS/CONNECTIVITY block parses into twelve hexahedral cells."""
    raw = _LegacyReader(str(_OFFSETS)).read()
    assert len(raw.cells) == 12
    assert raw.cell_types.tolist() == [12] * 12
    # First cell: the first eight connectivity entries, delimited by OFFSETS.
    assert raw.cells[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_offsets_form_captures_point_and_cell_data():
    """The 5.1 file's single-component cell and point scalars are both captured."""
    raw = _LegacyReader(str(_OFFSETS)).read()
    assert raw.cell_data["mat_id"].shape == (12,)
    assert raw.point_data["node_groups"].shape == (42,)


def test_volumetric_grid_is_refused():
    """A grid of only hexahedra has no surface cell and is refused, not built empty."""
    with pytest.raises(FileFormatNotSupportedError, match="no 2D surface cell"):
        VtkDriver().read(str(_OFFSETS))


def test_vtu_reads_quadratic_surface_triangles():
    """The XML file reads its two quadratic triangles as 3-corner faces."""
    mesh = _read(_VTU).mesh
    assert mesh.n_node == 22
    assert mesh.n_face == 2
    assert mesh.face_node_connectivity.data.shape == (2, 3)


def test_cell_type_mapping_takes_corner_nodes():
    """Type 24 (tetra) is dropped; each type-22 cell keeps its first three corners."""
    mesh = _read(_VTU).mesh
    faces = {tuple(int(v) for v in row) for row in mesh.face_node_connectivity.data}
    # The two quadratic triangles are connectivity[10:16] and [16:22]; their corners are
    # the first three of each, regardless of the winding fix's vertex order.
    assert faces == {(10, 11, 12), (16, 17, 18)} or faces == {
        (12, 11, 10),
        (18, 17, 16),
    }


def test_vtu_point_scalar_is_node_located():
    """The ``scalars`` PointData array becomes a node-located variable over 22 nodes."""
    scalars = _read(_VTU).get("scalars")
    assert scalars.location == "node"
    assert scalars.shape == (22,)


def test_point_and_cell_data_round_trip(tmp_path):
    """A classic file's POINT_DATA maps to a node variable, CELL_DATA to a face one."""
    path = tmp_path / "one.vtk"
    path.write_text(
        "# vtk DataFile Version 3.0\n"
        "test\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 float\n"
        "0 0 0 1 0 0 0 1 0\n"
        "CELLS 1 4\n"
        "3 0 1 2\n"
        "CELL_TYPES 1\n"
        "5\n"
        "CELL_DATA 1\n"
        "SCALARS region int\n"
        "LOOKUP_TABLE default\n"
        "7\n"
        "POINT_DATA 3\n"
        "SCALARS height float\n"
        "LOOKUP_TABLE default\n"
        "1.0 2.0 3.0\n",
        encoding="utf-8",
    )
    dataset = _read(path)
    height = dataset.get("height")
    region = dataset.get("region")
    assert height.location == "node"
    assert height.data.tolist() == [1.0, 2.0, 3.0]
    assert region.location == "face"
    assert region.data.tolist() == [7.0]


def test_can_open_declines_junk(tmp_path):
    """can_open accepts the real samples and declines junk, an XML non-VTK, and a dir."""
    driver = VtkDriver()
    assert driver.can_open(str(_CLASSIC)) is True
    assert driver.can_open(str(_VTU)) is True

    junk = tmp_path / "junk.vtk"
    junk.write_text("not a vtk file at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    other_xml = tmp_path / "other.vtu"
    other_xml.write_text("<vtkfile>not vtk</vtkfile>\n", encoding="utf-8")
    assert driver.can_open(str(other_xml)) is False

    empty = tmp_path / "empty.vtk"
    empty.write_bytes(b"")
    assert driver.can_open(str(empty)) is False

    directory = tmp_path / "d.vtk"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_dispatch_picks_legacy_or_xml_by_content():
    """Both spellings read through the one driver, chosen by content not suffix."""
    assert _read(_CLASSIC).mesh.n_face == 2  # legacy ASCII
    assert _read(_VTU).mesh.n_face == 2  # XML VTU


def test_appended_payload_is_refused(tmp_path):
    """An ``appended``/binary VTU payload is refused rather than mis-decoded."""
    path = tmp_path / "appended.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
        '<Piece NumberOfPoints="3" NumberOfCells="1">'
        '<Points><DataArray type="Float32" NumberOfComponents="3" '
        'format="appended" offset="0"/></Points>'
        "<Cells></Cells></Piece></UnstructuredGrid></VTKFile>\n",
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="ascii"):
        _read_vtu(str(path))


def test_cells_from_offsets_splits_at_boundaries():
    """The shared splitter cuts a flat connectivity array at its cumulative offsets."""
    cells = _cells_from_offsets(np.arange(6), np.array([0, 3, 6]))
    assert [cell.tolist() for cell in cells] == [[0, 1, 2], [3, 4, 5]]
