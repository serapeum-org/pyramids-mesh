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
    _Cursor,
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


# -- XML (.vtu) branches the real samples do not exercise ----------------------------
#
# tetra.vtu is a single quadratic-tetra-plus-two-triangles grid with a PointData scalar
# and an empty CellData; the branches below -- a 2-component geometry, a missing DataArray,
# a piece with no attribute sections at all, and the multi-component / nameless attribute
# skips -- need tiny inline VTU text the committed sample has no shape for. They are plain
# XML parsed by our own ``xml.etree`` walk, not reverse-engineered binary (the pattern the
# XDMF tests use), so authoring them here is honest white-box coverage.

_VTU_TRIANGLE_2D = (
    '<?xml version="1.0"?>\n'
    '<VTKFile type="UnstructuredGrid">\n'
    "  <UnstructuredGrid>\n"
    '    <Piece NumberOfPoints="3" NumberOfCells="1">\n'
    "      <Points>\n"
    '        <DataArray type="Float32" NumberOfComponents="2" format="ascii">\n'
    "          0 0  1 0  0 1\n"
    "        </DataArray>\n"
    "      </Points>\n"
    "      <Cells>\n"
    '        <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2</DataArray>\n'
    '        <DataArray type="Int64" Name="offsets" format="ascii">3</DataArray>\n'
    '        <DataArray type="UInt8" Name="types" format="ascii">5</DataArray>\n'
    "      </Cells>\n"
    "    </Piece>\n"
    "  </UnstructuredGrid>\n"
    "</VTKFile>\n"
)


def test_vtu_two_component_points_are_z_padded(tmp_path):
    """A 2-component ``<Points>`` geometry is padded to 3 columns, so z stays flat."""
    path = tmp_path / "flat2d.vtu"
    path.write_text(_VTU_TRIANGLE_2D, encoding="utf-8")
    dataset = _read(path)
    assert dataset.mesh.n_node == 3
    assert dataset.mesh.n_face == 1
    assert "node_z" not in dataset.data_variable_names


def test_vtu_missing_types_dataarray_is_refused(tmp_path):
    """A ``<Cells>`` block missing its ``types`` DataArray is refused by name."""
    path = tmp_path / "no_types.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
        '<Piece NumberOfPoints="3" NumberOfCells="1">'
        '<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'
        "0 0 0 1 0 0 0 1 0</DataArray></Points>"
        "<Cells>"
        '<DataArray type="Int64" Name="connectivity" format="ascii">0 1 2</DataArray>'
        '<DataArray type="Int64" Name="offsets" format="ascii">3</DataArray>'
        "</Cells></Piece></UnstructuredGrid></VTKFile>\n",
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="types"):
        _read_vtu(str(path))


def test_vtu_wrong_root_type_is_refused(tmp_path):
    """A VTKFile that is not an UnstructuredGrid is refused, naming the expected type."""
    path = tmp_path / "poly.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PolyData"></VTKFile>\n',
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="UnstructuredGrid"):
        _read_vtu(str(path))


def test_vtu_missing_child_element_is_refused(tmp_path):
    """A VTKFile with no inner <UnstructuredGrid> element names what is missing."""
    path = tmp_path / "empty.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n' '<VTKFile type="UnstructuredGrid"></VTKFile>\n',
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="UnstructuredGrid"):
        _read_vtu(str(path))


def test_vtu_without_attribute_sections_reads(tmp_path):
    """A piece with neither PointData nor CellData reads to a bare mesh, no variables."""
    path = tmp_path / "bare.vtu"
    path.write_text(
        _VTU_TRIANGLE_2D.replace(
            'NumberOfComponents="2" format="ascii">\n          0 0  1 0  0 1\n',
            'NumberOfComponents="3" format="ascii">\n          0 0 0  1 0 0  0 1 0\n',
        ),
        encoding="utf-8",
    )
    dataset = _read(path)
    assert dataset.mesh.n_face == 1
    assert list(dataset.data_variable_names) == []


def test_vtu_skips_multicomponent_and_nameless_attributes(tmp_path):
    """A multi-component array and a nameless one are skipped; the named scalar survives."""
    path = tmp_path / "attrs.vtu"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
        '<Piece NumberOfPoints="3" NumberOfCells="1">'
        "<PointData>"
        '<DataArray type="Float32" Name="vec" NumberOfComponents="2" format="ascii">'
        "0 0 0 0 0 0</DataArray>"
        '<DataArray type="Float32" NumberOfComponents="1" format="ascii">1 2 3</DataArray>'
        '<DataArray type="Float32" Name="good" format="ascii">4 5 6</DataArray>'
        "</PointData>"
        '<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'
        "0 0 0 1 0 0 0 1 0</DataArray></Points>"
        "<Cells>"
        '<DataArray type="Int64" Name="connectivity" format="ascii">0 1 2</DataArray>'
        '<DataArray type="Int64" Name="offsets" format="ascii">3</DataArray>'
        '<DataArray type="UInt8" Name="types" format="ascii">5</DataArray>'
        "</Cells></Piece></UnstructuredGrid></VTKFile>\n",
        encoding="utf-8",
    )
    dataset = _read(path)
    assert dataset.get("good").data.tolist() == [4.0, 5.0, 6.0]
    assert "vec" not in dataset.data_variable_names


# -- the forward token cursor --------------------------------------------------------


def test_cursor_take_refuses_a_truncated_stream():
    """``take`` on an exhausted cursor refuses the truncated file."""
    with pytest.raises(FileFormatNotSupportedError, match="truncated"):
        _Cursor([], "trunc.vtk").take()


def test_cursor_take_int_refuses_a_non_integer():
    """``take_int`` refuses a token that is not an integer, naming it."""
    with pytest.raises(FileFormatNotSupportedError, match="expected an integer"):
        _Cursor(["xyz"], "bad.vtk").take_int()


def test_cursor_take_block_refuses_a_short_block():
    """``take_block`` refuses when fewer than ``count`` tokens remain."""
    with pytest.raises(FileFormatNotSupportedError, match="truncated"):
        _Cursor(["1"], "short.vtk").take_block(3, np.float64)


def test_cursor_take_block_refuses_a_non_numeric_block():
    """``take_block`` refuses a block holding a non-numeric value."""
    with pytest.raises(FileFormatNotSupportedError, match="non-numeric"):
        _Cursor(["a", "b"], "bad.vtk").take_block(2, np.float64)


# -- legacy ASCII (.vtk) header and attribute branches -------------------------------
#
# Small inline legacy snippets: the committed gh-935.vtk is a clean v3.0 mesh, so the
# header refusals (binary encoding, a missing DATASET line, a non-UNSTRUCTURED_GRID
# dataset) and the attribute skips (VECTORS, COLOR_SCALARS, FIELD, an unknown keyword, a
# SCALARS block out of place) have no committed shape to read. VTK legacy is a plain
# token stream parsed by our own ``_LegacyReader``, so authoring them inline is honest.

_VTK_PREFIX = "# vtk DataFile Version 3.0\ntitle\n"
_VALID_TRIANGLE = (
    "ASCII\n"
    "DATASET UNSTRUCTURED_GRID\n"
    "POINTS 3 float\n"
    "0 0 0 1 0 0 0 1 0\n"
    "CELLS 1 4\n"
    "3 0 1 2\n"
    "CELL_TYPES 1\n"
    "5\n"
)


def _legacy(tmp_path, body, name="legacy.vtk"):
    """Write a legacy VTK body behind the magic + title prefix and return its path."""
    path = tmp_path / name
    path.write_text(_VTK_PREFIX + body, encoding="utf-8")
    return str(path)


def test_legacy_missing_points_block_is_refused(tmp_path):
    """A header with no POINTS or CELL_TYPES block is refused, not built empty."""
    path = _legacy(tmp_path, "ASCII\nDATASET UNSTRUCTURED_GRID\n")
    with pytest.raises(FileFormatNotSupportedError, match="POINTS or CELL_TYPES"):
        _LegacyReader(path).read()


def test_legacy_binary_encoding_is_refused(tmp_path):
    """A BINARY legacy file is refused rather than mis-read as ASCII."""
    path = _legacy(tmp_path, "BINARY\nDATASET UNSTRUCTURED_GRID\n")
    with pytest.raises(FileFormatNotSupportedError, match="only ASCII is read"):
        _LegacyReader(path).read()


def test_legacy_missing_dataset_line_is_refused(tmp_path):
    """An ASCII file whose keyword after ASCII is not DATASET is refused."""
    path = _legacy(tmp_path, "ASCII\nNOTDATASET UNSTRUCTURED_GRID\n")
    with pytest.raises(FileFormatNotSupportedError, match="no DATASET line"):
        _LegacyReader(path).read()


def test_legacy_non_unstructured_dataset_is_refused(tmp_path):
    """A DATASET other than UNSTRUCTURED_GRID is refused by name."""
    path = _legacy(tmp_path, "ASCII\nDATASET POLYDATA\n")
    with pytest.raises(FileFormatNotSupportedError, match="UNSTRUCTURED_GRID"):
        _LegacyReader(path).read()


def test_legacy_offsets_without_connectivity_is_refused(tmp_path):
    """An OFFSETS array not followed by a CONNECTIVITY array is refused."""
    body = (
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 float\n"
        "0 0 0 1 0 0 0 1 0\n"
        "CELLS 2 3\n"
        "OFFSETS vtktypeint64\n"
        "0 3\n"
        "NOTCONNECTIVITY vtktypeint64\n"
        "0 1 2\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="CONNECTIVITY"):
        _LegacyReader(_legacy(tmp_path, body)).read()


def test_legacy_scalars_outside_a_data_section_is_refused(tmp_path):
    """A SCALARS block before any POINT_DATA / CELL_DATA section is refused."""
    body = _VALID_TRIANGLE + "SCALARS foo float\nLOOKUP_TABLE default\n1 2 3\n"
    with pytest.raises(FileFormatNotSupportedError, match="outside POINT_DATA"):
        _LegacyReader(_legacy(tmp_path, body)).read()


def test_legacy_vectors_attribute_is_consumed_and_dropped(tmp_path):
    """A VECTORS array is read past and dropped, leaving the mesh otherwise intact."""
    body = _VALID_TRIANGLE + "POINT_DATA 3\nVECTORS vel float\n0 0 0 0 0 0 0 0 0\n"
    raw = _LegacyReader(_legacy(tmp_path, body)).read()
    assert "vel" not in raw.point_data
    assert len(raw.cells) == 1


def test_legacy_color_scalars_attribute_is_consumed(tmp_path):
    """A COLOR_SCALARS array declares its own component count and is read past."""
    body = _VALID_TRIANGLE + "CELL_DATA 1\nCOLOR_SCALARS cols 2\n0.1 0.2\n"
    raw = _LegacyReader(_legacy(tmp_path, body)).read()
    assert "cols" not in raw.cell_data


def test_legacy_field_attribute_is_consumed(tmp_path):
    """A FIELD block and all of its sub-arrays are read past and dropped."""
    body = _VALID_TRIANGLE + "POINT_DATA 3\nFIELD FieldData 1\narr 1 3 float\n1 2 3\n"
    raw = _LegacyReader(_legacy(tmp_path, body)).read()
    assert raw.point_data == {}


def test_legacy_vectors_outside_a_data_section_is_refused(tmp_path):
    """A VECTORS array before any POINT_DATA / CELL_DATA section is refused."""
    body = _VALID_TRIANGLE + "VECTORS v float\n0 0 0 0 0 0 0 0 0\n"
    with pytest.raises(FileFormatNotSupportedError, match="outside POINT_DATA"):
        _LegacyReader(_legacy(tmp_path, body)).read()


def test_legacy_unknown_attribute_keyword_is_refused(tmp_path):
    """An attribute keyword this reader does not know is refused rather than guessed."""
    body = _VALID_TRIANGLE + "BOGUS whatever\n"
    with pytest.raises(FileFormatNotSupportedError, match="does not read"):
        _LegacyReader(_legacy(tmp_path, body)).read()
