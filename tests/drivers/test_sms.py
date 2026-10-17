"""Tests for the SMS 2DM reader (D1).

2DM is the reference driver, so these cover the traps every later driver inherits:
native ids that are not indices, mixed element widths, higher-order elements, winding,
and the 1D/2D split. Fixtures are written inline -- the format is plain text, which is
exactly why it is the cheapest one to test.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import Network1d, UgridDataset

from pyramids_mesh.drivers import sms
from pyramids_mesh.drivers.sms import TwoDmDriver, build_minimal_fixture

_UNIT_SQUARE = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
E3T 1 1 2 3 1
E3T 2 1 3 4 1
"""


def _write(tmp_path, text, name="mesh.2dm"):
    """Write a 2DM fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _signed_area2(mesh, row):
    """Twice the signed area of one face of a built mesh."""
    real = row[row != -1]
    xs, ys = mesh.node_x[real], mesh.node_y[real]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def test_2dm_requires_mesh2d_header(tmp_path):
    """A file that does not open with MESH2D is refused, not guessed at."""
    path = _write(tmp_path, "ND 1 0.0 0.0 0.0\nE3T 1 1 1 1 1\n")

    with pytest.raises(FileFormatNotSupportedError, match="MESH2D"):
        TwoDmDriver().read(path)


@pytest.mark.parametrize(
    "first_line, expected", [("MESH2D", True), ("MESH2DX", False), ("MESH3D", False)]
)
def test_2dm_sniff_and_parser_agree_on_the_header(tmp_path, first_line, expected):
    """``can_open`` accepts exactly the headers ``read`` accepts, and no others.

    A sniff looser than the parser turns an unreadable file into a confusing error
    instead of a clean decline; a stricter one hides a readable file from dispatch.
    """
    path = _write(tmp_path, _UNIT_SQUARE.replace("MESH2D", first_line, 1))

    assert TwoDmDriver().can_open(path) is expected
    if expected:
        assert TwoDmDriver().read(path).mesh.n_face == 2
    else:
        with pytest.raises(FileFormatNotSupportedError, match="MESH2D"):
            TwoDmDriver().read(path)


@pytest.mark.parametrize(
    "text, match",
    [
        ("MESH2D\nND 1 x 0.0 0.0\nND 2 1 0 0\nND 3 1 1 0\nE3T 1 1 2 3 1\n", "ND"),
        (
            "MESH2D\nNUM_MATERIALS_PER_ELEM abc\nND 1 0 0 0\nND 2 1 0 0\n"
            "ND 3 1 1 0\nE3T 1 1 2 3 1\n",
            "NUM_MATERIALS_PER_ELEM",
        ),
        ("MESH2D\nND 1 0 0 0\nND 2 1 0 0\nND 3 1 1 0\nE3T 1 1 2 3 clay\n", "clay"),
        ("MESH2D\nND 1 0 0 0\nND 2 1 0 0\nND 3 1 1 0\nE3T x 1 2 3 1\n", "E3T"),
    ],
)
def test_2dm_non_numeric_field_names_the_file_and_record(tmp_path, text, match):
    """A malformed numeric field is reported with its file and record, not raw."""
    with pytest.raises(FileFormatNotSupportedError, match=match):
        TwoDmDriver().read(_write(tmp_path, text))


@pytest.mark.parametrize("declared", [0, -1])
def test_2dm_unusable_material_column_count_is_refused(tmp_path, declared):
    """A non-positive ``NUM_MATERIALS_PER_ELEM`` is malformed, not a silent clamp to 1.

    Zero and a negative both reach the same guard, and a negative is the one that would
    otherwise pass straight through ``range()`` and yield no material variable at all.
    """
    text = f"""MESH2D
NUM_MATERIALS_PER_ELEM {declared}
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 5
"""
    with pytest.raises(FileFormatNotSupportedError, match="not a usable column count"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_absurd_material_count_is_refused(tmp_path):
    """A declared column count beyond the cap is refused, not allocated.

    Each column costs an ``n_face`` array and a variable, and the count comes from an
    untrusted file, so one corrupt line must not be able to ask for thousands.
    """
    text = """MESH2D
NUM_MATERIALS_PER_ELEM 20000
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 5
"""
    with pytest.raises(FileFormatNotSupportedError, match="expected 1 to 16"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_nd_records_disagreeing_about_z_are_refused(tmp_path):
    """A file where only some ND records carry z would need elevation invented."""
    text = """MESH2D
ND 1 0.0 0.0
ND 2 1.0 0.0 5.0
ND 3 1.0 1.0
E3T 1 1 2 3 1
"""
    with pytest.raises(
        FileFormatNotSupportedError, match="disagree about the z column"
    ):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_malformed_material_names_the_real_element_keyword(tmp_path):
    """A bad material column names the element it came from, not a placeholder."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
E4Q 1 1 2 3 4 clay
"""
    with pytest.raises(FileFormatNotSupportedError, match="malformed E4Q record"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_duplicate_line_element_id_is_refused(tmp_path):
    """Two E2L records sharing an id is refused, as two faces sharing one already was."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
E2L 1 1 2 1
E2L 1 2 1 1
"""
    with pytest.raises(ValueError, match="duplicate native id"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_reads_nodes_and_triangles(tmp_path):
    """Nodes and triangular faces arrive with the right counts and coordinates."""
    dataset = TwoDmDriver().read(_write(tmp_path, _UNIT_SQUARE))

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    np.testing.assert_allclose(dataset.mesh.node_x, [0.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(dataset.mesh.node_y, [0.0, 0.0, 1.0, 1.0])


def test_2dm_connectivity_is_zero_based_after_read(tmp_path):
    """The 1-based file becomes 0-based data, with the file's base still recorded."""
    dataset = TwoDmDriver().read(_write(tmp_path, _UNIT_SQUARE))
    connectivity = dataset.mesh.face_node_connectivity

    np.testing.assert_array_equal(connectivity.data, [[0, 1, 2], [0, 2, 3]])
    assert connectivity.data.min() == 0
    assert connectivity.original_start_index == 1


def test_2dm_mixed_e3t_e4q_pads_to_four(tmp_path):
    """A mixed triangle/quad file is padded to the widest element with -1."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
ND 5 2.0 0.0 0.0
ND 6 2.0 1.0 0.0
E3T 1 1 2 3 1
E4Q 2 2 5 6 3 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))
    connectivity = dataset.mesh.face_node_connectivity

    assert connectivity.data.shape == (2, 4)
    assert connectivity.data[0, 3] == -1
    np.testing.assert_array_equal(connectivity.data[1], [1, 4, 5, 2])


def test_2dm_e6t_read_as_six_node_face(tmp_path):
    """E6T is a real six-node face, never demoted to its corner triangle."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 2.0 0.0 0.0
ND 4 1.5 1.0 0.0
ND 5 1.0 2.0 0.0
ND 6 0.5 1.0 0.0
E6T 1 1 2 3 4 5 6 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))
    connectivity = dataset.mesh.face_node_connectivity

    assert connectivity.data.shape == (1, 6)
    np.testing.assert_array_equal(connectivity.data[0], [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize("element", ["E8Q", "E9Q", "E3L"])
def test_2dm_e8q_raises_unsupported_element(tmp_path, element):
    """Unsupported element types refuse the whole file and name the element."""
    text = f"""MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
{element} 1 1 2 3 1
"""
    with pytest.raises(FileFormatNotSupportedError, match=element):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_non_contiguous_node_ids_resolve_correctly(tmp_path):
    """Gapped node ids resolve through the id map, not by assuming id == index + 1."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 5 1.0 0.0 0.0
ND 9 1.0 1.0 0.0
ND 12 0.0 1.0 0.0
E3T 1 1 5 9 1
E3T 2 1 9 12 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    # The node count is the number of ND records, never the highest id, or the mesh
    # would carry phantom nodes at (0, 0) and wrong bounds.
    assert dataset.mesh.n_node == 4
    np.testing.assert_array_equal(
        dataset.mesh.face_node_connectivity.data, [[0, 1, 2], [0, 2, 3]]
    )


def test_2dm_out_of_order_node_ids_resolve_correctly(tmp_path):
    """Node ids out of file order still wire each face to the right coordinates."""
    text = """MESH2D
ND 7 0.0 0.0 0.0
ND 3 1.0 0.0 0.0
ND 5 1.0 1.0 0.0
E3T 1 7 3 5 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    np.testing.assert_array_equal(dataset.mesh.face_node_connectivity.data, [[0, 1, 2]])
    np.testing.assert_allclose(dataset.mesh.node_x, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(dataset.mesh.node_y, [0.0, 0.0, 1.0])


def test_2dm_material_becomes_face_variable(tmp_path):
    """The material id becomes a face-located variable."""
    text = _UNIT_SQUARE.replace("E3T 1 1 2 3 1", "E3T 1 1 2 3 4").replace(
        "E3T 2 1 3 4 1", "E3T 2 1 3 4 7"
    )
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    assert "material" in dataset.data_variable_names
    material = dataset.get("material")
    assert material.location == "face"
    np.testing.assert_allclose(material.data, [4.0, 7.0])


def test_2dm_num_materials_per_elem_switches_parser(tmp_path):
    """NUM_MATERIALS_PER_ELEM makes the trailing columns materials, not aux."""
    text = """MESH2D
NUM_MATERIALS_PER_ELEM 2
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 7 9
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    np.testing.assert_allclose(dataset.get("material").data, [7.0])
    np.testing.assert_allclose(dataset.get("material_2").data, [9.0])


def test_2dm_e2l_only_file_yields_network1d(tmp_path):
    """A file of only E2L line elements is a 1D network."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 2.0 0.0 0.0
E2L 1 1 2 1
E2L 2 2 3 1
"""
    result = TwoDmDriver().read(_write(tmp_path, text))

    assert isinstance(result, Network1d)
    assert result.n_node == 3
    assert result.n_edge == 2
    np.testing.assert_array_equal(result.edge_node_connectivity.data, [[0, 1], [1, 2]])


_MIXED = _UNIT_SQUARE + "E2L 3 1 2 1\nE2L 4 2 3 1\n"


def test_2dm_mesh_names_lists_both_topologies_of_a_mixed_file(tmp_path):
    """A file carrying faces and E2L names its two topologies separately."""
    path = _write(tmp_path, _MIXED)

    assert TwoDmDriver().mesh_names(path) == ["mesh2d", "network1d"]


def test_2dm_mesh_names_falls_back_on_an_unreadable_path(tmp_path):
    """``mesh_names`` never raises: an unreadable path reports the surface name.

    pyramids calls ``mesh_names`` on every driver whatever it declares, so failing here
    would break dispatch for a file another driver could have read.
    """
    assert TwoDmDriver().mesh_names(str(tmp_path / "absent.2dm")) == ["mesh2d"]


def test_2dm_mesh_names_lists_only_what_the_file_holds(tmp_path):
    """A faces-only and an E2L-only file each name one topology."""
    faces_only = _write(tmp_path, _UNIT_SQUARE, name="faces.2dm")
    lines_only = _write(
        tmp_path,
        "MESH2D\nND 1 0.0 0.0 0.0\nND 2 1.0 0.0 0.0\nE2L 1 1 2 1\n",
        name="lines.2dm",
    )

    assert TwoDmDriver().mesh_names(faces_only) == ["mesh2d"]
    assert TwoDmDriver().mesh_names(lines_only) == ["network1d"]


def test_2dm_mixed_file_1d_elements_are_reachable(tmp_path):
    """The 1D elements of a mixed file are read by naming the network topology."""
    path = _write(tmp_path, _MIXED)

    network = TwoDmDriver().read(path, mesh_name="network1d")

    assert isinstance(network, Network1d)
    assert network.n_edge == 2
    np.testing.assert_array_equal(network.edge_node_connectivity.data, [[0, 1], [1, 2]])


def test_2dm_1d_read_of_a_mixed_file_reports_the_dropped_surface_mesh(tmp_path):
    """Reading the network half says the surface mesh is not in the result.

    The mirror of the surface-half warning. Without it the 2D topology -- the larger
    half of the file -- vanished without a word, while three docstrings promised
    otherwise.
    """
    path = _write(tmp_path, _MIXED)

    with pytest.warns(UserWarning, match="surface mesh") as record:
        network = TwoDmDriver().read(path, mesh_name="network1d")

    assert isinstance(network, Network1d)
    assert any("UgridDataset.read_file" in str(w.message) for w in record)


def test_2dm_mixed_file_1d_elements_reachable_through_network_read_file(tmp_path):
    """The documented route works: Network1d.read_file with the network's name."""
    path = _write(tmp_path, _MIXED)

    network = Network1d.read_file(path, mesh_name="network1d")

    assert isinstance(network, Network1d)
    assert network.n_edge == 2


def test_2dm_e2l_elements_never_become_the_surface_edges(tmp_path):
    """E2L is a separate 1D topology, so it never reaches ``edge_node_connectivity``.

    This is the whole point of keeping ``network_edges`` apart from ``edges``: feeding
    channel elements to ``Mesh2d.edge_node_connectivity`` would silently relabel them as
    the surface mesh's own edges, and every edge-located quantity derived from the mesh
    would then be indexed by the wrong topology. The mixed-file tests asserted the face
    count and the warning, but nothing asserted the mislabelling had not happened.
    """
    with pytest.warns(UserWarning, match="Network1d"):
        dataset = TwoDmDriver().read(_write(tmp_path, _MIXED))

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_face == 2
    assert dataset.mesh.edge_node_connectivity is None


def test_2dm_network_records_the_file_base_for_a_writer(tmp_path):
    """The 1D half records base 1 too, so a writer restores the file's numbering.

    ``original_start_index`` means "the base a writer should emit", and the surface half
    was pinned to 1 while the network half -- built through a different code path -- was
    not checked at all.
    """
    with pytest.warns(UserWarning, match="Network1d carries a topology"):
        network = TwoDmDriver().read(_write(tmp_path, _MIXED), mesh_name="network1d")

    connectivity = network.edge_node_connectivity
    assert connectivity.original_start_index == 1
    np.testing.assert_array_equal(connectivity.data, [[0, 1], [1, 2]])


def test_2dm_requesting_a_network_from_a_faces_only_file_raises(tmp_path):
    """Asking for a topology the file does not hold is an error naming what it has."""
    path = _write(tmp_path, _UNIT_SQUARE)

    with pytest.raises(ValueError, match="network1d"):
        TwoDmDriver().read(path, mesh_name="network1d")


def test_2dm_e2l_with_faces_warns(tmp_path):
    """A mixed-dimension file returns the faces and warns, naming Network1d."""
    text = _UNIT_SQUARE + "E2L 3 1 2 1\n"

    with pytest.warns(UserWarning, match="Network1d"):
        dataset = TwoDmDriver().read(_write(tmp_path, text))

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_face == 2


def test_2dm_e2l_only_file_reports_its_dropped_elevation(tmp_path):
    """An E2L-only file carrying real z values says so rather than losing them quietly.

    The original test used ``z = 0.0`` throughout, which hid the loss entirely.
    """
    text = """MESH2D
ND 1 0.0 0.0 5.0
ND 2 1.0 0.0 6.0
ND 3 2.0 0.0 7.0
E2L 1 1 2 1
E2L 2 2 3 1
"""
    with pytest.warns(UserWarning, match="node_z"):
        network = TwoDmDriver().read(_write(tmp_path, text))

    assert isinstance(network, Network1d)
    assert network.n_node == 3


def test_2dm_z_becomes_node_variable(tmp_path):
    """Node elevation is carried as a node variable, since the model is planar."""
    text = """MESH2D
ND 1 0.0 0.0 1.5
ND 2 1.0 0.0 2.5
ND 3 1.0 1.0 3.5
E3T 1 1 2 3 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    assert "node_z" in dataset.data_variable_names
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    np.testing.assert_allclose(node_z.data, [1.5, 2.5, 3.5])


def test_2dm_clockwise_face_is_reversed(tmp_path):
    """A clockwise face in the file comes back counter-clockwise."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 1.0 0.0
ND 3 1.0 0.0 0.0
E3T 1 1 2 3 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    assert _signed_area2(dataset.mesh, dataset.mesh.face_node_connectivity.data[0]) > 0


def test_2dm_can_open_false_on_empty_binary_and_directory(tmp_path):
    """can_open declines anything that is not a 2DM file, and never raises."""
    driver = TwoDmDriver()
    empty = tmp_path / "empty.2dm"
    empty.write_bytes(b"")
    binary = tmp_path / "binary.2dm"
    binary.write_bytes(bytes(range(256)) * 4)
    directory = tmp_path / "dir.2dm"
    directory.mkdir()
    foreign = tmp_path / "notes.txt"
    foreign.write_text("MESH2D\n", encoding="utf-8")

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(binary)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "missing.2dm")) is False
    # Right content, wrong suffix: the suffix gate still applies.
    assert driver.can_open(str(foreign)) is False
    assert driver.can_open(_write(tmp_path, _UNIT_SQUARE)) is True


def test_2dm_undefined_node_reference_raises(tmp_path):
    """An element naming a node id no ND record defines is an error, not a guess."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 99 1
"""
    with pytest.raises(ValueError, match="node id 99"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_committed_fixture_reads(tmp_path):
    """The committed minimal fixture reads, and matches build_minimal_fixture."""
    committed = Path(__file__).parent.parent / "data" / "minimal.2dm"
    dataset = TwoDmDriver().read(str(committed))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2

    generated = tmp_path / "generated.2dm"
    build_minimal_fixture(str(generated))
    assert generated.read_text(encoding="utf-8") == committed.read_text(
        encoding="utf-8"
    )


def test_2dm_has_no_crs(tmp_path):
    """2DM carries no CRS, and none is guessed from the coordinate magnitudes."""
    dataset = TwoDmDriver().read(_write(tmp_path, _UNIT_SQUARE))

    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


def test_2dm_header_may_follow_blank_and_comment_lines(tmp_path):
    """Leading blank and comment lines do not hide the MESH2D header."""
    text = "\n\n# written by some tool\n" + _UNIT_SQUARE
    path = _write(tmp_path, text)

    assert TwoDmDriver().can_open(path) is True
    assert TwoDmDriver().read(path).mesh.n_face == 2


def test_2dm_reads_a_file_saved_with_a_byte_order_mark(tmp_path):
    """A UTF-8 BOM, which Windows editors add routinely, does not hide the header."""
    path = tmp_path / "bom.2dm"
    path.write_text(_UNIT_SQUARE, encoding="utf-8-sig")

    assert TwoDmDriver().can_open(str(path)) is True
    assert TwoDmDriver().read(str(path)).mesh.n_face == 2


def test_2dm_header_found_past_a_long_comment_header(tmp_path):
    """A comment header longer than one sniff block does not make can_open disagree.

    The sniff reading a single fixed prefix declined files that ``read`` parsed fine,
    which is exactly the asymmetry the sniff is written to avoid.
    """
    path = _write(tmp_path, "# padding\n" * 500 + _UNIT_SQUARE)

    assert TwoDmDriver().can_open(path) is True
    assert TwoDmDriver().read(path).mesh.n_face == 2


@pytest.mark.parametrize("header", ["MESH2D", "MESH2DX"])
@pytest.mark.parametrize("offset", range(4088, 4100))
def test_2dm_sniff_agrees_with_read_across_a_block_boundary(tmp_path, header, offset):
    """``can_open`` agrees with ``read`` wherever the block edge falls.

    The sniff reads in blocks, and a verdict taken from the last element of
    ``splitlines()`` is a verdict taken from a line that is still being read. With the
    boundary inside the first significant line that declined valid ``MESH2D`` files and
    accepted ``MESH2DX`` truncated to exactly ``MESH2D`` -- both of the asymmetries the
    sniff exists to prevent, at a six-byte window every 4096 bytes.
    """
    padding = "#" + "x" * (offset - 2) + "\n"
    assert len(padding) == offset
    body = _UNIT_SQUARE.replace("MESH2D", header, 1)
    path = _write(tmp_path, padding + body)

    can_open = TwoDmDriver().can_open(path)
    try:
        TwoDmDriver().read(path)
        readable = True
    except FileFormatNotSupportedError:
        readable = False

    assert can_open is readable, f"offset {offset}, header {header}"


def test_2dm_can_open_declines_a_file_that_is_only_comments(tmp_path):
    """A file with no significant line at all is still declined, and bounded."""
    path = _write(tmp_path, "# nothing here\n" * 2000)

    assert TwoDmDriver().can_open(path) is False


def test_2dm_sniff_gives_up_at_its_limit(tmp_path, monkeypatch):
    """The sniff stops after its byte limit rather than reading an unbounded file.

    Both the block size and the limit are lowered here instead of writing a 256 KiB
    fixture; the behaviour under test is that the loop ends at the limit, not the size of
    either constant. The block must be smaller than the limit, or the very first read
    would swallow the whole file before the limit is ever consulted.
    """
    monkeypatch.setattr(sms, "_SNIFF_BLOCK", 16)
    monkeypatch.setattr(sms, "_SNIFF_LIMIT", 64)
    path = _write(tmp_path, "# padding padding padding\n" * 20 + _UNIT_SQUARE)

    assert TwoDmDriver().can_open(path) is False


def test_2dm_empty_file_is_refused(tmp_path):
    """A file with no significant records is refused, naming the missing header."""
    path = _write(tmp_path, "\n\n# nothing but comments\n")

    with pytest.raises(FileFormatNotSupportedError, match="is empty"):
        TwoDmDriver().read(path)


def test_2dm_file_without_elements_is_refused(tmp_path):
    """Nodes alone are not a mesh, so a file with no elements is refused."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
"""
    with pytest.raises(FileFormatNotSupportedError, match="declares no elements"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_unknown_keywords_are_ignored(tmp_path):
    """Records this driver does not model are skipped, not treated as an error.

    A real 2DM carries parameter blocks and metadata beside the topology; this driver
    ignores what it does not recognise.
    """
    text = """MESH2D
MESHNAME "my mesh"
BEGPARAMDEF
GM "Mesh"
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 1
ENDPARAMDEF
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_node == 3
    assert dataset.mesh.n_face == 1


def test_2dm_duplicate_node_id_is_refused(tmp_path):
    """Two ND records sharing an id make the connectivity ambiguous, so it raises."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 1 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 1 3 1
"""
    with pytest.raises(ValueError, match="duplicate native id"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_truncated_node_record_is_refused(tmp_path):
    """An ND record missing its coordinates is reported as a format error."""
    text = """MESH2D
ND 1 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 1
"""
    with pytest.raises(FileFormatNotSupportedError, match="ND"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_truncated_element_record_is_refused(tmp_path):
    """An element record short of its vertex ids is reported as a format error."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2
"""
    with pytest.raises(FileFormatNotSupportedError, match="E3T"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_file_without_a_z_column_carries_no_elevation(tmp_path):
    """A file declaring no z gets no ``node_z`` -- zeros would be invented data."""
    text = """MESH2D
ND 1 0.0 0.0
ND 2 1.0 0.0
ND 3 1.0 1.0
E3T 1 1 2 3 1
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    assert "node_z" not in dataset.data_variable_names


def test_2dm_face_without_material_defaults_to_zero(tmp_path):
    """An element record carrying no material column still reads."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    np.testing.assert_allclose(dataset.get("material").data, [0.0])


def test_2dm_aux_columns_do_not_shift_the_material(tmp_path):
    """Trailing aux columns are ignored; the material is indexed from the start."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 5 111 222 333
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    np.testing.assert_allclose(dataset.get("material").data, [5.0])
    assert "material_2" not in dataset.data_variable_names


def test_2dm_keywords_are_matched_case_insensitively(tmp_path):
    """A file written in lower case reads, and is sniffed, exactly as upper case does.

    Every keyword is folded with ``.upper()`` in the sniff, the record loop and the
    element dispatch alike, and every other fixture is upper case, so a fold dropped
    from any one of the three would read as working.
    """
    path = _write(tmp_path, _UNIT_SQUARE.lower(), name="lower.2dm")

    assert TwoDmDriver().can_open(path) is True
    dataset = TwoDmDriver().read(path)

    assert dataset.mesh.n_face == 2
    np.testing.assert_array_equal(
        dataset.mesh.face_node_connectivity.data, [[0, 1, 2], [0, 2, 3]]
    )


def test_2dm_num_materials_per_elem_may_follow_the_elements(tmp_path):
    """The tag applies to elements already read, which is why the split is deferred.

    Face columns are stored verbatim and only divided once the whole file has been
    seen, precisely so a writer that puts the tag at the end is read correctly. Every
    existing fixture declares it first, so the deferral itself was untested.
    """
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 7 9
NUM_MATERIALS_PER_ELEM 2
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    np.testing.assert_allclose(dataset.get("material").data, [7.0])
    np.testing.assert_allclose(dataset.get("material_2").data, [9.0])


def test_2dm_element_short_of_a_declared_material_column_defaults_to_zero(tmp_path):
    """A missing trailing column contributes 0.0 rather than failing the read.

    Writers omit a trailing column routinely, and refusing an otherwise sound file over
    it would be worse than carrying the gap.
    """
    text = """MESH2D
NUM_MATERIALS_PER_ELEM 2
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 7
"""
    dataset = TwoDmDriver().read(_write(tmp_path, text))

    np.testing.assert_allclose(dataset.get("material").data, [7.0])
    np.testing.assert_allclose(dataset.get("material_2").data, [0.0])


def test_2dm_duplicate_face_id_is_refused(tmp_path):
    """Two E3T records sharing an id is refused, exactly as two nodes or two E2L are.

    Nothing references an element by id, so the check exists only to refuse a malformed
    file; the node and line-element halves were tested and the face half was not.
    """
    text = _UNIT_SQUARE.replace("E3T 2 1 3 4 1", "E3T 1 1 3 4 1")

    with pytest.raises(ValueError, match="duplicate native id 1"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_truncated_line_element_record_is_refused(tmp_path):
    """An E2L record short of its two vertex ids is reported with its keyword."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
E2L 1 1
"""
    with pytest.raises(FileFormatNotSupportedError, match="malformed E2L record"):
        TwoDmDriver().read(_write(tmp_path, text))


def test_2dm_crlf_line_endings_read(tmp_path):
    """A file written on Windows reads: the carriage return is stripped, not parsed.

    Left in place it would ride along on the last field of every record and turn each
    numeric parse into a format error.
    """
    path = tmp_path / "crlf.2dm"
    path.write_bytes(_UNIT_SQUARE.replace("\n", "\r\n").encode("utf-8"))

    assert TwoDmDriver().can_open(str(path)) is True
    dataset = TwoDmDriver().read(str(path))

    assert dataset.mesh.n_face == 2
    np.testing.assert_allclose(dataset.get("material").data, [1.0, 1.0])


@pytest.mark.parametrize("header, expected", [("MESH2D", True), ("MESH3D", False)])
def test_2dm_sniff_decides_a_final_line_with_no_trailing_newline(
    tmp_path, header, expected
):
    """The remainder carried past the last newline is judged at end of file.

    The sniff only ever judges complete lines, so the last line of a file that does
    not end in a newline is decided by the end-of-file arm alone. The one test that
    reaches that arm is the all-comments file, whose verdict is "undecided" either
    way -- so dropping the end-of-file judgement entirely, and declining every such
    file, would have passed the suite.
    """
    path = tmp_path / "noeol.2dm"
    path.write_bytes(f"# written by some tool\n{header}".encode("utf-8"))

    assert TwoDmDriver().can_open(str(path)) is expected


def test_2dm_header_line_may_carry_trailing_fields(tmp_path):
    """The sniff compares the header *token*, exactly as the parser compares it.

    ``_Scan.run`` tests ``keyword``, so a line reading ``MESH2D`` followed by anything
    else is still the header. A sniff comparing the whole stripped line would decline
    such a file and ``read`` would then parse it fine -- the mirror of the ``MESH2DX``
    case and the same asymmetry, in the other direction.
    """
    path = _write(tmp_path, _UNIT_SQUARE.replace("MESH2D", "MESH2D extra words", 1))

    assert TwoDmDriver().can_open(path) is True
    assert TwoDmDriver().read(path).mesh.n_face == 2


def test_2dm_mesh_names_folds_lower_case_element_keywords(tmp_path):
    """``mesh_names`` upper-cases its own keywords, or a lower-case file loses its 1D half.

    The fold here is separate from the parser's. On a faces-only file a dropped fold
    is invisible -- the ``or`` fallback returns the surface name regardless -- but on a
    lower-case ``e2l`` file it would answer ``['mesh2d']``, and ``read`` would then
    refuse ``mesh_name='network1d'`` for a topology the file plainly holds.
    """
    path = _write(
        tmp_path,
        "mesh2d\nnd 1 0.0 0.0 0.0\nnd 2 1.0 0.0 0.0\ne2l 1 1 2 1\n",
        name="lower1d.2dm",
    )
    driver = TwoDmDriver()

    assert driver.mesh_names(path) == ["network1d"]

    with pytest.warns(UserWarning, match="node_z"):
        network = driver.read(path, mesh_name="network1d")

    assert isinstance(network, Network1d)
    np.testing.assert_array_equal(network.edge_node_connectivity.data, [[0, 1]])


@pytest.mark.parametrize(
    "declared, accepted",
    [
        pytest.param(1, True, id="lower-bound"),
        pytest.param(16, True, id="the-cap-itself"),
        pytest.param(17, False, id="one-past-the-cap"),
    ],
)
def test_2dm_material_column_bounds_are_inclusive(tmp_path, declared, accepted):
    """``1 <= declared <= 16`` is checked at both ends, and both ends are inclusive.

    The guard is proved from below (``0`` and ``-1``) and from far above (``20000``),
    so neither edge is pinned. Written ``< 16`` the documented maximum would itself be
    refused, and written ``<= 0`` a declaration of ``1`` would be.
    """
    columns = " ".join(str(number) for number in range(1, 18))
    text = f"""MESH2D
NUM_MATERIALS_PER_ELEM {declared}
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
E3T 1 1 2 3 {columns}
"""
    path = _write(tmp_path, text)

    if not accepted:
        with pytest.raises(
            FileFormatNotSupportedError, match="not a usable column count"
        ):
            TwoDmDriver().read(path)
        return

    dataset = TwoDmDriver().read(path)
    materials = {
        name for name in dataset.data_variable_names if name.startswith("material")
    }
    assert len(materials) == declared, f"got {sorted(materials)}"


def test_2dm_binary_content_is_a_format_error_not_a_decode_error(tmp_path):
    """Undecodable bytes are replaced, so a binary file fails as a format error.

    ``iter_records`` decodes with ``errors="replace"`` for exactly this: a
    ``UnicodeDecodeError`` escaping ``read`` is not in the driver contract, and
    ``can_open`` declining such a file was the only half ever asserted -- it reads the
    bytes through a different path and would still decline if ``read`` blew up.
    """
    path = tmp_path / "binary.2dm"
    path.write_bytes(bytes(range(256)) * 4)

    with pytest.raises(FileFormatNotSupportedError, match="is not a 2DM file"):
        TwoDmDriver().read(str(path))


def test_2dm_mesh_names_is_a_single_entry(tmp_path):
    """2DM holds one topology, so ``mesh_names`` returns one name."""
    assert TwoDmDriver().mesh_names(_write(tmp_path, _UNIT_SQUARE)) == ["mesh2d"]


def test_2dm_is_reachable_through_read_file(tmp_path):
    """The registered entry point dispatches a .2dm path to this driver."""
    dataset = UgridDataset.read_file(_write(tmp_path, _UNIT_SQUARE))

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_face == 2


def test_2dm_network_is_reachable_through_network_read_file(tmp_path):
    """An E2L-only file is read through ``Network1d.read_file``."""
    text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
E2L 1 1 2 1
"""
    network = Network1d.read_file(_write(tmp_path, text))

    assert isinstance(network, Network1d)
    assert network.n_edge == 1


class TestTwoDmWriter:
    """Tests for the 2DM writer (W1) — the round trip against the reader."""

    def _roundtrip(self, tmp_path, text, **options):
        """Read ``text``, write it back through the driver, and read the result."""
        src = _write(tmp_path, text, "in.2dm")
        dataset = TwoDmDriver().read(src)
        out = tmp_path / "out.2dm"
        dataset.to_file(str(out), driver="2dm", **options)
        return dataset, TwoDmDriver().read(str(out)), out

    def test_round_trip_preserves_coordinates_exactly(self, tmp_path):
        """Projected coordinates survive the round trip bit-for-bit at full precision."""
        text = """MESH2D
ND 1 500000.123456789 4000000.5 0.0
ND 2 500001.0 4000000.5 0.0
ND 3 500001.0 4000001.0 0.0
E3T 1 1 2 3 1
"""
        before, after, _ = self._roundtrip(tmp_path, text)
        assert np.array_equal(before.mesh.node_x, after.mesh.node_x)
        assert np.array_equal(before.mesh.node_y, after.mesh.node_y)

    def test_round_trip_preserves_connectivity_and_start_index(self, tmp_path):
        """Connectivity and the recorded 1-based numbering round-trip unchanged."""
        before, after, _ = self._roundtrip(tmp_path, _UNIT_SQUARE)
        bc = before.mesh.face_node_connectivity
        ac = after.mesh.face_node_connectivity
        assert ac.data.tolist() == bc.data.tolist()
        assert ac.original_start_index == bc.original_start_index == 1

    def test_round_trip_mixed_element_types(self, tmp_path):
        """A quad and a triangle each keep their element type across the round trip."""
        text = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
ND 5 2.0 0.5 0.0
E4Q 1 1 2 3 4 1
E3T 2 2 5 3 1
"""
        _, after, out = self._roundtrip(tmp_path, text)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert any(line.startswith("E4Q") for line in lines)
        assert any(line.startswith("E3T") for line in lines)
        rows = after.mesh.face_node_connectivity.data.tolist()
        assert rows[0] == [0, 1, 2, 3]
        assert rows[1] == [1, 4, 2, -1]

    def test_write_full_precision_by_default(self, tmp_path):
        """The default write keeps every digit of a float64 coordinate."""
        text = "MESH2D\nND 1 1.0000000000000002 0.0 0.0\nND 2 1.0 0.0 0.0\n"
        text += "ND 3 1.0 1.0 0.0\nE3T 1 1 2 3 1\n"
        before, after, _ = self._roundtrip(tmp_path, text)
        assert float(after.mesh.node_x[0]) == 1.0000000000000002

    def test_write_precision_option_rounds(self, tmp_path):
        """An integer ``precision`` option limits the significant digits written."""
        text = "MESH2D\nND 1 500000.123456 0.0 0.0\nND 2 1.0 0.0 0.0\n"
        text += "ND 3 1.0 1.0 0.0\nE3T 1 1 2 3 1\n"
        src = _write(tmp_path, text, "in.2dm")
        out = tmp_path / "out.2dm"
        TwoDmDriver().read(src).to_file(str(out), driver="2dm", precision=6)
        nd1 = [ln for ln in out.read_text().splitlines() if ln.startswith("ND 1")][0]
        assert "500000" in nd1 and "123456" not in nd1

    def test_write_node_z_from_variable(self, tmp_path):
        """A node_z variable is written as the ND z column and round-trips."""
        text = """MESH2D
ND 1 0.0 0.0 2.5
ND 2 1.0 0.0 3.5
ND 3 1.0 1.0 4.5
E3T 1 1 2 3 1
"""
        before, after, _ = self._roundtrip(tmp_path, text)
        assert after.get("node_z").data.tolist() == [2.5, 3.5, 4.5]

    def test_write_without_material_writes_zero(self, tmp_path):
        """A mesh with no material variable writes a 0 material column."""
        # a frame with z but element lines lacking a material column -> no material var
        text = "MESH2D\nND 1 0.0 0.0 0.0\nND 2 1.0 0.0 0.0\nND 3 1.0 1.0 0.0\nE3T 1 1 2 3\n"
        src = _write(tmp_path, text, "in.2dm")
        dataset = TwoDmDriver().read(src)
        out = tmp_path / "out.2dm"
        dataset.to_file(str(out), driver="2dm")
        e3t = [ln for ln in out.read_text().splitlines() if ln.startswith("E3T")][0]
        assert e3t.split()[-1] == "0"

    def test_write_declares_write_capabilities(self):
        """The driver advertises write_mesh and write_data so to_file dispatches to it."""
        caps = set(TwoDmDriver.capabilities)
        assert {"write_mesh", "write_data"} <= caps

    def test_e2l_only_file_reads_as_a_network_not_a_mesh(self, tmp_path):
        """An E2L-only (1D link) file reads as a Network1d, which has no faces to write."""
        text = "MESH2D\nND 1 0.0 0.0 0.0\nND 2 1.0 0.0 0.0\nE2L 1 1 2 1\n"
        src = _write(tmp_path, text, "net.2dm")
        result = TwoDmDriver().read(src)
        assert isinstance(result, Network1d)

    @pytest.mark.parametrize(
        "connectivity",
        [
            None,  # mesh carries no face_node_connectivity at all
            type("_NoData", (), {"data": None})(),  # connectivity present but empty
        ],
    )
    def test_write_without_face_connectivity_is_refused(self, tmp_path, connectivity):
        """The writer refuses a faces-less dataset, naming the missing connectivity.

        Exercises the guard directly (the E2L-only read above only proves the *read*
        yields a faces-less object); a dataset whose mesh has no face connectivity -- or a
        connectivity whose data is ``None`` -- cannot be written as a 2DM mesh.
        """
        mesh = type("_Mesh", (), {"face_node_connectivity": connectivity})()
        dataset = type("_Dataset", (), {"mesh": mesh})()
        with pytest.raises(FileFormatNotSupportedError, match="no face connectivity"):
            TwoDmDriver().write(dataset, str(tmp_path / "out.2dm"))
