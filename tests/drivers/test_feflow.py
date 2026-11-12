"""Tests for the DHI FEFLOW classic ASCII ``.fem`` reader, against a committed sample.

A real FEFLOW mesh is committed under ``tests/data/feflow/``: ``disv_nodes.fem`` is a 2D
mesh of 4842 triangles on 2496 nodes (``MODFLOW-ORG/modflow6-examples``, ``ex-gwe-ates``,
CC0 / public domain). ``binary_block.fem`` is a FEFLOW v6.215 deck whose mesh lives in a
``BINARY`` block (``pham-hung/Subsidence_Feflow``, MIT, truncated) -- the reader must
decline it with an accurate message rather than mis-read it as a missing ``NODE`` section
(issue #96). The ``.fem`` suffix is shared with the Nastran bulk-data format, so a Nastran
deck (committed under ``tests/data/nastran/``) must be declined by ``can_open``. Tiny inline
decks in ``tmp_path`` cover the 3D, binary and defensive branches the committed meshes do
not reach.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import pytest
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MeshDataset, _drivers

from pyramids_mesh.drivers.feflow import FeflowDriver, _FeflowReader

_DATA = Path(__file__).parent.parent / "data"
_FEM = _DATA / "feflow" / "disv_nodes.fem"
_BINARY_FEM = _DATA / "feflow" / "binary_block.fem"
_NASTRAN = _DATA / "nastran" / "cylinder.fem"


def _read(path):
    """Read a committed .fem file, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FeflowDriver().read(str(path))


def test_reads_the_triangular_mesh():
    """disv_nodes.fem reads 4842 triangles on 2496 nodes."""
    mesh = _read(_FEM).mesh
    assert mesh.n_node == 2496
    assert mesh.n_face == 4842
    assert mesh.face_node_connectivity.data.shape == (4842, 3)


def test_node_ids_are_one_based_resolved_to_zero_based():
    """The 1-based NODE ids are resolved to 0-based, and the base recorded."""
    connectivity = _read(_FEM).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 2495  # 2496 nodes -> 0..2495
    # the first NODE row is "175 82 68", 1-based -> {174, 81, 67} after re-basing
    assert set(connectivity.data[0].tolist()) == {174, 81, 67}


def test_scale_is_applied_to_the_coordinates():
    """COOR values are mapped by the per-axis SCALE (scale, offset) pairs."""
    mesh = _read(_FEM).mesh
    # SCALE is "0.78125, 1000, 1, 1, 0, 0"; the first node's raw x is 135.0, raw y 0.0.
    assert mesh.node_x[0] == pytest.approx(135.0 * 0.78125 + 1000.0)
    assert mesh.node_y[0] == pytest.approx(0.0 * 1.0 + 1.0)


def test_a_2d_mesh_has_no_elevation_and_no_crs():
    """COOR carries two coordinates per node, so node_z is absent, and crs is None."""
    dataset = _read(_FEM)
    assert dataset.mesh.crs is None
    assert dataset.get("node_z") is None


def test_can_open_accepts_the_feflow_file():
    """can_open accepts the real FEFLOW file on its PROBLEM:/CLASS signature."""
    assert FeflowDriver().can_open(str(_FEM)) is True


def test_can_open_declines_a_nastran_deck():
    """The .fem suffix is shared with Nastran; a Nastran bulk-data deck is declined."""
    assert FeflowDriver().can_open(str(_NASTRAN)) is False


def test_can_open_declines_junk_and_a_directory(tmp_path):
    """can_open declines a .fem whose first statements are not PROBLEM:/CLASS."""
    driver = FeflowDriver()
    junk = tmp_path / "fake.fem"
    junk.write_text("not a feflow mesh\nnor a class line\n")
    assert driver.can_open(str(junk)) is False
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False
    assert driver.can_open(str(tmp_path)) is False


def test_dispatch_through_read_file_by_driver_name():
    """read_file routes to this driver when it is registered and named."""
    original = _drivers._CACHE
    try:
        registry = dict(original or {})
        registry["feflow"] = FeflowDriver()
        _drivers._CACHE = registry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = MeshDataset.read_file(str(_FEM), driver="feflow")
    finally:
        _drivers._CACHE = original
    assert dataset.mesh.n_face == 4842
    assert dataset.mesh.n_node == 2496


_HEADER = "PROBLEM:\nCLASS (v.7.501)\n   5    1\n"


def _fem_text(
    dimens="DIMENS\n  3   1   3\n",
    scale="SCALE\n 1.0, 0.0, 1.0, 0.0, 1.0, 0.0\n",
    node="NODE\n  1 2 3\n",
    coor="COOR\n 0.0, 1.0, 0.0, 0.0, 0.0, 1.0\nEND\n",
):
    """Assemble a tiny valid 2D .fem body; override one section to malform it."""
    return f"{_HEADER}{dimens}{scale}{node}{coor}"


def test_a_tiny_inline_2d_mesh_reads(tmp_path):
    """The section scanner builds a one-triangle mesh from an inline deck."""
    fem_path = tmp_path / "tri.fem"
    fem_path.write_text(_fem_text())
    parsed = _FeflowReader(str(fem_path)).build()
    assert parsed.faces.shape == (1, 3)
    assert parsed.node_x.tolist() == [0.0, 1.0, 0.0]
    assert parsed.node_y.tolist() == [0.0, 0.0, 1.0]
    assert parsed.node_z is None
    assert parsed.global_attributes["source"] == "DHI FEFLOW .fem"


def test_a_3d_deck_carries_node_z(tmp_path):
    """A COOR stream with three coordinates per node yields node_z."""
    fem = (
        f"{_HEADER}"
        "DIMENS\n  4   1   4\n"
        "SCALE\n 1.0, 0.0, 1.0, 0.0, 1.0, 0.0\n"
        "NODE\n  1 2 3 4\n"
        "COOR\n 0.0,1.0,1.0,0.0, 0.0,0.0,1.0,1.0, 0.0,1.0,2.0,3.0\nEND\n"
    )
    fem_path = tmp_path / "tet.fem"
    fem_path.write_text(fem)
    parsed = _FeflowReader(str(fem_path)).build()
    assert parsed.faces.shape == (1, 4)
    assert parsed.node_z is not None
    assert parsed.node_z.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_missing_dimens_section_is_a_format_error(tmp_path):
    """A deck without DIMENS cannot be sized and is a clean format error."""
    fem_path = tmp_path / "no_dimens.fem"
    fem_path.write_text(f"{_HEADER}NODE\n  1 2 3\nCOOR\n 0,1,0,0,0,1\nEND\n")
    with pytest.raises(FileFormatNotSupportedError, match="no DIMENS section"):
        _FeflowReader(str(fem_path)).build()


def test_a_short_dimens_row_is_a_format_error(tmp_path):
    """A DIMENS row with fewer than three integers is a clean format error."""
    fem_path = tmp_path / "short.fem"
    fem_path.write_text(_fem_text(dimens="DIMENS\n  3   1\n"))
    with pytest.raises(FileFormatNotSupportedError, match="<n_nodes> <n_elements>"):
        _FeflowReader(str(fem_path)).build()


def test_a_node_count_mismatch_is_a_format_error(tmp_path):
    """A DIMENS element count that disagrees with the NODE rows is a format error."""
    fem_path = tmp_path / "mismatch.fem"
    fem_path.write_text(_fem_text(dimens="DIMENS\n  3   2   3\n"))
    with pytest.raises(FileFormatNotSupportedError, match="but the NODE section has 1"):
        _FeflowReader(str(fem_path)).build()


def test_a_coor_stream_that_is_not_a_node_multiple_is_a_format_error(tmp_path):
    """A COOR length that is not 2x or 3x the node count is a clean format error."""
    fem_path = tmp_path / "bad_coor.fem"
    # 3 nodes but 7 coordinates: neither 6 (2D) nor 9 (3D)
    fem_path.write_text(_fem_text(coor="COOR\n 0,1,0,0,0,1,0\nEND\n"))
    with pytest.raises(FileFormatNotSupportedError, match="whole multiple"):
        _FeflowReader(str(fem_path)).build()


def test_a_missing_coor_section_is_a_format_error(tmp_path):
    """A deck without COOR has no coordinates and is a clean format error."""
    fem_path = tmp_path / "no_coor.fem"
    fem_path.write_text(f"{_HEADER}DIMENS\n  3   1   3\nNODE\n  1 2 3\nEND\n")
    with pytest.raises(FileFormatNotSupportedError, match="no COOR section"):
        _FeflowReader(str(fem_path)).build()


def test_read_of_a_non_fem_path_is_a_format_error(tmp_path):
    """read() on a path that is not a file is refused as a format error."""
    missing = tmp_path / "gone.fem"
    with pytest.raises(FileFormatNotSupportedError, match="not a FEFLOW .fem file"):
        FeflowDriver().read(str(missing))


def test_sniff_declines_a_path_that_cannot_be_opened(tmp_path):
    """_sniff returns False on an OS read error; it never raises."""
    assert FeflowDriver()._sniff(str(tmp_path)) is False


def test_sniff_skips_leading_blank_lines(tmp_path):
    """Blank lines before the PROBLEM:/CLASS signature are skipped when sniffing."""
    fem_path = tmp_path / "blank_lead.fem"
    fem_path.write_text("\n   \nPROBLEM:\nCLASS (v.7.501)\n   5    1\n")
    assert FeflowDriver()._sniff(str(fem_path)) is True


def test_read_sections_drops_data_before_the_first_header(tmp_path):
    """A data row before any keyword header has no current section and is dropped."""
    fem_path = tmp_path / "stray.fem"
    fem_path.write_text("  12345\nPROBLEM:\nCLASS\n   5   1\n")
    sections = _FeflowReader(str(fem_path))._read_sections()
    assert "PROBLEM" in sections
    assert all("12345" not in rows for rows in sections.values())


def test_sniff_declines_a_file_with_too_few_statements(tmp_path):
    """A .fem with fewer than two non-blank statements fails the signature check."""
    fem_path = tmp_path / "one_line.fem"
    fem_path.write_text("PROBLEM:\n")
    assert FeflowDriver()._sniff(str(fem_path)) is False


def test_read_sections_reports_an_unreadable_path(tmp_path):
    """_read_sections turns an OS read error into a clean format error."""
    with pytest.raises(FileFormatNotSupportedError, match="could not be read"):
        _FeflowReader(str(tmp_path)).build()


def test_a_sub_triangular_valence_is_a_format_error(tmp_path):
    """A DIMENS valence below three cannot describe an element and is refused."""
    fem_path = tmp_path / "valence.fem"
    fem_path.write_text(_fem_text(dimens="DIMENS\n  3   1   2\n"))
    with pytest.raises(FileFormatNotSupportedError, match="at least three nodes"):
        _FeflowReader(str(fem_path)).build()


def test_a_missing_node_section_is_a_format_error(tmp_path):
    """A deck without a NODE section has no connectivity and is a format error."""
    fem_path = tmp_path / "no_node.fem"
    fem_path.write_text(
        f"{_HEADER}DIMENS\n  3   1   3\nSCALE\n 1,0,1,0,1,0\nCOOR\n 0,1,0,0,0,1\nEND\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="no NODE section"):
        _FeflowReader(str(fem_path)).build()


def test_a_non_numeric_node_row_is_a_format_error(tmp_path):
    """A NODE row that is not integer node ids is a clean format error."""
    fem_path = tmp_path / "bad_node.fem"
    fem_path.write_text(_fem_text(node="NODE\n  1 2 x\n"))
    with pytest.raises(FileFormatNotSupportedError, match="non-numeric"):
        _FeflowReader(str(fem_path)).build()


def test_a_node_row_without_ids_is_a_format_error(tmp_path):
    """A NODE row carrying only separators resolves to no ids and is refused."""
    fem_path = tmp_path / "empty_node.fem"
    fem_path.write_text(_fem_text(node="NODE\n ,,,\n"))
    with pytest.raises(FileFormatNotSupportedError, match="no node ids"):
        _FeflowReader(str(fem_path)).build()


def test_a_non_numeric_coor_stream_is_a_format_error(tmp_path):
    """A COOR stream with non-numeric tokens is a clean format error."""
    fem_path = tmp_path / "bad_coor_stream.fem"
    fem_path.write_text(_fem_text(coor="COOR\n 0 1 x 0 0 1\nEND\n"))
    with pytest.raises(FileFormatNotSupportedError, match="non-numeric"):
        _FeflowReader(str(fem_path)).build()


def test_a_coor_implying_one_coordinate_per_node_is_a_format_error(tmp_path):
    """A COOR length that implies neither 2D nor 3D coordinates is refused."""
    fem_path = tmp_path / "one_dim.fem"
    fem_path.write_text(
        f"{_HEADER}DIMENS\n  1   1   3\nSCALE\n 1,0,1,0,1,0\n"
        "NODE\n 1 1 1\nCOOR\n 5.0\nEND\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="only 2D and 3D"):
        _FeflowReader(str(fem_path)).build()


# -- binary FEFLOW .fem (issue #96): a v6/v7 deck storing its mesh in a BINARY block ------


def test_a_binary_fem_is_declined_with_an_explicit_message():
    """A real binary FEFLOW .fem is refused as a binary deck, not a missing NODE section.

    ``binary_block.fem`` is a FEFLOW v6.215 deck whose mesh lives in a ``BINARY`` block
    (``pham-hung/Subsidence_Feflow``, MIT, truncated). ``can_open`` passes on its ASCII
    ``PROBLEM:`` / ``CLASS`` header, so the reader itself must reject it with an accurate
    message instead of failing later as a missing ``NODE`` section.
    """
    reader = _FeflowReader(str(_BINARY_FEM))
    with pytest.raises(FileFormatNotSupportedError, match="is a binary FEFLOW"):
        reader.build()


def test_a_binary_fem_is_declined_through_the_driver_read():
    """The public read() path also refuses the binary deck with the binary message."""
    with pytest.raises(FileFormatNotSupportedError, match="is a binary FEFLOW"):
        FeflowDriver().read(str(_BINARY_FEM))


def test_a_binary_fem_does_not_raise_the_misleading_no_node_message():
    """The old, misleading 'no NODE section' message must not surface for a binary deck."""
    reader = _FeflowReader(str(_BINARY_FEM))
    with pytest.raises(FileFormatNotSupportedError) as excinfo:
        reader.build()
    assert "no NODE section" not in str(excinfo.value)


def test_can_open_still_accepts_a_binary_fem():
    """can_open accepts the binary deck (it is a FEFLOW file); the reader declines it.

    Keeping ``can_open`` True means dispatch still routes a binary ``.fem`` to this driver
    and so surfaces the explicit binary message, rather than a generic 'no driver' error.
    """
    assert FeflowDriver().can_open(str(_BINARY_FEM)) is True


def test_an_inline_binary_block_deck_is_declined(tmp_path):
    """A ``BINARY`` section header plus NUL bytes is refused as a binary deck."""
    fem_path = tmp_path / "inline_binary.fem"
    fem_path.write_bytes(
        b"PROBLEM: A new FEFLOW problem\nCLASS (v.6.215)\n   2    1\n"
        b"DIMENS\n  4   1   3\nBINARY FEM 1.0\n" + b"\x00" * 16
    )
    reader = _FeflowReader(str(fem_path))
    with pytest.raises(FileFormatNotSupportedError, match="is a binary FEFLOW"):
        reader.build()


def test_nul_bytes_alone_decline_a_deck_without_a_binary_header(tmp_path):
    """NUL bytes make a deck binary even when no ``BINARY`` header is written."""
    fem_path = tmp_path / "nul_only.fem"
    fem_path.write_bytes(
        b"PROBLEM:\nCLASS (v.7.0)\n   2    1\nDIMENS\n  3   1   3\n"
        b"NODE\n  1 2 3\x00\x00\x00\nEND\n"
    )
    reader = _FeflowReader(str(fem_path))
    with pytest.raises(FileFormatNotSupportedError, match="is a binary FEFLOW"):
        reader.build()


def test_a_binary_section_header_without_nul_bytes_is_declined(tmp_path):
    """A ``BINARY`` section header alone (no NUL bytes) is still refused."""
    fem_path = tmp_path / "binary_header.fem"
    fem_path.write_text(
        f"{_HEADER}DIMENS\n  3   1   3\nBINARY FEM 1.0\n some ascii payload\nEND\n"
    )
    reader = _FeflowReader(str(fem_path))
    with pytest.raises(FileFormatNotSupportedError, match="is a binary FEFLOW"):
        reader.build()


def test_an_ascii_deck_with_a_binary_lookalike_header_still_reads(tmp_path):
    """Only an exact ``BINARY`` section header triggers the guard, not a lookalike.

    A ``BINARYLABEL`` header shares the ``BINARY`` prefix but is a different keyword, so a
    clean ASCII deck carrying it must still read rather than be refused as binary.
    """
    fem_path = tmp_path / "ascii_binaryish.fem"
    fem_path.write_text(
        f"{_HEADER}DIMENS\n  3   1   3\nSCALE\n 1,0,1,0,1,0\n"
        "BINARYLABEL\n metadata row\n"
        "NODE\n  1 2 3\nCOOR\n 0.0, 1.0, 0.0, 0.0, 0.0, 1.0\nEND\n"
    )
    parsed = _FeflowReader(str(fem_path)).build()
    assert parsed.faces.shape == (1, 3)


def test_the_nul_only_message_names_nul_bytes_not_a_block(tmp_path):
    """A NUL-only deck is reported via its NUL bytes, not a (non-existent) BINARY block."""
    fem_path = tmp_path / "nul_only_msg.fem"
    fem_path.write_bytes(
        b"PROBLEM:\nCLASS (v.7.0)\n   2    1\nDIMENS\n  3   1   3\n"
        b"NODE\n  1 2 3\x00\x00\x00\nEND\n"
    )
    reader = _FeflowReader(str(fem_path))
    with pytest.raises(FileFormatNotSupportedError) as excinfo:
        reader.build()
    message = str(excinfo.value)
    assert "NUL bytes" in message, message
    assert "block" not in message, message


def test_the_binary_header_message_does_not_claim_nul_bytes(tmp_path):
    """A BINARY-header deck with no NUL bytes is reported via the section, not NUL bytes."""
    fem_path = tmp_path / "binary_header_msg.fem"
    fem_path.write_text(
        f"{_HEADER}DIMENS\n  3   1   3\nBINARY FEM 1.0\n some ascii payload\nEND\n"
    )
    reader = _FeflowReader(str(fem_path))
    with pytest.raises(FileFormatNotSupportedError) as excinfo:
        reader.build()
    message = str(excinfo.value)
    assert "BINARY section" in message, message
    assert "NUL" not in message, message


def test_a_control_byte_in_a_data_row_does_not_split_the_row(tmp_path):
    """A stray form-feed inside a NODE row stays one row (universal-newline splitting).

    `str.splitlines()` would split on `\\x0c`, turning one NODE row into two and tripping the
    element-count check; the reader must match the old universal-newline `readlines()` and
    keep it a single row.
    """
    fem_path = tmp_path / "formfeed.fem"
    fem_path.write_text(
        f"{_HEADER}DIMENS\n  3   1   3\nSCALE\n 1,0,1,0,1,0\n"
        "NODE\n  1 2\x0c3\nCOOR\n 0.0, 1.0, 0.0, 0.0, 0.0, 1.0\nEND\n"
    )
    parsed = _FeflowReader(str(fem_path)).build()
    assert parsed.faces.shape == (1, 3)
    assert set(parsed.faces[0].tolist()) == {1, 2, 3}
