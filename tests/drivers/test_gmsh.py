"""Tests for the Gmsh ``.msh`` reader (C3).

The committed fixtures are the same mesh written by Gmsh in both supported layouts --
``insulated-2.2.msh`` (the legacy record form) and ``insulated-4.1.msh`` (the entity-block
form) -- so the central assertion is that the two layouts read to the *same* 67 nodes and
111 triangular faces. Alongside that: boundary lines are dropped, winding is normalised,
dispatch finds the driver by suffix, and junk is declined.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset, _drivers

from pyramids_mesh.drivers.gmsh import GmshDriver, _as_float, _as_int

_DATA = Path(__file__).parent.parent / "data" / "gmsh"
_V22 = str(_DATA / "insulated-2.2.msh")
_V41 = str(_DATA / "insulated-4.1.msh")

_EXPECTED_NODES = 67
_EXPECTED_FACES = 111


def _write(tmp_path, text, name="mesh.msh"):
    """Write a Gmsh fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _signed_area2(mesh, row):
    """Twice the signed area of one face of a built mesh (CCW is positive)."""
    real = row[row != -1]
    xs, ys = mesh.node_x[real], mesh.node_y[real]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


@pytest.fixture
def gmsh_dispatch():
    """Register the Gmsh driver into the discovery cache for the duration of a test.

    The driver is not in the installed entry points, so ``read_file`` would not find it
    on its own; injecting it into the cache lets a dispatch test exercise the real
    suffix/``can_open`` routing without a reinstall. The cache is cleared afterwards so no
    other test inherits the injected driver.
    """
    _drivers.clear_driver_cache()
    registry = dict(_drivers.mesh_drivers())
    registry.setdefault("gmsh", GmshDriver())
    _drivers._CACHE = registry
    try:
        yield
    finally:
        _drivers.clear_driver_cache()


@pytest.mark.parametrize("path", [_V22, _V41])
def test_gmsh_reads_expected_counts(path):
    """Each supported layout reads to 67 nodes and 111 triangular faces."""
    dataset = GmshDriver().read(path)

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_node == _EXPECTED_NODES
    assert dataset.mesh.n_face == _EXPECTED_FACES


def test_gmsh_v22_and_v41_read_the_same_mesh():
    """The two layouts are the same mesh, so nodes and connectivity must match exactly."""
    v22 = GmshDriver().read(_V22)
    v41 = GmshDriver().read(_V41)

    np.testing.assert_allclose(v22.mesh.node_x, v41.mesh.node_x)
    np.testing.assert_allclose(v22.mesh.node_y, v41.mesh.node_y)
    np.testing.assert_array_equal(
        v22.mesh.face_node_connectivity.data, v41.mesh.face_node_connectivity.data
    )


@pytest.mark.parametrize("path", [_V22, _V41])
def test_gmsh_faces_are_counter_clockwise(path):
    """Every face comes back wound counter-clockwise, whatever the file's order."""
    dataset = GmshDriver().read(path)
    connectivity = dataset.mesh.face_node_connectivity.data

    areas = [_signed_area2(dataset.mesh, row) for row in connectivity]
    assert all(area > 0 for area in areas)


@pytest.mark.parametrize("path", [_V22, _V41])
def test_gmsh_boundary_lines_are_skipped(path):
    """The 21 boundary line elements are dropped; only the 111 triangles become faces.

    Both fixtures carry 132 elements -- 21 type-1 lines and 111 type-2 triangles. A
    reader folding the lines into the face table would report 132 faces of mixed width.
    """
    dataset = GmshDriver().read(path)

    assert dataset.mesh.n_face == _EXPECTED_FACES
    # All faces are triangles, so the table is exactly three columns wide with no pad.
    assert dataset.mesh.face_node_connectivity.data.shape == (_EXPECTED_FACES, 3)


@pytest.mark.parametrize("path", [_V22, _V41])
def test_gmsh_has_no_crs(path):
    """Gmsh carries engineering coordinates, so no CRS is produced or guessed."""
    dataset = GmshDriver().read(path)

    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


def test_gmsh_can_open_accepts_both_layouts():
    """``can_open`` recognises both supported headers."""
    driver = GmshDriver()

    assert driver.can_open(_V22) is True
    assert driver.can_open(_V41) is True


def test_gmsh_can_open_declines_wrong_suffix(tmp_path):
    """Right content under a suffix this driver does not claim is still declined."""
    foreign = tmp_path / "mesh.txt"
    foreign.write_text(Path(_V22).read_text(encoding="utf-8"), encoding="utf-8")

    assert GmshDriver().can_open(str(foreign)) is False


def test_gmsh_can_open_declines_junk(tmp_path):
    """A ``.msh`` file that does not open with ``$MeshFormat`` is declined, not guessed."""
    junk = _write(tmp_path, "just some text\nwith no gmsh header\n", name="junk.msh")

    assert GmshDriver().can_open(junk) is False


def test_gmsh_can_open_declines_empty_and_directory(tmp_path):
    """``can_open`` never raises: an empty file, a directory and a missing path decline."""
    driver = GmshDriver()
    empty = tmp_path / "empty.msh"
    empty.write_bytes(b"")
    directory = tmp_path / "dir.msh"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "missing.msh")) is False


def test_gmsh_binary_file_is_refused(tmp_path):
    """A binary Gmsh file (file-type 1) is refused rather than mis-read."""
    text = "$MeshFormat\n2.2 1 8\n$EndMeshFormat\n"
    with pytest.raises(FileFormatNotSupportedError, match="binary"):
        GmshDriver().read(_write(tmp_path, text))


def test_gmsh_unsupported_version_is_refused(tmp_path):
    """A version this reader does not understand is refused by name."""
    text = "$MeshFormat\n3.0 0 8\n$EndMeshFormat\n"
    with pytest.raises(FileFormatNotSupportedError, match="version"):
        GmshDriver().read(_write(tmp_path, text))


def test_gmsh_file_without_faces_is_refused(tmp_path):
    """A file of only boundary lines has no surface, so it is refused."""
    text = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0.0 0.0 0.0
2 1.0 0.0 0.0
$EndNodes
$Elements
1
1 1 2 0 1 1 2
$EndElements
"""
    with pytest.raises(FileFormatNotSupportedError, match="no triangular"):
        GmshDriver().read(_write(tmp_path, text))


@pytest.mark.parametrize("path", [_V22, _V41])
def test_gmsh_is_reachable_through_read_file(path, gmsh_dispatch):
    """The suffix dispatch routes a ``.msh`` path to this driver via ``read_file``."""
    dataset = UgridDataset.read_file(path)

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_node == _EXPECTED_NODES
    assert dataset.mesh.n_face == _EXPECTED_FACES


# -- field-conversion helpers --------------------------------------------------------


def test_as_int_refuses_a_non_integer():
    """``_as_int`` names the file and the field when a token is not an integer."""
    with pytest.raises(FileFormatNotSupportedError, match="malformed node count"):
        _as_int("x", "node count", "p.msh")


def test_as_float_refuses_a_non_number():
    """``_as_float`` names the file and the field when a token is not a number."""
    with pytest.raises(FileFormatNotSupportedError, match="malformed node x"):
        _as_float("x", "node x", "p.msh")


# -- sniff edge cases that must answer False, never raise ----------------------------


def test_sniff_declines_a_header_with_no_version_line(tmp_path):
    """A file that is just ``$MeshFormat`` with no version line declines, not raises."""
    path = _write(tmp_path, "$MeshFormat\n", name="only_header.msh")

    assert GmshDriver().can_open(path) is False


def test_sniff_declines_a_non_numeric_version(tmp_path):
    """A ``$MeshFormat`` whose version is not a number declines rather than raising."""
    path = _write(tmp_path, "$MeshFormat\nabc 0 8\n$EndMeshFormat\n", name="badver.msh")

    assert GmshDriver().can_open(path) is False


# -- malformed-record refusals the committed samples do not carry --------------------
#
# The two insulated-*.msh fixtures are clean, so the truncation and short-record
# branches below need tiny inline Gmsh text. The format is plain ASCII parsed by our own
# record walk, so authoring these snippets is honest white-box coverage, not a fabricated
# binary.


def test_truncated_file_is_refused(tmp_path):
    """A file ending where a section was expected is refused, naming what was read."""
    path = _write(tmp_path, "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")

    with pytest.raises(FileFormatNotSupportedError, match="ends while reading"):
        GmshDriver().read(path)


def test_first_record_not_header_is_refused(tmp_path):
    """A file whose first record is not ``$MeshFormat`` is refused as not a Gmsh file."""
    path = _write(tmp_path, "$Nodes\n1\n1 0 0 0\n$EndNodes\n")

    with pytest.raises(FileFormatNotSupportedError, match="expected a .MeshFormat"):
        GmshDriver().read(path)


def test_v22_short_node_record_is_refused(tmp_path):
    """A v2.2 node record with fewer than four fields is refused."""
    text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n" "$Nodes\n1\n1 0.0 0.0\n$EndNodes\n"

    with pytest.raises(FileFormatNotSupportedError, match="short node record"):
        GmshDriver().read(_write(tmp_path, text))


def test_v41_short_coordinate_line_is_refused(tmp_path):
    """A v4.1 node coordinate line with fewer than three values is refused."""
    text = (
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$Nodes\n1 1 1 1\n0 1 0 1\n1\n0.0 0.0\n$EndNodes\n"
    )

    with pytest.raises(FileFormatNotSupportedError, match="short node coordinate"):
        GmshDriver().read(_write(tmp_path, text))


def test_v22_short_element_record_is_refused(tmp_path):
    """A v2.2 triangle record carrying too few node tags is refused."""
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 0 1 1 2\n$EndElements\n"
    )

    with pytest.raises(FileFormatNotSupportedError, match="short element record"):
        GmshDriver().read(_write(tmp_path, text))


def test_v41_short_element_record_is_refused(tmp_path):
    """A v4.1 triangle record carrying too few node tags is refused."""
    text = (
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$Nodes\n1 3 1 3\n0 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n"
        "$Elements\n1 1 1\n2 1 2 1\n1 1 2\n$EndElements\n"
    )

    with pytest.raises(FileFormatNotSupportedError, match="short element record"):
        GmshDriver().read(_write(tmp_path, text))


def test_unaccounted_element_type_is_refused(tmp_path):
    """An element of a type with no known node count refuses the file, not guesses."""
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 99 2 0 1 1 2 3\n$EndElements\n"
    )

    with pytest.raises(FileFormatNotSupportedError, match="element type 99"):
        GmshDriver().read(_write(tmp_path, text))


def test_element_referencing_an_undefined_node_tag_is_refused(tmp_path):
    """A face element referencing a node tag no node defines raises a ValueError."""
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 0 1 1 2 9\n$EndElements\n"
    )

    with pytest.raises(ValueError, match="node tag 9"):
        GmshDriver().read(_write(tmp_path, text))
