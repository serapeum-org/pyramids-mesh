"""Tests for the GiD / CIMNE ASCII mesh reader (issue #68).

The committed fixture ``tube_2D_spline.GiD.msh`` (GetFEM, LGPL-3.0-or-later) is a real
GiD export of quadratic triangles -- ``ElemType Triangle Nnode 6`` -- so the central
assertions are that it reads to its 943 nodes and 414 faces and that each six-node
element is reduced to its three corner nodes. Alongside that: winding is normalised,
``crs`` is ``None``, and -- because ``.msh`` is shared with the Gmsh reader -- the two
drivers never both accept either sample.

The clean fixture carries no comments, gapped ids, quadrilaterals or malformed records,
so the defensive branches below use tiny inline GiD snippets in ``tmp_path``. The format
is plain ASCII parsed by our own record walk, so authoring these is honest white-box
coverage, not a fabricated binary.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset, _drivers

from pyramids_mesh.drivers.gid import GidDriver, _as_float, _as_int
from pyramids_mesh.drivers.gmsh import GmshDriver

_DATA = Path(__file__).parent.parent / "data"
_GID = str(_DATA / "gid" / "tube_2D_spline.GiD.msh")
_GMSH = str(_DATA / "gmsh" / "insulated-2.2.msh")

_EXPECTED_NODES = 943
_EXPECTED_FACES = 414


def _write(tmp_path, text, name="mesh.msh"):
    """Write a GiD fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _signed_area2(mesh, row):
    """Twice the signed area of one face of a built mesh (CCW is positive)."""
    real = row[row != -1]
    xs, ys = mesh.node_x[real], mesh.node_y[real]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


@pytest.fixture
def gid_dispatch():
    """Register the GiD driver into the discovery cache for the duration of a test.

    The driver is not in the installed entry points, so ``read_file`` would not find it
    on its own; injecting it alongside the registered drivers lets a dispatch test
    exercise the real suffix/``can_open`` routing -- including the co-existence with the
    registered Gmsh driver on the shared ``.msh`` suffix -- without a reinstall. The
    cache is cleared afterwards so no other test inherits the injected driver.
    """
    _drivers.clear_driver_cache()
    registry = dict(_drivers.mesh_drivers())
    registry.setdefault("gid", GidDriver())
    _drivers._CACHE = registry
    try:
        yield
    finally:
        _drivers.clear_driver_cache()


# -- the real GetFEM sample ----------------------------------------------------------


def test_gid_reads_expected_counts():
    """The quadratic-triangle sample reads to its 943 nodes and 414 faces."""
    dataset = GidDriver().read(_GID)

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_node == _EXPECTED_NODES
    assert dataset.mesh.n_face == _EXPECTED_FACES


def test_gid_quadratic_triangles_reduce_to_three_corners():
    """``Nnode 6`` elements keep only their three corner nodes, so the table is 3 wide."""
    dataset = GidDriver().read(_GID)

    assert dataset.mesh.face_node_connectivity.data.shape == (_EXPECTED_FACES, 3)


def test_gid_connectivity_is_zero_based_with_recorded_base():
    """Node ids arrive 0-based, within range, with the GiD 1-based base recorded."""
    dataset = GidDriver().read(_GID)
    connectivity = dataset.mesh.face_node_connectivity

    data = connectivity.data
    assert data.min() == 0
    assert data.max() == _EXPECTED_NODES - 1
    assert connectivity.original_start_index == 1


def test_gid_faces_are_counter_clockwise():
    """Every face comes back wound counter-clockwise, whatever the file's order."""
    dataset = GidDriver().read(_GID)
    connectivity = dataset.mesh.face_node_connectivity.data

    areas = [_signed_area2(dataset.mesh, row) for row in connectivity]
    assert all(area > 0 for area in areas)


def test_gid_has_no_crs():
    """GiD carries engineering coordinates, so no CRS is produced or guessed."""
    dataset = GidDriver().read(_GID)

    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


def test_gid_is_reachable_through_read_file(gid_dispatch):
    """The suffix dispatch routes the ``.msh`` sample to this driver via ``read_file``."""
    dataset = UgridDataset.read_file(_GID)

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_node == _EXPECTED_NODES
    assert dataset.mesh.n_face == _EXPECTED_FACES


# -- co-existence with the Gmsh reader on the shared .msh suffix ---------------------


def test_gid_can_open_accepts_the_sample():
    """``can_open`` recognises the GiD ``MESH`` signature."""
    assert GidDriver().can_open(_GID) is True


def test_gid_can_open_declines_a_gmsh_file():
    """A Gmsh ``$MeshFormat`` file under the shared suffix is declined, not guessed."""
    assert GidDriver().can_open(_GMSH) is False


def test_gmsh_declines_the_gid_sample():
    """The shipped Gmsh reader declines the GiD sample: the two never both accept it."""
    assert GmshDriver().can_open(_GID) is False


def test_gid_and_gmsh_do_not_both_accept_either_sample():
    """Neither sample is accepted by both drivers -- the signatures are exclusive."""
    gid, gmsh = GidDriver(), GmshDriver()

    assert [gid.can_open(_GID), gmsh.can_open(_GID)] == [True, False]
    assert [gid.can_open(_GMSH), gmsh.can_open(_GMSH)] == [False, True]


def test_gid_can_open_declines_empty_and_directory(tmp_path):
    """``can_open`` never raises: an empty file, a directory and a missing path decline."""
    driver = GidDriver()
    empty = tmp_path / "empty.msh"
    empty.write_bytes(b"")
    directory = tmp_path / "dir.msh"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "missing.msh")) is False


def test_gid_can_open_declines_wrong_suffix(tmp_path):
    """Right content under a suffix this driver does not claim is still declined."""
    foreign = tmp_path / "mesh.txt"
    foreign.write_text(Path(_GID).read_text(encoding="utf-8"), encoding="utf-8")

    assert GidDriver().can_open(str(foreign)) is False


# -- defensive branches, with tiny inline GiD snippets -------------------------------


def test_gid_reads_a_linear_triangle_mesh(tmp_path):
    """A plain ``Nnode 3`` triangle mesh reads with no corner reduction needed."""
    text = (
        "# a linear triangle mesh\n"
        "MESH dimension 2 ElemType Triangle Nnode 3\n"
        "Coordinates\n"
        "1 0.0 0.0\n2 1.0 0.0\n3 1.0 1.0\n4 0.0 1.0\n"
        "end coordinates\n"
        "Elements\n1 1 2 3\n2 1 3 4\n"
        "end elements\n"
    )
    dataset = GidDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    assert dataset.mesh.face_node_connectivity.data.shape == (2, 3)


def test_gid_reads_a_quadrilateral_mesh(tmp_path):
    """A ``Quadrilateral`` block keeps four corner nodes per face."""
    text = (
        "MESH dimension 2 ElemType Quadrilateral Nnode 4\n"
        "Coordinates\n"
        "1 0.0 0.0\n2 1.0 0.0\n3 1.0 1.0\n4 0.0 1.0\n"
        "end coordinates\n"
        "Elements\n1 1 2 3 4\n"
        "end elements\n"
    )
    dataset = GidDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_face == 1
    assert dataset.mesh.face_node_connectivity.data.shape == (1, 4)
    assert sorted(dataset.mesh.face_node_connectivity.data[0].tolist()) == [0, 1, 2, 3]


def test_gid_resolves_gapped_node_ids(tmp_path):
    """Gapped, out-of-order node ids map by position, not value, so faces stay wired."""
    text = (
        "MESH dimension 2 ElemType Triangle Nnode 3\n"
        "Coordinates\n"
        "10 0.0 0.0\n5 1.0 0.0\n42 0.0 1.0\n"
        "end coordinates\n"
        "Elements\n1 10 5 42\n"
        "end elements\n"
    )
    dataset = GidDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_node == 3
    # 10->0, 5->1, 42->2 by order of first appearance; winding may reverse the triangle.
    assert sorted(dataset.mesh.face_node_connectivity.data[0].tolist()) == [0, 1, 2]


def test_gid_keeps_three_corners_of_a_quadratic_triangle(tmp_path):
    """A single ``Nnode 6`` element drops its three mid-edge nodes, keeping the corners.

    The corners are listed first (ids 1, 2, 3) and the mid-edge nodes after (4, 5, 6),
    so the built face references only the first three nodes -- indices 0, 1, 2.
    """
    text = (
        "MESH dimension 2 ElemType Triangle Nnode 6\n"
        "Coordinates\n"
        "1 0.0 0.0\n2 2.0 0.0\n3 0.0 2.0\n"
        "4 1.0 0.0\n5 1.0 1.0\n6 0.0 1.0\n"
        "end coordinates\n"
        "Elements\n1 1 2 3 4 5 6\n"
        "end elements\n"
    )
    dataset = GidDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_node == 6
    assert dataset.mesh.face_node_connectivity.data.shape == (1, 3)
    assert sorted(dataset.mesh.face_node_connectivity.data[0].tolist()) == [0, 1, 2]


def test_gid_unclosed_coordinates_block_is_refused(tmp_path):
    """A ``Coordinates`` block that runs to end of file is refused, not read truncated."""
    text = (
        "MESH dimension 2 ElemType Triangle Nnode 3\n"
        "Coordinates\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="Coordinates block"):
        GidDriver().read(_write(tmp_path, text))


def test_gid_unclosed_elements_block_is_refused(tmp_path):
    """An ``Elements`` block that runs to end of file is refused."""
    text = (
        "MESH dimension 2 ElemType Triangle Nnode 3\n"
        "Coordinates\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\nend coordinates\n"
        "Elements\n1 1 2 3\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="Elements block"):
        GidDriver().read(_write(tmp_path, text))


def test_gid_header_without_elemtype_is_refused(tmp_path):
    """A ``MESH`` header missing ``ElemType`` is refused, naming the missing keyword."""
    text = "MESH dimension 2 Nnode 3\n" "Coordinates\n1 0.0 0.0\nend coordinates\n"
    with pytest.raises(FileFormatNotSupportedError, match="elemtype"):
        GidDriver().read(_write(tmp_path, text))


def test_gid_file_without_faces_is_refused(tmp_path):
    """A file of only ``Line`` elements has no surface, so it is refused."""
    text = (
        "MESH dimension 2 ElemType Line Nnode 2\n"
        "Coordinates\n1 0.0 0.0\n2 1.0 0.0\nend coordinates\n"
        "Elements\n1 1 2\nend elements\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="no surface or volumetric"):
        GidDriver().read(_write(tmp_path, text))


def test_gid_element_referencing_an_undefined_node_is_refused(tmp_path):
    """A face referencing a node id no coordinate record defines raises a ValueError."""
    text = (
        "MESH dimension 2 ElemType Triangle Nnode 3\n"
        "Coordinates\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\nend coordinates\n"
        "Elements\n1 1 2 9\nend elements\n"
    )
    with pytest.raises(ValueError, match="node id 9"):
        GidDriver().read(_write(tmp_path, text))


def test_gid_short_element_record_is_refused(tmp_path):
    """A triangle record carrying too few node ids is refused."""
    text = (
        "MESH dimension 2 ElemType Triangle Nnode 3\n"
        "Coordinates\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\nend coordinates\n"
        "Elements\n1 1 2\nend elements\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="short element record"):
        GidDriver().read(_write(tmp_path, text))


# -- field-conversion helpers --------------------------------------------------------


def test_as_int_refuses_a_non_integer():
    """``_as_int`` names the file and the field when a token is not an integer."""
    with pytest.raises(FileFormatNotSupportedError, match="malformed node id"):
        _as_int("x", "node id", "p.msh")


def test_as_float_refuses_a_non_number():
    """``_as_float`` names the file and the field when a token is not a number."""
    with pytest.raises(FileFormatNotSupportedError, match="malformed node x"):
        _as_float("x", "node x", "p.msh")
