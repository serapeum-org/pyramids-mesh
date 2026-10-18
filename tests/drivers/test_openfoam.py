"""Tests for the OpenFOAM polyMesh reader, against a real committed sample directory.

An OpenFOAM mesh is a ``polyMesh`` directory of plain-text component files. A genuine
sample is committed under ``tests/data/openfoam/polyMesh/``: 260 points and 321 faces,
of which 258 are boundary faces grouped into five patches (the remaining 63 are the
interior faces this reader drops). The tests read the committed bytes.

The driver is directory-based: its component files are extensionless, so the recognised
file is the ParaView ``<case>.foam`` marker (its declared suffix), while a ``polyMesh``
directory or its ``points`` component is accepted directly by ``can_open``. These tests
drive ``OpenFoamDriver`` on the directory, the way ``read_file(path, driver="openfoam")``
would, and also cover the ``.foam`` marker resolution.
"""

import pyramids  # noqa: F401  # isort:skip

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.openfoam import (
    OpenFoamDriver,
    _foam_list,
    _OpenFoamReader,
    _parse_boundary,
    _parse_faces,
    _parse_points,
    _require_ascii,
)

_DATA = Path(__file__).parent.parent / "data" / "openfoam"
_POLYMESH = _DATA / "polyMesh"


def _read(directory):
    """Read a committed polyMesh directory, ignoring the expected winding warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OpenFoamDriver().read(str(directory))


def test_foam_marker_resolves_to_the_case_polymesh(tmp_path):
    """A ``<case>.foam`` marker resolves to the case's ``constant/polyMesh`` and reads."""
    import shutil

    poly = tmp_path / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    for component in ("points", "faces", "owner", "neighbour", "boundary"):
        shutil.copy2(_POLYMESH / component, poly / component)
    marker = tmp_path / "case.foam"
    marker.write_text("", encoding="utf-8")
    driver = OpenFoamDriver()
    assert driver.can_open(str(marker)) is True
    assert driver._poly_dir(str(marker)) == str(poly)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert driver.read(str(marker)).mesh.n_node == 260


def test_reads_nodes_and_boundary_faces():
    """The polyMesh reads 260 nodes and the 258 boundary faces, not all 321."""
    mesh = _read(_POLYMESH).mesh
    assert mesh.n_node == 260
    assert mesh.n_face == 258


def test_faces_are_quads_fill_padded_to_the_widest():
    """Every boundary face of the sample is a quad, so the connectivity is (258, 4)."""
    mesh = _read(_POLYMESH).mesh
    assert mesh.face_node_connectivity.data.shape == (258, 4)


def test_connectivity_is_pinned_not_just_counted():
    """The exact boundary selection and compaction are locked against a golden hash.

    Gathering the wrong face ranges, or remapping the compacted points differently,
    could keep the same face *count* while wiring different nodes, so this pins the
    final connectivity (patch gathering, point compaction and the CCW normalisation
    applied downstream) against a hash derived from the committed sample.
    """
    faces = _read(_POLYMESH).mesh.face_node_connectivity.data
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    digest = hashlib.sha256(faces.tobytes()).hexdigest()
    assert digest == (
        "641d8bd4f5d050867a8b6436618170c8c41836cb3d2548f8eec82299d741a170"
    ), "OpenFOAM boundary selection/compaction changed; regenerate the hash if intended."


def test_point_indices_are_zero_based():
    """OpenFOAM indices are 0-based, so nothing is re-based and the base is recorded 0."""
    connectivity = _read(_POLYMESH).mesh.face_node_connectivity
    assert connectivity.original_start_index == 0
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 259


def test_third_coordinate_becomes_node_z():
    """The z coordinate becomes a node-located node_z variable."""
    node_z = _read(_POLYMESH).get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (260,)


def test_crs_is_none_for_cfd_local_coordinates():
    """CFD local coordinates carry no georeference, so crs is None."""
    assert _read(_POLYMESH).mesh.crs is None


def test_can_open_accepts_directory_and_points_component():
    """can_open accepts the polyMesh directory and its points file, not other files."""
    driver = OpenFoamDriver()
    assert driver.can_open(str(_POLYMESH)) is True
    assert driver.can_open(str(_POLYMESH / "points")) is True
    # a non-points component inside a polyMesh is not a recognised entry point
    assert driver.can_open(str(_POLYMESH / "faces")) is False


def test_can_open_declines_non_polymesh(tmp_path):
    """can_open declines a directory lacking the components, and a plain file."""
    driver = OpenFoamDriver()
    assert driver.can_open(str(tmp_path)) is False
    stray = tmp_path / "points"
    stray.write_text("not a polymesh")
    # a lone 'points' file whose directory has no faces/boundary is not a polyMesh
    assert driver.can_open(str(stray)) is False
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False


def test_missing_component_is_a_format_error(tmp_path):
    """A directory missing a required component is a format error."""
    partial = tmp_path / "polyMesh"
    partial.mkdir()
    (partial / "points").write_text("0\n(\n)\n")
    # no faces / boundary
    with pytest.raises(FileFormatNotSupportedError, match="missing"):
        OpenFoamDriver().read(str(partial))


def test_read_of_a_non_polymesh_path_is_a_format_error(tmp_path):
    """read() on a path that is neither a polyMesh dir nor a points file is refused."""
    other = tmp_path / "notes.txt"
    other.write_text("not a polymesh")
    with pytest.raises(FileFormatNotSupportedError, match="not an OpenFOAM polyMesh"):
        OpenFoamDriver().read(str(other))


def _write_polymesh(directory, *, points, faces, boundary):
    """Write a minimal ASCII polyMesh from raw body strings."""
    directory.mkdir()

    def _file(obj, cls, body):
        return (
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n    object {obj};\n}}\n{body}\n"
        )

    (directory / "points").write_text(_file("points", "vectorField", points))
    (directory / "faces").write_text(_file("faces", "faceList", faces))
    (directory / "boundary").write_text(_file("boundary", "polyBoundaryMesh", boundary))
    return directory


def test_mixed_triangle_and_quad_is_fill_padded(tmp_path):
    """A boundary of one triangle and one quad pads the triangle row with -1."""
    points = "5\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n(2 0 0)\n)\n"
    faces = "2\n(\n3(0 1 2)\n4(0 1 4 3)\n)\n"
    boundary = (
        "1\n(\n    walls\n    {\n        type wall;\n"
        "        nFaces 2;\n        startFace 0;\n    }\n)\n"
    )
    built = _write_polymesh(
        tmp_path / "polyMesh", points=points, faces=faces, boundary=boundary
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = OpenFoamDriver().read(str(built)).mesh
    conn = mesh.face_node_connectivity.data
    assert conn.shape == (2, 4)
    # exactly one row carries a single -1 pad (the triangle), the quad carries none
    pad_counts = sorted(int(np.sum(row == -1)) for row in conn)
    assert pad_counts == [0, 1]


def test_binary_format_is_refused(tmp_path):
    """A component whose FoamFile header declares format binary is refused."""
    built = tmp_path / "polyMesh"
    built.mkdir()
    header = (
        "FoamFile\n{{\n    version 2.0;\n    format binary;\n"
        "    class {cls};\n    object {obj};\n}}\n"
    )
    (built / "points").write_text(header.format(cls="vectorField", obj="points"))
    (built / "faces").write_text(header.format(cls="faceList", obj="faces"))
    (built / "boundary").write_text(
        header.format(cls="polyBoundaryMesh", obj="boundary")
    )
    with pytest.raises(FileFormatNotSupportedError, match="ascii"):
        OpenFoamDriver().read(str(built))


def test_boundary_range_past_the_face_list_is_refused(tmp_path):
    """A patch naming faces beyond the face list is a clean format error."""
    points = "3\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n)\n"
    faces = "1\n(\n3(0 1 2)\n)\n"
    boundary = (
        "1\n(\n    walls\n    {\n        type wall;\n"
        "        nFaces 5;\n        startFace 0;\n    }\n)\n"
    )
    built = _write_polymesh(
        tmp_path / "polyMesh", points=points, faces=faces, boundary=boundary
    )
    with pytest.raises(FileFormatNotSupportedError, match="spans faces"):
        OpenFoamDriver().read(str(built))


def test_boundary_naming_no_faces_is_refused(tmp_path):
    """A boundary whose only patch names zero faces leaves no surface and is refused."""
    points = "3\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n)\n"
    faces = "1\n(\n3(0 1 2)\n)\n"
    boundary = (
        "1\n(\n    empty\n    {\n        type empty;\n"
        "        nFaces 0;\n        startFace 0;\n    }\n)\n"
    )
    built = _write_polymesh(
        tmp_path / "polyMesh", points=points, faces=faces, boundary=boundary
    )
    with pytest.raises(FileFormatNotSupportedError, match="names no faces"):
        OpenFoamDriver().read(str(built))


def test_compact_refuses_an_empty_boundary():
    """_compact refuses directly when the boundary names no faces at all."""
    with pytest.raises(FileFormatNotSupportedError, match="names no faces"):
        _OpenFoamReader._compact(np.zeros((1, 3), dtype=np.float64), [])


def test_poly_dir_of_a_nonexistent_path_is_none():
    """A path that is neither a directory nor a file resolves to no polyMesh."""
    assert OpenFoamDriver._poly_dir("no_such_openfoam_path_xyz") is None


def test_poly_dir_foam_marker_without_a_case_polymesh_is_none(tmp_path, monkeypatch):
    """A '.foam' marker whose case has no polyMesh directory resolves to None."""
    monkeypatch.chdir(tmp_path)
    # A bare filename: its dirname is empty, so none of the candidate directories
    # (constant/polyMesh, polyMesh, the empty case) exists -- the loop finds nothing.
    (tmp_path / "bare.foam").write_text("", encoding="utf-8")
    assert OpenFoamDriver._poly_dir("bare.foam") is None


class TestFoamParsers:
    """The FoamFile parsing helpers read counts, faces and patches correctly."""

    def test_foam_list_keeps_nested_parentheses_inside_the_body(self):
        """The body is matched to the final paren, so a nested list is not cut short."""
        count, body = _foam_list("2\n(\n    a { inGroups 1(x); }\n    b\n)\n")
        assert count == 2
        assert "1(x)" in body

    def test_foam_list_without_a_list_is_a_format_error(self):
        """Text with no '<count> ( ... )' list is a clean format error."""
        with pytest.raises(FileFormatNotSupportedError):
            _foam_list("there is no list here")

    def test_parse_points_checks_the_declared_count(self):
        """A points header whose count disagrees with the triples found is refused."""
        text = "FoamFile{format ascii;}\n3\n(\n(0 0 0)\n(1 0 0)\n)\n"
        with pytest.raises(FileFormatNotSupportedError, match="declares 3 points"):
            _parse_points(text)

    def test_parse_faces_checks_each_entry_size(self):
        """A face whose declared node count disagrees with its indices is refused."""
        text = "FoamFile{format ascii;}\n1\n(\n4(0 1 2)\n)\n"
        with pytest.raises(FileFormatNotSupportedError, match="declares 4 nodes"):
            _parse_faces(text)

    def test_parse_boundary_reads_patch_ranges(self):
        """Each patch's name, startFace and nFaces are read, in file order."""
        text = (
            "FoamFile{format ascii;}\n2\n(\n"
            "    inlet\n    {\n        type patch;\n"
            "        nFaces 4;\n        startFace 10;\n    }\n"
            "    wall\n    {\n        type wall;\n"
            "        nFaces 2;\n        startFace 14;\n    }\n)\n"
        )
        patches = _parse_boundary(text)
        assert patches == [("inlet", 10, 4), ("wall", 14, 2)]

    def test_require_ascii_passes_text_with_no_foamfile_header(self):
        """Text carrying no FoamFile header is accepted (format cannot be checked)."""
        # no FoamFile{...} block -- the format guard returns without raising
        _require_ascii("3\n(\n(0 0 0)\n)\n", "points")

    def test_parse_faces_checks_the_total_face_count(self):
        """A faces header whose count disagrees with the entries found is refused."""
        # one valid entry, but the header declares three faces
        text = "FoamFile{format ascii;}\n3\n(\n3(0 1 2)\n)\n"
        with pytest.raises(FileFormatNotSupportedError, match="declares 3 faces"):
            _parse_faces(text)

    def test_parse_boundary_missing_startface_is_refused(self):
        """A patch missing its startFace entry is refused."""
        text = (
            "FoamFile{format ascii;}\n1\n(\n    walls\n    {\n"
            "        type wall;\n        nFaces 2;\n    }\n)\n"
        )
        with pytest.raises(
            FileFormatNotSupportedError, match="missing startFace or nFaces"
        ):
            _parse_boundary(text)
