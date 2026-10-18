"""Tests for the Medit (INRIA) ``.mesh`` reader.

Medit is a keyword-block text format: ``MeshVersionFormatted``, a ``Dimension``, a
``Vertices`` block and the element sections. The committed sample
``tests/data/medit/cube86.mesh`` is a 3D cube surface -- 39 vertices, 72 triangles and 86
tetrahedra -- so the tests pin that the surface triangles become faces, the tetrahedra
are dropped, and the vertex z column becomes ``node_z``.

The ``.mesh`` suffix is shared with the DHI Mike21 frame format, so the sniff is what
keeps the two drivers from both claiming the same file: Medit requires its
``MeshVersionFormatted`` signature, and Mike21 requires its numeric EUM header. A
dedicated test confirms neither accepts the other's committed sample.

Defensive branches the committed 3D sample has no shape for -- a 2D file, a mixed
triangle+quad surface, the volume-only refusal, an unknown section, the truncation and
non-numeric errors -- use tiny inline ``.mesh`` snippets in ``tmp_path``.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.medit import MeditDriver
from pyramids_mesh.drivers.misc import Mike21Driver

_DATA = Path(__file__).parent.parent / "data"
_CUBE = _DATA / "medit" / "cube86.mesh"
_MIKE21_SAMPLE = _DATA / "minimal.mesh"


def _read(path):
    """Read a sample, ignoring the expected winding/degeneracy warnings.

    The committed cube surface is three-dimensional, so many triangles project to a line
    in the x-y plane and the base validator reports them as degenerate; those warnings
    are not what these tests are about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MeditDriver().read(str(path))


def _write(tmp_path, text, name="mesh.mesh"):
    """Write an inline Medit fixture and return its path as a string."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# -- the committed 3D cube sample ----------------------------------------------------


def test_medit_reads_committed_cube():
    """The cube sample reads 39 vertices and 72 triangular surface faces."""
    dataset = _read(_CUBE)

    assert dataset.mesh.n_node == 39
    assert dataset.mesh.n_face == 72
    assert dataset.mesh.face_node_connectivity.data.shape == (72, 3)


def test_medit_drops_volume_cells():
    """The 86 tetrahedra are dropped -- only the surface triangles become faces."""
    dataset = _read(_CUBE)

    # 72 triangles only; the tetrahedra would have made this far larger if kept.
    assert dataset.mesh.n_face == 72


def test_medit_index_base():
    """Connectivity arrives 0-based, with the file's 1-based numbering recorded."""
    connectivity = _read(_CUBE).mesh.face_node_connectivity

    assert int(connectivity.data.min()) == 0
    # Vertex 39 (index 38) is the interior node used only by the dropped tetrahedra, so
    # the surface triangles reach only index 37 -- 0-based either way.
    assert int(connectivity.data.max()) == 37
    assert connectivity.original_start_index == 1


def test_medit_elevation_becomes_node_z_for_3d():
    """A 3D file's vertex z column becomes a node-located ``node_z`` variable."""
    dataset = _read(_CUBE)

    assert list(dataset.data_variable_names) == ["node_z"]
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (39,)


def test_medit_crs_is_none():
    """Medit carries no CRS, so it is left unset rather than guessed."""
    dataset = _read(_CUBE)

    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


def test_medit_faces_are_counter_clockwise():
    """Every built face is counter-clockwise (or degenerate) after validation."""
    mesh = _read(_CUBE).mesh

    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 >= 0


# -- the shared-suffix contract with Mike21 ------------------------------------------


def test_medit_and_mike21_do_not_both_accept_either_sample():
    """The two ``.mesh`` drivers never collide: each accepts only its own sample."""
    medit = MeditDriver()
    mike21 = Mike21Driver()

    # Medit accepts the Medit sample and declines the Mike21 one.
    assert medit.can_open(str(_CUBE)) is True
    assert medit.can_open(str(_MIKE21_SAMPLE)) is False

    # Mike21 accepts the Mike21 sample and declines the Medit one.
    assert mike21.can_open(str(_MIKE21_SAMPLE)) is True
    assert mike21.can_open(str(_CUBE)) is False


def test_medit_can_open_declines_junk(tmp_path):
    """``can_open`` declines empty, foreign and directory paths without raising."""
    driver = MeditDriver()

    empty = tmp_path / "empty.mesh"
    empty.write_bytes(b"")
    foreign = tmp_path / "other.mesh"
    foreign.write_text("not a medit header at all\n", encoding="utf-8")
    directory = tmp_path / "folder.mesh"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(foreign)) is False
    assert driver.can_open(str(directory)) is False
    assert driver.can_open(str(tmp_path / "absent.mesh")) is False


def test_medit_suffix_is_required(tmp_path):
    """Right content under a suffix the driver does not claim is still declined."""
    path = tmp_path / "cube.txt"
    path.write_text(_CUBE.read_text(encoding="utf-8"), encoding="utf-8")

    assert MeditDriver().can_open(str(path)) is False


# -- defensive branches the 3D sample has no shape for -------------------------------


def _TWO_D_SQUARE():
    """Return a minimal 2D Medit mesh of two triangles."""
    return (
        "MeshVersionFormatted 1\n"
        "Dimension 2\n"
        "Vertices\n"
        "4\n"
        "0 0 1\n"
        "1 0 1\n"
        "1 1 1\n"
        "0 1 1\n"
        "Triangles\n"
        "2\n"
        "1 2 3 0\n"
        "1 3 4 0\n"
        "End\n"
    )


def test_medit_2d_file_has_no_node_z(tmp_path):
    """A 2D file has no z column, so no ``node_z`` variable is produced."""
    dataset = MeditDriver().read(_write(tmp_path, _TWO_D_SQUARE()))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    assert list(dataset.data_variable_names) == []


def test_medit_dimension_value_on_its_own_line(tmp_path):
    """A ``Dimension`` whose value is on the next line reads, as the format allows."""
    text = _TWO_D_SQUARE().replace("Dimension 2\n", "Dimension\n2\n")
    dataset = MeditDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_node == 4


def test_medit_comment_lines_are_ignored(tmp_path):
    """A ``#`` comment runs to end of line and does not disturb the token stream."""
    text = _TWO_D_SQUARE().replace("Vertices\n", "# a GAMBIT-style comment\nVertices\n")
    dataset = MeditDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_face == 2


def test_medit_quadrilaterals_are_fill_padded_with_triangles(tmp_path):
    """A surface mixing triangles and quads pads the triangle row to the quad width."""
    text = (
        "MeshVersionFormatted 1\n"
        "Dimension 2\n"
        "Vertices\n"
        "5\n"
        "0 0 1\n"
        "1 0 1\n"
        "1 1 1\n"
        "0 1 1\n"
        "2 0 1\n"
        "Triangles\n"
        "1\n"
        "2 5 3 0\n"
        "Quadrilaterals\n"
        "1\n"
        "1 2 3 4 0\n"
        "End\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MeditDriver().read(_write(tmp_path, text))

    connectivity = dataset.mesh.face_node_connectivity
    assert connectivity.data.shape == (2, 4)
    rows = connectivity.data.tolist()
    # The triangle (first section) is padded to width four with -1; the quad is whole.
    assert rows[0] == [1, 4, 2, -1]
    assert rows[1] == [0, 1, 2, 3]


def test_medit_edges_section_is_skipped(tmp_path):
    """An ``Edges`` section is read past but not folded into the mesh."""
    text = (
        "MeshVersionFormatted 1\n"
        "Dimension 2\n"
        "Vertices\n"
        "4\n"
        "0 0 1\n"
        "1 0 1\n"
        "1 1 1\n"
        "0 1 1\n"
        "Edges\n"
        "1\n"
        "1 2 1\n"
        "Triangles\n"
        "2\n"
        "1 2 3 0\n"
        "1 3 4 0\n"
        "End\n"
    )
    dataset = MeditDriver().read(_write(tmp_path, text))

    assert dataset.mesh.n_face == 2
    assert dataset.mesh.edge_node_connectivity is None


def test_medit_volume_only_file_is_refused(tmp_path):
    """A file with only tetrahedra and no surface section is refused, not built empty."""
    text = (
        "MeshVersionFormatted 1\n"
        "Dimension 3\n"
        "Vertices\n"
        "4\n"
        "0 0 0 1\n"
        "1 0 0 1\n"
        "0 1 0 1\n"
        "0 0 1 1\n"
        "Tetrahedra\n"
        "1\n"
        "1 2 3 4 0\n"
        "End\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="no 2D surface section"):
        MeditDriver().read(_write(tmp_path, text))


def test_medit_unknown_section_is_refused(tmp_path):
    """A section keyword this reader does not know is refused rather than mis-parsed."""
    text = (
        "MeshVersionFormatted 1\n"
        "Dimension 2\n"
        "Vertices\n"
        "3\n"
        "0 0 1\n"
        "1 0 1\n"
        "0 1 1\n"
        "Corners\n"
        "1\n"
        "1\n"
        "Triangles\n"
        "1\n"
        "1 2 3 0\n"
        "End\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="does not read"):
        MeditDriver().read(_write(tmp_path, text))


def test_medit_unsupported_dimension_is_refused(tmp_path):
    """A dimension other than 2 or 3 is refused."""
    text = _TWO_D_SQUARE().replace("Dimension 2", "Dimension 1")
    with pytest.raises(FileFormatNotSupportedError, match="only 2 and 3 are read"):
        MeditDriver().read(_write(tmp_path, text))


def test_medit_missing_dimension_is_refused(tmp_path):
    """A file that never declares a dimension is refused."""
    text = "MeshVersionFormatted 1\n" "Triangles\n" "1\n" "1 2 3 0\n" "End\n"
    with pytest.raises(FileFormatNotSupportedError, match="declares no Dimension"):
        MeditDriver().read(_write(tmp_path, text))


def test_medit_truncated_count_is_refused(tmp_path):
    """A count that outruns the file's rows is refused, naming what is missing."""
    text = (
        "MeshVersionFormatted 1\n"
        "Dimension 2\n"
        "Vertices\n"
        "4\n"
        "0 0 1\n"
        "1 0 1\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="ends before"):
        MeditDriver().read(_write(tmp_path, text))


def test_medit_non_numeric_field_is_refused(tmp_path):
    """A non-numeric coordinate is refused with the field named."""
    text = _TWO_D_SQUARE().replace("0 0 1\n1 0 1", "east 0 1\n1 0 1")
    with pytest.raises(FileFormatNotSupportedError, match="expected a number"):
        MeditDriver().read(_write(tmp_path, text))


def test_medit_declares_read_mesh():
    """The driver advertises exactly the read_mesh capability."""
    assert "read_mesh" in MeditDriver.capabilities
    assert "needs_mesh" not in MeditDriver.capabilities
