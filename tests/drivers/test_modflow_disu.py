"""Tests for the MODFLOW 6 DISU reader, against a real committed sample grid.

A ``.disu`` file is the ASCII input of one MODFLOW 6 model. A genuine grid is committed
under ``tests/data/modflow6_disu/``: ``keating.disu`` is a quad DISU grid of 324 cells on
162 vertices (``MODFLOW-ORG/modflow6-testmodels``, ``test016_Keating_disu_250``, CC0 /
public domain). The tests read the committed bytes -- the mesh is the horizontal footprint
of 324 quadrilateral cells. A connectivity-only DISU (no vertices) is refused, since it
carries no geometry to build a mesh from.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset, _drivers
from pyramids_mesh.drivers.modflow import Modflow6DisuDriver, _Modflow6DisuReader

_DATA = Path(__file__).parent.parent / "data" / "modflow6_disu"
_DISU = _DATA / "keating.disu"


def _read(path):
    """Read a committed DISU file, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Modflow6DisuDriver().read(str(path))


def test_reads_the_horizontal_cell_footprint():
    """keating.disu reads 324 quad cells on 162 vertices -- one footprint."""
    mesh = _read(_DISU).mesh
    assert mesh.n_node == 162
    assert mesh.n_face == 324
    assert mesh.face_node_connectivity.data.shape == (324, 4)


def test_vertex_ids_are_one_based_resolved_to_zero_based():
    """The 1-based icvert ids are resolved to 0-based, and the base recorded."""
    connectivity = _read(_DISU).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 161  # NVERT 162 -> 0..161


def test_faces_are_counter_clockwise():
    """DISU lists cells clockwise; they come back counter-clockwise after validation."""
    mesh = _read(_DISU).mesh
    for row in mesh.face_node_connectivity.data:
        real = row[row != -1]
        xs = mesh.node_x[real]
        ys = mesh.node_y[real]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_cell_centres_become_face_centres():
    """The CELL2D xc/yc columns are carried as the mesh face centres."""
    mesh = _read(_DISU).mesh
    assert mesh.face_x is not None and mesh.face_y is not None
    assert mesh.face_x.shape == (324,)
    # The first cell's centre is 125 0.5 in the committed file.
    assert mesh.face_x[0] == pytest.approx(125.0)
    assert mesh.face_y[0] == pytest.approx(0.5)


def test_no_crs_or_elevation_in_a_disu_file():
    """A .disu file carries geometry only, so crs and node_z are absent."""
    dataset = _read(_DISU)
    assert dataset.mesh.crs is None
    assert dataset.get("node_z") is None


def test_can_open_accepts_the_disu_file():
    """can_open accepts the real DISU file on its NODES/NJA dimensions, not its suffix."""
    assert Modflow6DisuDriver().can_open(str(_DISU)) is True


def test_can_open_declines_junk_and_a_disv_grid(tmp_path):
    """can_open declines a .disu that is not DISU, a DISV deck, and a directory."""
    driver = Modflow6DisuDriver()
    # right suffix, wrong content -- no DISU dimensions
    junk = tmp_path / "fake.disu"
    junk.write_text("this is not a modflow grid\n")
    assert driver.can_open(str(junk)) is False
    # a DISV-shaped deck carries NCPL, never NJA, so it is not a DISU grid
    disv_like = tmp_path / "grid.disu"
    disv_like.write_text(
        "BEGIN DIMENSIONS\n  NLAY 1\n  NCPL 1\n  NVERT 3\nEND DIMENSIONS\n"
    )
    assert driver.can_open(str(disv_like)) is False
    # a non-.disu file is declined on its suffix
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False
    # a directory is declined
    assert driver.can_open(str(tmp_path)) is False


def test_dispatch_through_read_file_by_driver_name():
    """read_file routes to this driver when it is registered and named.

    The entry point is reported for a maintainer to add to pyproject.toml, so the driver
    is not discovered in the test environment. Injecting it into pyramids' driver cache
    exercises the real read_file resolution path (driver= by name) without a reinstall.
    """
    original = _drivers._CACHE
    try:
        registry = dict(original or {})
        registry["modflow6_disu"] = Modflow6DisuDriver()
        _drivers._CACHE = registry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = UgridDataset.read_file(str(_DISU), driver="modflow6_disu")
    finally:
        _drivers._CACHE = original
    assert dataset.mesh.n_face == 324
    assert dataset.mesh.n_node == 162


def test_read_of_a_non_disu_path_is_a_format_error(tmp_path):
    """read() on a path that is not a file is refused as a format error."""
    missing = tmp_path / "gone.disu"
    with pytest.raises(FileFormatNotSupportedError, match="not a MODFLOW 6 DISU file"):
        Modflow6DisuDriver().read(str(missing))


def _disu_text(
    dimensions="  NODES 1\n  NJA 1\n  NVERT 3\n",
    vertices="  1 0.0 0.0\n  2 1.0 0.0\n  3 0.0 1.0\n",
    cells="  1 0.33 0.33 3 1 2 3\n",
):
    """Assemble a tiny valid .disu body; override one block to malform it."""
    return (
        f"BEGIN DIMENSIONS\n{dimensions}END DIMENSIONS\n\n"
        f"BEGIN VERTICES\n{vertices}END VERTICES\n\n"
        f"BEGIN CELL2D\n{cells}END CELL2D\n"
    )


def test_a_connectivity_only_disu_is_refused(tmp_path):
    """A DISU with no NVERT/VERTICES/CELL2D carries no geometry and is refused."""
    disu = tmp_path / "graph.disu"
    disu.write_text(
        "BEGIN DIMENSIONS\n"
        "  NODES 3\n"
        "  NJA 9\n"
        "END DIMENSIONS\n\n"
        "BEGIN CONNECTIONDATA\n"
        "  IAC\n    INTERNAL\n      3 3 3\n"
        "END CONNECTIONDATA\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="connectivity-only"):
        Modflow6DisuDriver().read(str(disu))


def test_a_zero_nvert_disu_is_refused(tmp_path):
    """NVERT 0 is the same as no vertices: refused as connectivity-only."""
    disu = tmp_path / "zero.disu"
    disu.write_text("BEGIN DIMENSIONS\n  NODES 1\n  NJA 1\n  NVERT 0\nEND DIMENSIONS\n")
    with pytest.raises(FileFormatNotSupportedError, match="connectivity-only"):
        Modflow6DisuDriver().read(str(disu))


def test_a_cell_count_mismatch_is_a_format_error(tmp_path):
    """A NODES count that disagrees with the CELL2D rows is a clean format error."""
    disu = tmp_path / "bad.disu"
    disu.write_text(_disu_text(dimensions="  NODES 2\n  NJA 4\n  NVERT 3\n"))
    with pytest.raises(FileFormatNotSupportedError, match="but has 1 CELL2D"):
        Modflow6DisuDriver().read(str(disu))


def test_reuses_the_disv_block_parser(tmp_path):
    """The inherited DISV helpers build a tiny inline DISU footprint."""
    disu = tmp_path / "tiny.disu"
    disu.write_text(_disu_text())
    parsed = _Modflow6DisuReader(str(disu)).build()
    assert parsed.faces.shape == (1, 3)
    assert parsed.global_attributes["source"] == "MODFLOW 6 DISU"


def test_open_close_defers_vertices_to_an_external_file(tmp_path):
    """A VERTICES block given as OPEN/CLOSE reads the rows from the external file."""
    (tmp_path / "verts.inc").write_text("1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n")
    disu = tmp_path / "ext.disu"
    disu.write_text(_disu_text(vertices="  OPEN/CLOSE verts.inc\n"))
    parsed = _Modflow6DisuReader(str(disu)).build()
    assert parsed.faces.shape == (1, 3)
    assert parsed.node_x.tolist() == [0.0, 1.0, 0.0]


def test_open_close_to_a_missing_file_is_a_format_error(tmp_path):
    """An OPEN/CLOSE pointing at a missing file is a clean format error."""
    disu = tmp_path / "ext.disu"
    disu.write_text(_disu_text(vertices="  OPEN/CLOSE gone.inc\n"))
    with pytest.raises(FileFormatNotSupportedError, match="could not be read"):
        _Modflow6DisuReader(str(disu)).build()


def test_open_close_without_a_file_name_is_a_format_error(tmp_path):
    """An OPEN/CLOSE directive naming no file is a clean format error."""
    disu = tmp_path / "ext.disu"
    disu.write_text(_disu_text(vertices="  OPEN/CLOSE\n"))
    with pytest.raises(FileFormatNotSupportedError, match="no file name"):
        _Modflow6DisuReader(str(disu)).build()


def test_sniff_declines_a_path_that_cannot_be_opened(tmp_path):
    """_sniff returns False on an OS read error; can_open's isfile guard hides it."""
    assert Modflow6DisuDriver()._sniff(str(tmp_path)) is False


def test_cell_count_mismatch_names_the_nodes_dimension(tmp_path):
    """A DISU CELL2D/NODES mismatch names NODES (not DISV's NCPL) in the error."""
    disu = tmp_path / "bad.disu"
    disu.write_text(
        "BEGIN DIMENSIONS\n  NODES 2\n  NJA 6\n  NVERT 3\nEND DIMENSIONS\n\n"
        "BEGIN VERTICES\n  1 0.0 0.0\n  2 1.0 0.0\n  3 0.0 1.0\nEND VERTICES\n\n"
        "BEGIN CELL2D\n  1 0.33 0.33 3 1 2 3\nEND CELL2D\n",
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="NODES 2 but has 1"):
        Modflow6DisuDriver().read(str(disu))
