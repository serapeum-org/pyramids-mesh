"""Tests for the MODFLOW 6 DISV reader (C5), against a real committed sample grid.

A ``.disv`` file is the ASCII input of one MODFLOW 6 model. A genuine grid is committed
under ``tests/data/modflow6/``: ``tri_model_cnst.disv`` is a triangular DISV grid of 200
cells per layer over 4 layers, built on 121 vertices (FloPy ``test003_gwftri_disv``, CC0 /
public domain). The tests read the committed bytes -- the mesh is the single horizontal
footprint of 200 triangular cells.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset, _drivers
from pyramids_mesh.drivers.modflow import Modflow6DisvDriver, _Modflow6DisvReader

_DATA = Path(__file__).parent.parent / "data" / "modflow6"
_DISV = _DATA / "tri_model_cnst.disv"
_GRB = _DATA / "tri_model.disv.grb"


def _read(path):
    """Read a committed DISV file, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Modflow6DisvDriver().read(str(path))


def test_reads_the_horizontal_cell_footprint():
    """tri_model_cnst reads 200 triangular cells on 121 vertices -- one layer."""
    mesh = _read(_DISV).mesh
    assert mesh.n_node == 121
    assert mesh.n_face == 200
    assert mesh.face_node_connectivity.data.shape == (200, 3)


def test_vertex_ids_are_one_based_resolved_to_zero_based():
    """The 1-based icvert ids are resolved to 0-based, and the base recorded."""
    connectivity = _read(_DISV).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 120  # NVERT 121 -> 0..120


def test_faces_are_counter_clockwise():
    """DISV lists cells clockwise; they come back counter-clockwise after validation."""
    mesh = _read(_DISV).mesh
    for row in mesh.face_node_connectivity.data:
        real = row[row != -1]
        xs = mesh.node_x[real]
        ys = mesh.node_y[real]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_cell_centres_become_face_centres():
    """The CELL2D xc/yc columns are carried as the mesh face centres."""
    mesh = _read(_DISV).mesh
    assert mesh.face_x is not None and mesh.face_y is not None
    assert mesh.face_x.shape == (200,)
    # The first cell's centre is 0.33 9.67 in the committed file.
    assert mesh.face_x[0] == pytest.approx(0.33)
    assert mesh.face_y[0] == pytest.approx(9.67)


def test_no_crs_or_elevation_in_a_disv_file():
    """A .disv file carries geometry only, so crs and node_z are absent."""
    dataset = _read(_DISV)
    assert dataset.mesh.crs is None
    assert dataset.get("node_z") is None


def test_can_open_accepts_the_disv_file():
    """can_open accepts the real DISV file on its blocks, not only its suffix."""
    assert Modflow6DisvDriver().can_open(str(_DISV)) is True


def test_can_open_declines_junk(tmp_path):
    """can_open declines a .disv whose content is not DISV, and a directory."""
    driver = Modflow6DisvDriver()
    # right suffix, wrong content -- no DISV blocks
    junk = tmp_path / "fake.disv"
    junk.write_text("this is not a modflow grid\n")
    assert driver.can_open(str(junk)) is False
    # a non-.disv file is declined on its suffix
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False
    # the binary .grb companion is not claimed (suffix is not .disv)
    assert driver.can_open(str(_GRB)) is False
    # a directory is declined
    assert driver.can_open(str(tmp_path)) is False


def test_dispatch_through_read_file_by_driver_name():
    """read_file routes to this driver when it is registered and named.

    The entry point is reported for a maintainer to add to pyproject.toml, so the driver
    is not discovered in the test environment. Injecting it into pyramids' driver cache
    exercises the real read_file resolution path (driver= by name) without depending on a
    reinstall.
    """
    original = _drivers._CACHE
    try:
        registry = dict(original or {})
        registry["modflow6_disv"] = Modflow6DisvDriver()
        _drivers._CACHE = registry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = UgridDataset.read_file(str(_DISV), driver="modflow6_disv")
    finally:
        _drivers._CACHE = original
    assert dataset.mesh.n_face == 200
    assert dataset.mesh.n_node == 121


def test_read_of_a_non_disv_path_is_a_format_error(tmp_path):
    """read() on a path that is not a file is refused as a format error."""
    missing = tmp_path / "gone.disv"
    with pytest.raises(FileFormatNotSupportedError, match="not a MODFLOW 6 DISV file"):
        Modflow6DisvDriver().read(str(missing))


def test_a_cell_count_mismatch_is_a_format_error(tmp_path):
    """A NCPL that disagrees with the CELL2D rows is a clean format error."""
    disv = tmp_path / "bad.disv"
    disv.write_text(
        "BEGIN DIMENSIONS\n"
        "  NLAY 1\n"
        "  NCPL 2\n"
        "  NVERT 3\n"
        "END DIMENSIONS\n\n"
        "BEGIN VERTICES\n"
        "  1 0.0 0.0\n"
        "  2 1.0 0.0\n"
        "  3 0.0 1.0\n"
        "END VERTICES\n\n"
        "BEGIN CELL2D\n"
        "  1 0.33 0.33 3 1 2 3\n"
        "END CELL2D\n"
    )
    with pytest.raises(FileFormatNotSupportedError, match="NCPL 2 but has 1 CELL2D"):
        Modflow6DisvDriver().read(str(disv))


def _disv_text(
    dimensions="  NLAY 1\n  NCPL 1\n  NVERT 3\n",
    vertices="  1 0.0 0.0\n  2 1.0 0.0\n  3 0.0 1.0\n",
    cells="  1 0.33 0.33 3 1 2 3\n",
):
    """Assemble a tiny valid .disv body; override one block to malform it."""
    return (
        f"BEGIN DIMENSIONS\n{dimensions}END DIMENSIONS\n\n"
        f"BEGIN VERTICES\n{vertices}END VERTICES\n\n"
        f"BEGIN CELL2D\n{cells}END CELL2D\n"
    )


def test_sniff_declines_a_path_that_cannot_be_opened(tmp_path):
    """_sniff returns False on an OS read error; can_open's isfile guard hides it."""
    assert Modflow6DisvDriver()._sniff(str(tmp_path)) is False


def test_read_blocks_wraps_an_unreadable_file_as_a_format_error(tmp_path):
    """_read_blocks turns an OS read error into a format error (read's isfile guard precedes it)."""
    missing = tmp_path / "gone.disv"
    with pytest.raises(FileFormatNotSupportedError, match="could not be read"):
        _Modflow6DisvReader(str(missing)).build()


def test_a_non_numeric_dimensions_line_is_skipped(tmp_path):
    """A DIMENSIONS entry whose value is not an integer is ignored, not fatal."""
    disv = tmp_path / "ok.disv"
    disv.write_text(_disv_text(dimensions="  NLAY xyz\n  NCPL 1\n  NVERT 3\n"))
    parsed = _Modflow6DisvReader(str(disv)).build()
    assert parsed.faces.shape == (1, 3)


def test_an_empty_vertices_block_is_a_format_error(tmp_path):
    """A VERTICES block with no entries is a clean format error."""
    disv = tmp_path / "no_vertices.disv"
    disv.write_text(_disv_text(vertices=""))
    with pytest.raises(FileFormatNotSupportedError, match="no VERTICES block entries"):
        Modflow6DisvDriver().read(str(disv))


def test_a_short_vertices_line_is_a_format_error(tmp_path):
    """A VERTICES line that is not '<iv> <x> <y>' is a clean format error."""
    disv = tmp_path / "short_vertex.disv"
    disv.write_text(_disv_text(vertices="  1 0.0\n"))
    with pytest.raises(FileFormatNotSupportedError, match="is not '<iv> <x> <y>'"):
        Modflow6DisvDriver().read(str(disv))


def test_an_out_of_range_vertex_id_is_a_format_error(tmp_path):
    """A vertex id outside the declared NVERT is a clean format error."""
    disv = tmp_path / "oor_vertex.disv"
    disv.write_text(
        _disv_text(dimensions="  NCPL 1\n  NVERT 1\n", vertices="  5 0.0 0.0\n")
    )
    with pytest.raises(
        FileFormatNotSupportedError, match="out of range or non-numeric"
    ):
        Modflow6DisvDriver().read(str(disv))


def test_a_zero_vertex_id_is_a_format_error(tmp_path):
    """A vertex id of 0 would index from the end and silently overwrite; it is refused."""
    disv = tmp_path / "zero_vertex.disv"
    disv.write_text(
        _disv_text(dimensions="  NCPL 1\n  NVERT 1\n", vertices="  0 0.0 0.0\n")
    )
    with pytest.raises(
        FileFormatNotSupportedError, match="out of range or non-numeric"
    ):
        Modflow6DisvDriver().read(str(disv))


def test_an_out_of_range_cell2d_id_is_a_format_error(tmp_path):
    """A CELL2D id outside the declared NCPL is a clean format error, not an IndexError."""
    disv = tmp_path / "oor_cell.disv"
    disv.write_text(_disv_text(cells="  5 0.33 0.33 3 1 2 3\n"))
    with pytest.raises(
        FileFormatNotSupportedError, match="CELL2D id 5 is out of range"
    ):
        Modflow6DisvDriver().read(str(disv))


def test_a_missing_vertex_id_is_a_format_error(tmp_path):
    """Declaring more vertices than appear leaves a gap, which is a format error."""
    disv = tmp_path / "missing_vertex.disv"
    disv.write_text(
        _disv_text(
            dimensions="  NCPL 1\n  NVERT 3\n",
            vertices="  1 0.0 0.0\n  2 1.0 0.0\n",
        )
    )
    with pytest.raises(FileFormatNotSupportedError, match="vertex ids are missing"):
        Modflow6DisvDriver().read(str(disv))


def test_an_empty_cell2d_block_is_a_format_error(tmp_path):
    """A CELL2D block with no entries is a clean format error."""
    disv = tmp_path / "no_cells.disv"
    disv.write_text(_disv_text(cells=""))
    with pytest.raises(FileFormatNotSupportedError, match="no CELL2D block entries"):
        Modflow6DisvDriver().read(str(disv))


def test_a_short_cell2d_line_is_a_format_error(tmp_path):
    """A CELL2D line with fewer than five fields is a clean format error."""
    disv = tmp_path / "short_cell.disv"
    disv.write_text(_disv_text(cells="  1 0.33 0.33 3\n"))
    with pytest.raises(FileFormatNotSupportedError, match="<icell2d> <xc> <yc>"):
        Modflow6DisvDriver().read(str(disv))


def test_a_non_numeric_cell2d_line_is_a_format_error(tmp_path):
    """A CELL2D line with non-numeric fields is a clean format error."""
    disv = tmp_path / "nan_cell.disv"
    disv.write_text(_disv_text(cells="  1 x y 3 1 2 3\n"))
    with pytest.raises(FileFormatNotSupportedError, match="is non-numeric"):
        Modflow6DisvDriver().read(str(disv))


def test_a_cell2d_line_with_too_few_vertices_is_a_format_error(tmp_path):
    """A CELL2D ncvert below three (not a polygon) is a clean format error."""
    disv = tmp_path / "degenerate_cell.disv"
    disv.write_text(_disv_text(cells="  1 0.33 0.33 2 1 2\n"))
    with pytest.raises(FileFormatNotSupportedError, match="at least three"):
        Modflow6DisvDriver().read(str(disv))


def test_stray_and_short_lines_are_ignored(tmp_path):
    """A line outside any block and a one-token DIMENSIONS line are both skipped."""
    disv = tmp_path / "noise.disv"
    disv.write_text(
        "stray line outside any block\n"
        "BEGIN DIMENSIONS\n"
        "  SOMEFLAG\n"
        "  NCPL 1\n"
        "  NVERT 3\n"
        "END DIMENSIONS\n\n"
        "BEGIN VERTICES\n"
        "  1 0.0 0.0\n  2 1.0 0.0\n  3 0.0 1.0\n"
        "END VERTICES\n\n"
        "BEGIN CELL2D\n"
        "  1 0.33 0.33 3 1 2 3\n"
        "END CELL2D\n"
    )
    parsed = _Modflow6DisvReader(str(disv)).build()
    assert parsed.faces.shape == (1, 3)
