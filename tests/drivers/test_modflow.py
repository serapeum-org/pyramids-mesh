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
from pyramids_mesh.drivers.modflow import Modflow6DisvDriver

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
