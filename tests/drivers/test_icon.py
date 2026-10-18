"""Tests for the ICON reader (C7), against a real committed sample.

ICON stores an icosahedral triangular grid: cell centres (``clon``/``clat``), triangle
vertices (``vlon``/``vlat``) and a 1-based ``vertex_of_cell`` table. The fixture is a
genuine Apache-2.0 subset of the DWD R02B04 grid committed under ``tests/data/icon/`` --
the first 320 cells and their 197 vertices -- so the tests read the committed bytes. The
vertices become 197 nodes and ``vertex_of_cell`` the 320 triangular faces.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid import _drivers
from pyramids_mesh.drivers.icon import IconDriver

_GRID = Path(__file__).parent.parent / "data" / "icon" / "icon_grid_R02B04_subset.nc"


def _read():
    """Read the committed ICON grid."""
    return IconDriver().read(str(_GRID))


def test_vertices_and_cells_build_the_mesh():
    """The 197 vertices become nodes and the 320 cells become triangular faces."""
    mesh = _read().mesh
    assert mesh.n_node == 197
    assert mesh.n_face == 320
    # every face is a triangle of three real nodes
    data = mesh.face_node_connectivity.data
    assert data.shape == (320, 3)
    assert bool((data != -1).all())


def test_vertex_of_cell_is_one_based_resolved_to_zero_based():
    """The 1-based vertex_of_cell arrives 0-based; the base is recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    # the fixture's vertices run 1..197 one-based, so 0..196 after re-basing
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 196


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise, as the mesh model requires."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_cell_centres_populate_face_xy():
    """clon/clat become the face centres, one per cell."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert mesh.face_y is not None
    assert len(mesh.face_x) == 320


def test_cell_area_is_a_face_variable():
    """cell_area is carried as a static face-located variable."""
    area = _read().get("cell_area")
    assert area.location == "face"
    assert area.shape == (320,)
    assert float(area.data.min()) > 0


def test_coordinates_are_kept_in_radians():
    """Coordinates are preserved in ICON's native radians, not converted to degrees.

    A degree conversion would push latitudes past 90; the fixture is a polar cap, so the
    radian latitudes stay within +/- pi/2 and a degree reading would not.
    """
    mesh = _read().mesh
    assert float(np.abs(mesh.node_y).max()) <= np.pi / 2 + 1e-9


def test_crs_is_none():
    """ICON declares a sphere but no authority, so the mesh carries no CRS."""
    assert _read().mesh.crs is None


def test_can_open_accepts_icon_and_declines_non_icon_and_junk(tmp_path):
    """can_open accepts the ICON grid and declines a non-ICON .nc and non-NetCDF junk."""
    driver = IconDriver()
    assert driver.can_open(str(_GRID)) is True

    junk = tmp_path / "plain.nc"
    junk.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_can_open_declines_an_mpas_lookalike(tmp_path):
    """A .nc carrying the MPAS cell-vertex array is declined, not claimed as ICON.

    ICON and MPAS both use ``.nc``; the discriminator keys on ICON's ``vertex_of_cell``
    plus ``clon``/``vlon`` and rejects the MPAS ``verticesOnCell`` array, so a file
    renamed or mislabelled cannot be claimed by both.
    """
    from osgeo import gdal

    path = tmp_path / "mpas.nc"
    with gdal.ExceptionMgr(useExceptions=True):
        drv = gdal.GetDriverByName("netCDF")
        dataset = drv.CreateMultiDimensional(str(path))
        root = dataset.GetRootGroup()
        cell = root.CreateDimension("nCells", "", "", 2)
        vertex = root.CreateDimension("maxEdges", "", "", 3)
        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        i32 = gdal.ExtendedDataType.Create(gdal.GDT_Int32)
        root.CreateMDArray("lonCell", [cell], f64)
        root.CreateMDArray("verticesOnCell", [cell, vertex], i32)
        dataset.Close()

    assert IconDriver().can_open(str(path)) is False


def test_reached_through_read_file_netcdf_fallback():
    """A plain read_file dispatches to this driver via the NetCDF-suffix fallback.

    The driver is not registered through an entry point in this change, so the registry
    cache is primed with it directly -- the same mechanism pyramids' own driver doctests
    use -- to exercise the suffix-fallback dispatch path end to end.
    """
    _drivers.clear_driver_cache()
    _drivers._CACHE = {"icon": IconDriver()}
    try:
        dataset = UgridDataset.read_file(str(_GRID))
    finally:
        _drivers.clear_driver_cache()
    assert dataset.mesh.n_node == 197
    assert dataset.mesh.n_face == 320


def test_require_refuses_a_file_missing_a_defining_array():
    """_require raises a clear error when a defining ICON array is missing."""
    from pyramids.base._errors import FileFormatNotSupportedError

    with pytest.raises(FileFormatNotSupportedError, match="not an ICON grid"):
        IconDriver._require("partial.nc", {"vlon", "vlat"})  # vertex_of_cell missing
