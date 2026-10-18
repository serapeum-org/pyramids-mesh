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

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid import _drivers
from pyramids_mesh.drivers.icon import IconDriver, _orient_faces

_GRID = Path(__file__).parent.parent / "data" / "icon" / "icon_grid_R02B04_subset.nc"


def _read():
    """Read the committed ICON grid."""
    return IconDriver().read(str(_GRID))


def test_orient_faces_transposes_the_spec_layout():
    """The ICON ``(nv=3, ncells)`` layout is transposed so each row is one triangle."""
    voc = np.array([[1, 4], [2, 5], [3, 6]])  # (3, 2): two triangles down the columns
    assert _orient_faces(voc).tolist() == [[1, 2, 3], [4, 5, 6]]


def test_orient_faces_accepts_the_transposed_layout():
    """A file already stored ``(ncells, 3)`` is accepted as-is, not transposed again."""
    voc = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    assert _orient_faces(voc).tolist() == [[1, 2, 3], [4, 5, 6]]


def test_orient_faces_refuses_a_non_triangle_shape():
    """A connectivity array with no dimension of 3 is refused, not silently mis-wired."""
    with pytest.raises(FileFormatNotSupportedError):
        _orient_faces(np.zeros((4, 5), dtype=np.int64))


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


class _FakeAttribute:
    """A stand-in GDAL attribute reporting a name and a readable value."""

    def __init__(self, name, value=None, raises=False) -> None:
        self._name = name
        self._value = value
        self._raises = raises

    def GetName(self) -> str:
        """Return the attribute's name."""
        return self._name

    def Read(self):
        """Return the attribute's value, or raise ``RuntimeError`` when so built."""
        if self._raises:
            raise RuntimeError("attribute read failed")
        return self._value


class _FakeArray:
    """A stand-in GDAL MD-array returning a fixed value from ``ReadAsArray``."""

    def __init__(self, values) -> None:
        self._values = values

    def ReadAsArray(self):
        """Return the array's values."""
        return self._values


class _FakeGroup:
    """A stand-in GDAL root group exposing array names, attributes and arrays."""

    def __init__(self, names=(), attributes=(), arrays=None) -> None:
        self._names = list(names)
        self._attributes = list(attributes)
        self._arrays = arrays or {}

    def GetMDArrayNames(self) -> list:
        """Return the root-group array names."""
        return list(self._names)

    def GetAttributes(self) -> list:
        """Return the global attribute handles."""
        return list(self._attributes)

    def OpenMDArray(self, name):
        """Return the named stand-in array."""
        return self._arrays[name]


def test_convention_reads_the_conventions_attribute():
    """_convention returns the ``Conventions`` global attribute when it reads cleanly."""
    from pyramids_mesh.drivers.icon import _convention

    group = _FakeGroup(
        attributes=[
            _FakeAttribute("history", "v1"),
            _FakeAttribute("Conventions", "CF-1.6"),
        ]
    )
    assert _convention(group) == "CF-1.6"


def test_convention_is_empty_when_the_attribute_read_raises():
    """A ``Conventions`` attribute whose Read raises yields an empty string, not an error."""
    from pyramids_mesh.drivers.icon import _convention

    group = _FakeGroup(attributes=[_FakeAttribute("Conventions", raises=True)])
    assert _convention(group) == ""


def test_is_icon_declines_the_mpas_vertex_array():
    """A group carrying MPAS's ``verticesOnCell`` array is not claimed as ICON."""
    from pyramids_mesh.drivers.icon import _is_icon

    group = _FakeGroup(names=["vertex_of_cell", "clon", "vlon", "verticesOnCell"])
    assert _is_icon(group) is False


def test_is_icon_declines_the_mpas_convention():
    """A group declaring the MPAS convention is declined even without the MPAS array."""
    from pyramids_mesh.drivers.icon import _is_icon

    group = _FakeGroup(
        names=["vertex_of_cell", "clon", "vlon"],
        attributes=[_FakeAttribute("Conventions", "MPAS")],
    )
    assert _is_icon(group) is False


def test_face_centres_absent_when_clon_or_clat_is_missing():
    """Cell centres are skipped when either ``clon`` or ``clat`` is absent."""
    group = _FakeGroup(arrays={})
    assert IconDriver._face_centres(group, {"vlon", "clon"}) == (None, None)


def test_variables_empty_when_cell_area_is_missing():
    """No ``cell_area`` array means no face variable."""
    group = _FakeGroup(arrays={})
    assert IconDriver._variables(group, {"vlon"}, 5) == {}


def test_variables_dropped_when_cell_area_has_the_wrong_shape():
    """A ``cell_area`` whose length is not the face count is left out, not mislabelled."""
    group = _FakeGroup(arrays={"cell_area": _FakeArray(np.zeros(3))})
    assert IconDriver._variables(group, {"cell_area"}, 5) == {}
