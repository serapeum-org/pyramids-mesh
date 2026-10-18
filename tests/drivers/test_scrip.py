"""Tests for the SCRIP reader, against a real committed sample.

SCRIP describes a remapping grid by each cell's corner coordinates; this driver
deduplicates the corners into a shared node list and builds face-node connectivity into
it. The fixture is a genuine cubed-sphere ne8 SCRIP grid committed under
``tests/data/scrip/`` (Apache-2.0, UXarray test data): 384 cells of four corners each
(``grid_corner_lat``/``grid_corner_lon``), with ``grid_center_lat``/``grid_center_lon``
the cell centres. The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.scrip import _ScripReader, _array_units, ScripDriver

_DATA = Path(__file__).parent.parent / "data"
_GRID = _DATA / "scrip" / "outCSne8.nc"
_ESMF = _DATA / "esmf" / "ne4np4-esmf.nc"
_GEOS = _DATA / "geos" / "test-c12.native.nc4"
_MPAS = _DATA / "mpas" / "mesh.QU.1920km.151026.nc"
_ICON = _DATA / "icon" / "icon_grid_R02B04_subset.nc"


def _read():
    """Read the committed SCRIP grid, ignoring any winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ScripDriver().read(str(_GRID))


def test_corners_deduplicate_into_a_shared_node_list():
    """The 1536 corner entries collapse to 407 shared nodes over 384 cells."""
    mesh = _read().mesh
    assert mesh.n_face == 384
    assert mesh.n_node == 407  # far fewer than 384 x 4 repeated corners
    data = mesh.face_node_connectivity.data
    assert data.shape == (384, 4)
    assert int(data.min()) == 0  # 0-based node indices
    assert int(data.max()) == 406


def test_connectivity_base_is_recorded_as_zero():
    """The dedup produces 0-based indices, so the writer base is zero."""
    assert _read().mesh.face_node_connectivity.original_start_index == 0


def test_cubed_sphere_cells_are_quads():
    """Every ne8 cell is a four-corner quad; none padded in this sample."""
    data = _read().mesh.face_node_connectivity.data
    widths = (data != -1).sum(axis=1)
    counts = {
        int(width): int(count)
        for width, count in zip(*np.unique(widths, return_counts=True))
    }
    assert counts == {4: 384}


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
    """grid_center_lon/lat become the face centres, one per cell."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert mesh.face_y is not None
    assert len(mesh.face_x) == 384


def test_coordinates_are_longitude_latitude_in_degrees():
    """The sample carries degrees, so latitudes span the poles and longitudes 0..360."""
    mesh = _read().mesh
    assert mesh.node_y.min() >= -90.0 - 1e-6
    assert mesh.node_y.max() <= 90.0 + 1e-6
    assert mesh.node_x.min() >= -1e-6
    assert mesh.node_x.max() <= 360.0 + 1e-6


def test_crs_is_none_on_the_sphere():
    """The grid is lon/lat on a sphere with no EPSG authority, so crs is None."""
    assert _read().mesh.crs is None


def test_can_open_accepts_scrip_and_declines_junk(tmp_path):
    """can_open accepts the SCRIP grid and declines non-NetCDF junk and a directory."""
    driver = ScripDriver()
    assert driver.can_open(str(_GRID)) is True

    junk = tmp_path / "plain.nc"
    junk.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_can_open_declines_every_sibling_nc_format():
    """The SCRIP sniff declines the ESMF, GEOS, MPAS and ICON .nc samples."""
    driver = ScripDriver()
    assert driver.can_open(str(_ESMF)) is False
    assert driver.can_open(str(_GEOS)) is False
    assert driver.can_open(str(_MPAS)) is False
    assert driver.can_open(str(_ICON)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else -- topology only."""
    assert set(ScripDriver().capabilities) == {"read_mesh"}


def test_radian_corners_are_converted_to_degrees():
    """A ``units`` of radians is honoured, so the corner read comes back in degrees."""
    from pyramids_mesh.drivers.scrip import _to_degrees

    radians = np.array([0.0, np.pi / 2, np.pi])
    assert np.allclose(_to_degrees(radians, "radians"), [0.0, 90.0, 180.0])
    # a degrees (or absent) unit is left untouched
    assert np.allclose(_to_degrees(radians, "degrees"), radians)


def test_require_refuses_a_missing_corner_array():
    """``_require`` raises when a defining corner array is absent from the file."""
    with pytest.raises(FileFormatNotSupportedError, match="not a SCRIP grid"):
        _ScripReader("absent.nc")._require(set())


def test_pad_repeated_corners_marks_duplicate_as_fill():
    """A corner repeating an earlier one in its cell becomes ``-1`` padding."""
    connectivity = np.array([[0, 1, 2, 2]])  # a triangle padded by a repeat
    padded = _ScripReader._pad_repeated_corners(connectivity)
    assert padded.tolist() == [[0, 1, 2, -1]]


def test_face_centres_absent_without_both_arrays():
    """Only one of the centre arrays yields no centres rather than a half pair."""
    face_x, face_y = _ScripReader._face_centres(None, {"grid_corner_lat"})
    assert face_x is None
    assert face_y is None


class _FakeAttribute:
    """A stand-in GDAL attribute reporting its name and a readable value."""

    def __init__(self, name: str, value=None, raises: bool = False) -> None:
        self._name = name
        self._value = value
        self._raises = raises

    def GetName(self) -> str:
        """Return the attribute's name."""
        return self._name

    def Read(self):
        """Return the stored value, or raise ``RuntimeError`` when unreadable."""
        if self._raises:
            raise RuntimeError("attribute value unreadable")
        return self._value


class _FakeArray:
    """A stand-in GDAL MD-array exposing a fixed attribute list."""

    def __init__(self, attributes: list) -> None:
        self._attributes = attributes

    def GetAttributes(self) -> list:
        """Return the array's attribute handles."""
        return self._attributes


class _FakeGroup:
    """A stand-in GDAL root group mapping array names to fake arrays."""

    def __init__(self, arrays: dict) -> None:
        self._arrays = arrays

    def OpenMDArray(self, name: str) -> _FakeArray:
        """Return the named fake array."""
        return self._arrays[name]


def test_array_units_reads_a_present_units_string():
    """``_array_units`` returns the matching ``units`` attribute as a string."""
    group = _FakeGroup(
        {"grid_corner_lat": _FakeArray([_FakeAttribute("units", "radians")])}
    )
    assert _array_units(group, "grid_corner_lat") == "radians"


def test_array_units_is_empty_when_read_raises():
    """A ``units`` attribute whose ``Read`` raises is reported as an empty string."""
    group = _FakeGroup(
        {"grid_corner_lat": _FakeArray([_FakeAttribute("units", raises=True)])}
    )
    assert _array_units(group, "grid_corner_lat") == ""


def test_node_coordinates_keep_full_precision_not_the_rounded_key():
    """Dedup rounds only the matching key; stored node coords keep full precision."""
    lon = np.array([[0.123456789, 1.0, 0.0]])
    lat = np.array([[0.987654321, 0.0, 1.0]])
    node_x, node_y, _faces = _ScripReader(str(_GRID))._dedup(lon, lat)
    # np.unique sorts the node order, so assert the exact value survives somewhere,
    # not at a fixed index, and that it was not rounded to the 6-decimal key.
    assert 0.123456789 in node_x.tolist()
    assert 0.987654321 in node_y.tolist()
