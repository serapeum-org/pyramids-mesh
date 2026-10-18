"""Tests for the FLO-2D reader (D10), against a real committed project directory.

A FLO-2D project is a directory of files. A genuine 240-element flume (4 x 60 cells of
50-unit spacing) is committed under ``tests/data/flo2d/``: ``DEPFP.OUT`` (element centres +
max depth) and ``TIMDEP.HDF5`` (600-step results). The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.flo2d import Flo2dDriver, _Flo2dReader

_DATA = Path(__file__).parent.parent / "data" / "flo2d"


def _read(path=_DATA):
    """Read the committed FLO-2D project, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Flo2dDriver().read(str(path))


def test_reads_the_quad_grid():
    """The 240 square cells lift to a 305-node, 240-quad-face mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 305
    assert mesh.n_face == 240
    assert mesh.face_node_connectivity.data.shape == (240, 4)


def test_every_face_is_a_quad():
    """Each cell is a four-node quad -- no fill padding is used."""
    connectivity = _read().mesh.face_node_connectivity
    assert int(connectivity.data.min()) >= 0
    assert not bool((connectivity.data == -1).any())


def test_face_centres_are_the_element_centres():
    """The element coordinates from DEPFP.OUT become the face centres."""
    mesh = _read().mesh
    assert mesh.face_x is not None and mesh.face_y is not None
    assert len(mesh.face_x) == 240


def test_flow_depth_is_a_lazy_face_variable_over_time():
    """FLOW DEPTH is a face-located variable over the 600-step Times axis, lazily loaded."""
    depth = _read().get("FLOW DEPTH")
    assert depth.location == "face"
    assert depth.shape == (600, 240)
    assert len(depth.attributes["time_values"]) == 600
    assert depth.has_data_source
    assert depth._data is None
    assert depth.data.shape == (600, 240)


def test_mins_and_maxs_carried_as_range_attributes():
    """The precomputed Mins/Maxs travel on the variable's attributes."""
    depth = _read().get("FLOW DEPTH")
    assert "minimum" in depth.attributes and "maximum" in depth.attributes
    assert len(depth.attributes["maximum"]) == 600


def test_max_depth_column_is_a_static_face_variable():
    """The DEPFP.OUT max-depth column becomes a static face variable."""
    md = _read().get("max_floodplain_depth")
    assert md.location == "face"
    assert md.shape == (240,)


def test_vector_quantity_is_skipped_scalar_magnitude_kept():
    """The Velocity vector quantity is skipped; the scalar Velocity MAG is kept."""
    names = list(_read().data_variable_names)
    assert "Velocity MAG" in names
    assert "Velocity" not in names


def test_values_are_lazy_and_reopen_per_read():
    """The lazy loader reopens the file each call, so repeated reads all succeed."""
    depth = _read().get("FLOW DEPTH")
    first = depth.data
    second = depth.data
    assert np.array_equal(first, second)


def test_faces_are_counter_clockwise():
    """Every quad face comes back counter-clockwise."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data[:100]:
        xs = mesh.node_x[row]
        ys = mesh.node_y[row]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_can_open_accepts_directory_and_components():
    """can_open accepts the project directory and its DEPFP.OUT / TIMDEP.HDF5 components."""
    driver = Flo2dDriver()
    assert driver.can_open(str(_DATA)) is True
    assert driver.can_open(str(_DATA / "DEPFP.OUT")) is True
    assert driver.can_open(str(_DATA / "TIMDEP.HDF5")) is True


def test_can_open_declines_non_flo2d(tmp_path):
    """can_open declines a lone component, an unrelated file and an empty directory."""
    driver = Flo2dDriver()
    stray = tmp_path / "DEPFP.OUT"  # no TIMDEP.HDF5 sibling
    stray.write_text("1 0 0 0\n")
    assert driver.can_open(str(stray)) is False
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False
    assert driver.can_open(str(tmp_path)) is False


def test_missing_component_is_a_format_error(tmp_path):
    """A directory missing a required component is a clean format error."""
    partial = tmp_path / "proj"
    partial.mkdir()
    (partial / "DEPFP.OUT").write_text("1 0 0 0\n")  # no TIMDEP.HDF5
    with pytest.raises(FileFormatNotSupportedError):
        Flo2dDriver().read(str(partial))


def test_read_of_a_non_flo2d_path_is_a_format_error(tmp_path):
    """read() on a path that is neither a project directory nor a component is refused."""
    other = tmp_path / "notes.txt"
    other.write_text("not flo-2d")
    with pytest.raises(FileFormatNotSupportedError, match="not a FLO-2D project"):
        Flo2dDriver().read(str(other))


def test_malformed_depfp_table_is_a_format_error(tmp_path):
    """A DEPFP.OUT that is not the '<id> <x> <y> <depth>' table is a clean format error."""
    (tmp_path / "DEPFP.OUT").write_text("1\n2\n3\n")  # one column, not four
    with pytest.raises(FileFormatNotSupportedError, match="expected"):
        _Flo2dReader(str(tmp_path), lazy=True)._read_coords()


def test_min_gap_returns_none_for_a_single_distinct_value():
    """``_min_gap`` returns None when fewer than two distinct coordinates exist."""
    assert _Flo2dReader._min_gap(np.array([5.0, 5.0, 5.0])) is None
    assert _Flo2dReader._min_gap(np.array([0.0, 50.0, 100.0])) == 50.0


def test_results_hdf5_without_the_timdep_group_yields_no_time_series(tmp_path):
    """A TIMDEP.HDF5 lacking the results group leaves only the static max-depth variable."""
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(_DATA / "DEPFP.OUT", proj / "DEPFP.OUT")
    # a real HDF5 that genuinely has no 'TIMDEP NETCDF OUTPUT RESULTS' group
    shutil.copy(_DATA.parent / "xdmf" / "out.h5", proj / "TIMDEP.HDF5")
    names = list(_read(proj).data_variable_names)
    assert "max_floodplain_depth" in names
    assert "FLOW DEPTH" not in names


def test_add_quantity_skips_a_group_without_values_or_times():
    """A TIMDEP quantity group missing Values/Times is skipped, adding no variable."""

    class _Group:
        def GetMDArrayNames(self):
            return ["Mins"]  # neither Values nor Times

    class _Results:
        def OpenGroup(self, quantity):
            return _Group()

    variables: dict = {}
    _Flo2dReader(str(_DATA), lazy=True)._add_quantity(
        "unused.hdf5", _Results(), "FLOW DEPTH", 240, variables
    )
    assert variables == {}
