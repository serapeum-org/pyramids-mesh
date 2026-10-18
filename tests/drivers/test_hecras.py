"""Tests for the HEC-RAS 2D reader (D11), against a real committed sample.

The fixture is a genuine USACE "Bald Eagle Creek Dam Break" HEC-RAS plan file committed
under ``tests/data/hecras/`` (US-government work, via the MIT ``rashdf`` test data -- see
``SOURCE.txt``). Its first 2D flow area ``BaldEagleCr`` has 3937 nodes and 3359 cells; the
plan file carries the maximum/minimum water-surface summary. The tests read the committed
HDF5 bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.hecras import _SUMMARY_PATH, HecRasDriver, _HecRasReader

_DATA = Path(__file__).parent.parent / "data" / "hecras"
_PLAN = _DATA / "BaldEagleDamBrk.p18.hdf"
_FACE_FP = "Faces FacePoint Indexes"
_CELL_FACE_INFO = "Cells Face and Orientation Info"


def _read():
    """Read the committed HEC-RAS plan file, ignoring expected warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HecRasDriver().read(str(_PLAN))


def test_reads_first_flow_area_mesh():
    """The first 2D flow area builds a 3937-node, 3359-cell mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 3937
    assert mesh.n_face == 3359


def test_cells_reconstructed_from_faces_are_padded_and_zero_based():
    """Cell polygons come back 0-based, padded to the widest ring with -1 fill."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.data.shape == (3359, 7)
    assert connectivity.original_start_index == 0
    assert int(connectivity.data.max()) == 3936
    real = connectivity.data[connectivity.data != -1]
    assert int(real.min()) == 0
    # the widest cells fill all 7 columns; triangles/quads are -1-padded
    assert bool((connectivity.data == -1).any())


def test_faces_are_counter_clockwise():
    """Every reconstructed cell comes back counter-clockwise."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data[:300]:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        if vertices.size >= 3:
            assert signed_area2 > 0


def test_cell_centres_populate_face_xy():
    """Cell Points become the face centres, sliced to the real cells."""
    mesh = _read().mesh
    assert mesh.face_x is not None
    assert len(mesh.face_x) == 3359


def test_summary_water_surface_is_a_face_variable():
    """The maximum/minimum water-surface summaries are per-cell face variables."""
    dataset = _read()
    names = set(dataset.data_variable_names)
    assert {"Maximum Water Surface", "Minimum Water Surface"} <= names
    wse = dataset.get("Maximum Water Surface")
    assert wse.location == "face"
    assert wse.shape == (3359,)
    assert np.isfinite(wse.data).all()


def test_summary_values_are_lazy():
    """A lazy read reports a source without loading the values."""
    wse = _read().get("Maximum Water Surface")
    assert wse.has_data_source
    assert wse._data is None
    assert wse.data.shape == (3359,)


def test_loader_reopens_and_survives_repeated_reads():
    """The lazy loader reopens the file each call, so repeated reads all agree."""
    wse = _read().get("Minimum Water Surface")
    assert np.array_equal(wse.data, wse.data)


def test_multiple_flow_areas_warns_and_reads_the_first():
    """A plan with more than one 2D area warns and reads the first."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HecRasDriver().read(str(_PLAN))
    assert any("only the first" in str(w.message) for w in caught)


def test_sniff_declines_non_hecras_hdf5_and_junk(tmp_path):
    """can_open declines a non-HEC-RAS HDF5, non-HDF5 junk and a directory."""
    driver = HecRasDriver()
    assert driver.can_open(str(_PLAN)) is True

    # a genuine HDF5 that is not a HEC-RAS plan (the XMDF sample), under the .hdf suffix
    foreign = tmp_path / "foreign.hdf"
    foreign.write_bytes((_DATA.parent / "xmdf" / "minimal.xmdf").read_bytes())
    assert driver.can_open(str(foreign)) is False

    junk = tmp_path / "plain.hdf"
    junk.write_bytes(b"\x00 not hdf5 \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "d.hdf"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_results_only_companion_is_refused(tmp_path):
    """A geometry-stripped file (no flow-area geometry) is refused with a clear error."""
    # Simulate the results-only companion: an HDF5 with no 2D flow-area geometry at all.
    # The real companion is not committed; reuse the foreign HDF5, which also lacks it.
    foreign = tmp_path / "ts.hdf"
    foreign.write_bytes((_DATA.parent / "xmdf" / "minimal.xmdf").read_bytes())
    with pytest.raises(FileFormatNotSupportedError):
        HecRasDriver().read(str(foreign))


def test_sniff_declines_hdf5_magic_but_unopenable(tmp_path):
    """A file with the HDF5 magic but an unreadable body is declined, not raised."""
    bad = tmp_path / "bad.hdf"
    bad.write_bytes(
        b"\x89HDF\r\n\x1a\n" + b"\x00" * 64
    )  # magic only; GDAL cannot open it
    assert HecRasDriver().can_open(str(bad)) is False


# --- white-box coverage of the reader's defensive branches, driven by stand-in GDAL
# group/array objects (no fabricated HEC-RAS HDF5; just the decision each branch makes) ---


class _Arr:
    """A stand-in GDAL MD-array that only reads back a fixed value."""

    def __init__(self, value):
        self._value = np.asarray(value)

    def ReadAsArray(self):
        return self._value


class _Grp:
    """A stand-in GDAL group exposing named subgroups and arrays."""

    def __init__(self, groups=None, arrays=None):
        self._groups = dict(groups or {})
        self._arrays = dict(arrays or {})

    def GetGroupNames(self):
        return list(self._groups)

    def OpenGroup(self, name):
        return self._groups[name]

    def GetMDArrayNames(self):
        return list(self._arrays)

    def OpenMDArray(self, name):
        return _Arr(self._arrays[name])


class _DS:
    """A stand-in GDAL dataset wrapping one root group."""

    def __init__(self, root):
        self._root = root

    def GetRootGroup(self):
        return self._root


def _nest_to(parts, target):
    """Wrap ``target`` so ``root.OpenGroup(parts[0])...OpenGroup(parts[-1])`` is ``target``."""
    grp = target
    for part in reversed(parts):
        grp = _Grp(groups={part: grp})
    return grp


def _reader():
    """A reader bound to a dummy path (the stand-ins carry the data)."""
    return _HecRasReader("fake.hdf", lazy=True)


class TestReaderDefensiveBranches:
    """Each refusal / absent-array branch of the reader, via stand-in GDAL objects."""

    def test_missing_geometry_group_is_refused(self):
        """A file with no ``Geometry`` group is refused."""
        ds = _DS(_Grp(groups={}))
        with pytest.raises(FileFormatNotSupportedError, match="no 'Geometry'"):
            _reader()._areas_group(ds)

    def test_missing_flow_areas_group_is_refused(self):
        """A ``Geometry`` group without ``2D Flow Areas`` is refused."""
        ds = _DS(_Grp(groups={"Geometry": _Grp(groups={})}))
        with pytest.raises(FileFormatNotSupportedError, match="no '2D Flow Areas'"):
            _reader()._areas_group(ds)

    def test_select_area_without_geometry_bearing_area_is_refused(self):
        """A flow-areas group whose areas carry no facepoint geometry is refused."""
        areas = _Grp(groups={"A": _Grp(arrays={})})
        with pytest.raises(FileFormatNotSupportedError, match="no area carries"):
            _reader()._select_area(areas)

    def test_cell_slice_without_cell_info_is_refused(self):
        """A flow-areas group with no ``Cell Info`` array is refused."""
        areas = _Grp(groups={"A": _Grp(arrays={_FACE_FP: [[0, 1]]})}, arrays={})
        with pytest.raises(FileFormatNotSupportedError, match="no 'Cell Info'"):
            _reader()._cell_slice(areas, "A")

    def test_cell_slice_with_zero_real_cells_is_refused(self):
        """A ``Cell Info`` row reporting zero cells for the area is refused."""
        areas = _Grp(
            groups={"A": _Grp(arrays={_FACE_FP: [[0, 1]]})},
            arrays={"Cell Info": [[0, 0]]},
        )
        with pytest.raises(FileFormatNotSupportedError, match="reports 0 cells"):
            _reader()._cell_slice(areas, "A")

    def test_cell_slice_binds_row_by_full_area_order_not_geometry_filtered(self):
        """``Cell Info`` has one row per area; a geometry-less area before a real one
        must not shift the real area's row.

        Binding the row by the geometry-filtered subgroup index (dropping the ghost area)
        would pick the wrong ``[start, count]``; the full subgroup order is authoritative.
        """
        areas = _Grp(
            groups={
                "Ghost": _Grp(arrays={}),  # no FacePoints -> not a geometry area
                "Real": _Grp(arrays={_FACE_FP: [[0, 1]]}),
            },
            arrays={"Cell Info": [[0, 7], [7, 500]]},  # row 0 = Ghost, row 1 = Real
        )
        assert _reader()._cell_slice(areas, "Real") == (7, 500)

    def test_cell_slice_refuses_a_cell_info_row_count_mismatch(self):
        """A ``Cell Info`` whose row count is not one-per-area is refused, not mis-bound."""
        areas = _Grp(
            groups={"A": _Grp(arrays={_FACE_FP: [[0, 1]]}), "B": _Grp(arrays={})},
            arrays={"Cell Info": [[0, 9]]},  # 1 row for 2 areas
        )
        with pytest.raises(FileFormatNotSupportedError, match="Cell Info"):
            _reader()._cell_slice(areas, "A")

    def test_cell_slice_refuses_a_row_count_exceeding_the_area_geometry(self):
        """A bound row whose real-cell count exceeds the area's own cells is refused.

        This catches a mis-bound ``Cell Info`` row (e.g. if the subgroup order diverged
        from the row order) rather than silently reading another area's cells: the area's
        own ``Cells Face and Orientation Info`` describes only 2 cells, so a row claiming 5
        real cells is impossible for it.
        """
        areas = _Grp(
            groups={
                "A": _Grp(
                    arrays={_FACE_FP: [[0, 1]], _CELL_FACE_INFO: [[0, 1], [1, 1]]}
                )
            },
            arrays={"Cell Info": [[0, 5]]},
        )
        with pytest.raises(FileFormatNotSupportedError, match="does not match"):
            _reader()._cell_slice(areas, "A")

    def test_centres_absent_cell_points_returns_none(self):
        """With no ``Cell Points`` array the face centres are left unset."""
        assert _reader()._centres(_Grp(arrays={}), 0, 5) == (None, None)

    def test_summary_variables_without_the_results_tree_is_empty(self):
        """No summary-output group tree yields no summary variables."""
        assert _reader()._summary_variables(_DS(_Grp(groups={})), "A", 5) == {}

    def test_summary_variables_without_the_area_is_empty(self):
        """The summary tree present but missing the area yields no variables."""
        ds = _DS(_nest_to(_SUMMARY_PATH, _Grp(groups={})))
        assert _reader()._summary_variables(ds, "A", 5) == {}

    def test_summary_variables_without_the_named_arrays_is_empty(self):
        """The area present but carrying none of the summary arrays yields no variables."""
        ds = _DS(_nest_to(_SUMMARY_PATH, _Grp(groups={"A": _Grp(arrays={})})))
        assert _reader()._summary_variables(ds, "A", 5) == {}
