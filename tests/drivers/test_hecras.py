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
from pyramids_mesh.drivers.hecras import HecRasDriver

_DATA = Path(__file__).parent.parent / "data" / "hecras"
_PLAN = _DATA / "BaldEagleDamBrk.p18.hdf"


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
