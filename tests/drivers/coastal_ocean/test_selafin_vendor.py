"""Tests for the TELEMAC Selafin driver against real files, not fixtures of our own.

Every other Selafin test builds its fixture with ``write_selafin`` and reads it back, so the
reader and the writer share one set of assumptions about the record layout and cannot
contradict each other. These tests break that circle: they run the reader over files written
by TELEMAC itself and by BlueKenue, and pin the numbers those files imply.

The decisive one is the ``IKLE`` row-versus-column order, which the format specifies row-major
(one element's ``NDP`` node ids contiguous) but which no fixture could confirm, because a
transposed writer paired with a transposed reader round-trips perfectly. A real file settles
it: read row-major the elements tessellate the domain exactly once, and read column-major they
do not.

The four samples are committed under ``tests/data/selafin/`` like every other fixture, so these
run offline in the ordinary suite with no marker and no download -- a check that only runs when
someone remembers to ask for it is not a check. ``tests/data/selafin/README.md`` records each
file's source, licence and what it covers.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MeshDataset, UnsupportedMeshFormatError

from pyramids_mesh.drivers.coastal_ocean.telemac import SelafinDriver

from .selafin_header import read_header

_SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "selafin"


@pytest.fixture
def sample():
    """Return a resolver from a sample's name to its committed path.

    A fixture rather than a module constant so a missing file fails the test that needs it
    with the path it looked for, instead of at import time for the whole module.

    Returns:
        Callable[[str], str]: maps a file name under ``tests/data/selafin`` to its path.
    """

    def resolve(name: str) -> str:
        path = _SAMPLE_DIR / name
        assert path.exists(), f"missing committed sample: {path}"
        return str(path)

    return resolve


def _signed_areas(dataset: MeshDataset) -> np.ndarray:
    """Return the signed area of every triangular face, in node order."""
    connectivity = np.asarray(dataset.mesh.face_node_connectivity.data)
    x = np.asarray(dataset.mesh.node_x)[connectivity]
    y = np.asarray(dataset.mesh.node_y)[connectivity]
    return 0.5 * (
        x[:, 0] * (y[:, 1] - y[:, 2])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 0] - y[:, 1])
    )


def _transposed_areas(dataset: MeshDataset) -> np.ndarray:
    """Return the signed areas the file would give if ``IKLE`` were read column-major.

    Re-flattens the connectivity to the order it is stored in and reshapes it the other way
    round, which is what a transposed reader would have produced.
    """
    connectivity = np.asarray(dataset.mesh.face_node_connectivity.data)
    n_face, ndp = connectivity.shape
    flat = connectivity.reshape(-1)
    transposed = flat.reshape(ndp, n_face).T
    x = np.asarray(dataset.mesh.node_x)[transposed]
    y = np.asarray(dataset.mesh.node_y)[transposed]
    return 0.5 * (
        x[:, 0] * (y[:, 1] - y[:, 2])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 0] - y[:, 1])
    )


def _bbox_area(dataset: MeshDataset) -> float:
    """Return the area of the dataset's bounding box."""
    min_x, min_y, max_x, max_y = dataset.bounds
    return float((max_x - min_x) * (max_y - min_y))


def test_telemac_2d_output_reads_with_its_header_counts(sample):
    """A real TELEMAC 2D result file reads with the counts, variables and units it declares."""
    dataset = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")

    assert (dataset.mesh.n_node, dataset.mesh.n_face) == (648, 1030)
    assert dataset.variable_names == [
        "VELOCITY U",
        "VELOCITY V",
        "WATER DEPTH",
        "FREE SURFACE",
        "BOTTOM",
        # Not a result series: the IPOBO boundary numbering, carried through the read.
        "boundary_node_number",
    ]
    assert dataset.get("VELOCITY U").units == "M/S"
    assert dataset.get("VELOCITY U").location == "node"
    assert dataset.get("VELOCITY U").data.shape == (17, 648)
    assert dataset.bounds == pytest.approx((-25000.0, -500.0, 25000.0, 500.0))
    np.testing.assert_allclose(
        np.asarray(dataset.time_values)[:4], [0.0, 10000.0, 20000.0, 30000.0]
    )


def test_ikle_is_read_row_major_on_a_telemac_file(sample):
    """Row-major ``IKLE`` tessellates a real TELEMAC domain exactly once; column-major does not.

    This is the check no hand-built fixture can make: a transposed writer feeding a transposed
    reader agrees with itself. Against a file TELEMAC wrote, the row-major reading gives 1030
    counter-clockwise triangles whose areas sum to the domain's 50 km x 1 km footprint, while the
    column-major reading folds the mesh onto itself -- 537 inverted triangles covering 89 times
    the domain.
    """
    dataset = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")

    areas = _signed_areas(dataset)
    assert (areas > 0).all(), "every triangle should be counter-clockwise"
    assert np.abs(areas).sum() == pytest.approx(_bbox_area(dataset), rel=1e-6)

    transposed = _transposed_areas(dataset)
    assert (transposed < 0).sum() == 537
    assert np.abs(transposed).sum() > 80 * _bbox_area(dataset)


def test_every_node_of_a_telemac_file_is_referenced(sample):
    """No node is orphaned and no index escapes the declared range."""
    dataset = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")

    connectivity = np.asarray(dataset.mesh.face_node_connectivity.data)
    assert len(np.unique(connectivity)) == 648
    assert connectivity.min() == 0
    assert connectivity.max() == 647
    assert dataset.mesh.face_node_connectivity.original_start_index == 1


def test_bluekenue_written_geometry_reads_the_same_way(sample):
    """A file from a second, independent writer confirms the layout and the row-major order.

    ``qgismesh.slf`` was written by BlueKenue / QGIS rather than by TELEMAC, is big-endian, holds
    geometry only (one ``BOTTOM`` field, one step) and sits in projected UTM coordinates -- so it
    also covers the large-coordinate case the synthetic unit-square fixture cannot.
    """
    dataset = MeshDataset.read_file(sample("qgismesh.slf"), driver="selafin")

    assert (dataset.mesh.n_node, dataset.mesh.n_face) == (12814, 25007)
    assert dataset.variable_names == ["BOTTOM", "boundary_node_number"]
    assert dataset.bounds[0] == pytest.approx(315120.22, abs=0.01)
    assert dataset.bounds[3] == pytest.approx(5346307.5, abs=0.01)

    areas = _signed_areas(dataset)
    assert (areas > 0).all()
    assert np.abs(_transposed_areas(dataset)).sum() > 1000 * np.abs(areas).sum()


def test_crs_is_never_inferred_from_projected_coordinates(sample):
    """UTM-magnitude coordinates do not tempt the driver into guessing a CRS."""
    dataset = MeshDataset.read_file(sample("qgismesh.slf"), driver="selafin")

    assert dataset.crs is None


def test_a_genuine_layered_3d_file_is_refused(sample):
    """A real 3D file (``NDP=6`` prisms, 5 planes) is refused rather than read as flat 2D."""
    path = sample("r3d_bump.slf")

    with pytest.raises(UnsupportedMeshFormatError, match="layered 3D"):
        MeshDataset.read_file(path, driver="selafin")


def test_a_single_layer_extracted_from_a_3d_run_reads_as_2d(sample):
    """One layer extracted from a 3D run is a 2D triangular mesh and reads as one.

    ``r3d_bump_extracted_bottom_layer.slf`` carries ``IPARAM[6] = 5`` from the 3D run it came
    from, but its elements are ``NDP=3`` triangles: 2620 of them over 1452 nodes, tessellating
    the flume's 21.2 x 2.0 footprint exactly once. Keying the guard on the plane count refused
    it; keying it on ``NDP`` reads it.
    """
    dataset = MeshDataset.read_file(
        sample("r3d_bump_extracted_bottom_layer.slf"), driver="selafin"
    )

    assert (dataset.mesh.n_node, dataset.mesh.n_face) == (1452, 2620)
    areas = _signed_areas(dataset)
    assert (areas > 0).all()
    assert np.abs(areas).sum() == pytest.approx(41.92, rel=1e-4)
    assert dataset.get("ELEVATION Z").data.shape == (3, 1452)


def test_header_metadata_survives_the_read(sample):
    """The title, the start date and the boundary numbering reach the dataset.

    All three are in the header and were previously read only to keep the record count
    straight. Without the start date a series of seconds has no anchor, and the boundary
    numbering is an ordering that cannot be rederived from the topology.
    """
    dataset = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")

    assert dataset.global_attributes["title"] == "Sloped flume Rouse profile test"
    assert dataset.global_attributes["start_date"] == "1900-01-01T00:00:00"
    assert dataset.global_attributes["n_boundary_nodes"] == 264

    boundary = np.asarray(dataset.get("boundary_node_number").data)
    assert boundary.shape == (648,)
    assert np.count_nonzero(boundary) == 264
    assert boundary.max() == 264


def test_a_file_without_a_date_record_has_no_start_date(sample):
    """``start_date`` is absent rather than invented when the file declares no date.

    ``qgismesh.slf`` has ``IPARAM[9] = 0``, so there is no date record to anchor anything to
    and no key is added. Its title survives, short as it is.
    """
    dataset = MeshDataset.read_file(sample("qgismesh.slf"), driver="selafin")

    assert "start_date" not in dataset.global_attributes
    assert dataset.global_attributes["title"] == "new Selafin"


def test_the_boundary_record_is_read_even_when_iparam_claims_none(sample):
    """``IPOBO`` is read from its record, never trusted to ``IPARAM[7]``.

    BlueKenue writes a populated boundary record while leaving the boundary-point count at
    zero: ``qgismesh.slf`` has ``IPARAM[7] = 0`` and 619 non-zero ``IPOBO`` entries. Keying
    the read on the count would silently drop a real boundary numbering, so the record is
    always read and the count recomputed from it.
    """
    path = sample("qgismesh.slf")

    assert read_header(path)["iparam"][7] == 0

    dataset = MeshDataset.read_file(path, driver="selafin")
    boundary = np.asarray(dataset.get("boundary_node_number").data)
    assert np.count_nonzero(boundary) == 619
    assert dataset.global_attributes["n_boundary_nodes"] == 619


def test_can_open_accepts_what_read_accepts_and_declines_what_it_refuses(sample):
    """``can_open`` answers for readability, not merely for the format.

    Auto-detection trusts it, so a file the read refuses must not be claimed here: the
    layered 3D file is declined, while both writers' 2D files and the single-layer
    extraction are accepted.
    """
    driver = SelafinDriver()

    assert driver.can_open(sample("r2d_tidal_flats.slf")) is True
    assert driver.can_open(sample("qgismesh.slf")) is True
    assert driver.can_open(sample("r3d_bump_extracted_bottom_layer.slf")) is True
    assert driver.can_open(sample("r3d_bump.slf")) is False


def test_round_trip_of_a_real_file_preserves_every_value(tmp_path, sample):
    """Write a real file back out and read it again: mesh, times and values all survive."""
    source = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")
    out = tmp_path / "round_trip.slf"
    source.to_file(str(out), driver="selafin")
    back = MeshDataset.read_file(str(out), driver="selafin")

    assert (back.mesh.n_node, back.mesh.n_face) == (648, 1030)
    np.testing.assert_array_equal(
        np.asarray(source.mesh.face_node_connectivity.data),
        np.asarray(back.mesh.face_node_connectivity.data),
    )
    np.testing.assert_allclose(source.mesh.node_x, back.mesh.node_x)
    np.testing.assert_allclose(source.mesh.node_y, back.mesh.node_y)
    np.testing.assert_allclose(
        np.asarray(source.time_values), np.asarray(back.time_values)
    )
    assert back.variable_names == source.variable_names
    for name in source.variable_names:
        np.testing.assert_allclose(
            np.asarray(source.get(name).data), np.asarray(back.get(name).data)
        )


def test_round_trip_of_a_real_file_preserves_its_header(tmp_path, sample):
    """A written file keeps the source's identity instead of the writer's defaults.

    The title is the run's own name, the date record is the anchor for the time axis, and
    ``IPOBO`` is what TELEMAC needs to attach boundary conditions -- a file written with all
    three zeroed out is geometry that no longer belongs to the model that produced it.
    """
    source = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")
    out = tmp_path / "round_trip.slf"
    source.to_file(str(out), driver="selafin")

    header = read_header(out)

    assert header["title"] == "Sloped flume Rouse profile test"
    assert header["iparam"][7] == 264
    assert header["iparam"][9] == 1
    assert header["date"] == [1900, 1, 1, 0, 0, 0]
    assert np.count_nonzero(header["ipobo"]) == 264

    back = MeshDataset.read_file(str(out), driver="selafin")
    assert back.global_attributes["title"] == "Sloped flume Rouse profile test"
    assert back.global_attributes["start_date"] == "1900-01-01T00:00:00"
    assert back.global_attributes["n_boundary_nodes"] == 264


def test_the_boundary_numbering_is_not_written_as_a_result_series(tmp_path, sample):
    """``IPOBO`` goes back into its own record, not into the variable list.

    It arrives as a node variable so that it survives a read, which makes it a candidate for
    the data slabs; writing it there would invent a sixth result and shift the record layout.
    """
    source = MeshDataset.read_file(sample("r2d_tidal_flats.slf"), driver="selafin")
    out = tmp_path / "round_trip.slf"
    source.to_file(str(out), driver="selafin")

    names = read_header(out)["names"]

    assert len(names) == 5
    assert names == [
        "VELOCITY U",
        "VELOCITY V",
        "WATER DEPTH",
        "FREE SURFACE",
        "BOTTOM",
    ]


def test_refusing_a_file_is_the_documented_error_type(sample):
    """The 3D refusal is a format error, not a bare ``ValueError`` from the record helpers."""
    driver = SelafinDriver()
    path = sample("r3d_bump.slf")

    with pytest.raises(FileFormatNotSupportedError):
        driver.read(path)


def test_the_shared_header_helper_agrees_with_the_driver(sample):
    """The test helper's record walk matches the driver's on every committed sample.

    The helper is deliberately a separate implementation -- reading a written file back
    through the code that wrote it would make these tests circular -- which means the two
    can drift. This pins them together on real files, including the big-endian ones and the
    one with no date record. Its own doctest is not collected by CI (`--doctest-modules`
    targets `src/`), so this is what verifies the documented behaviour.
    """
    from pyramids_mesh._binary import detect_endianness
    from pyramids_mesh.drivers.coastal_ocean.telemac import _peek_header

    for name in ("r2d_tidal_flats.slf", "qgismesh.slf", "r3d_bump.slf"):
        path = sample(name)
        header = read_header(path)
        iparam, ndp = _peek_header(
            path, detect_endianness(path, expect_first_record=80)
        )

        assert header["ndp"] == ndp, f"{name}: ndp {header['ndp']} vs driver {ndp}"
        assert header["iparam"] == iparam.tolist(), f"{name}: IPARAM disagrees"
        assert len(header["ipobo"]) == header["npoin"], f"{name}: IPOBO length"

    # The date record is read exactly when the flag says it is there: qgismesh has no
    # flag, and reading one anyway would consume the dimensions record.
    assert read_header(sample("qgismesh.slf"))["date"] is None
    assert read_header(sample("r2d_tidal_flats.slf"))["date"] == [1900, 1, 1, 0, 0, 0]
