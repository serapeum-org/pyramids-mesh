"""Tests for the Esri TIN reader (D5), against real committed sample directories.

An Esri TIN is a directory of big-endian ``.adf`` files. Two genuine TINs are committed
under ``tests/data/esri_tin/``: ``dem`` (281 nodes, 528 surface triangles after masking)
and ``dem_with_holes`` (527 nodes, 773 triangles). The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset
from pyramids_mesh.drivers.esri import EsriTinDriver

_DATA = Path(__file__).parent.parent / "data" / "esri_tin"
_DEM = _DATA / "dem"
_HOLES = _DATA / "dem_with_holes"


def _read(directory):
    """Read a committed TIN directory, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EsriTinDriver().read(str(directory))


def test_reads_nodes_and_masked_triangles():
    """dem reads 281 nodes and the 528 unmasked surface triangles."""
    mesh = _read(_DEM).mesh
    assert mesh.n_node == 281
    assert mesh.n_face == 528
    assert mesh.face_node_connectivity.data.shape == (528, 3)


def test_mask_drops_hull_frame_triangles():
    """dem_with_holes keeps 773 of its 1048 triangles; the frame triangles are dropped."""
    mesh = _read(_HOLES).mesh
    assert mesh.n_node == 527
    assert mesh.n_face == 773


def test_triangle_indices_are_one_based_resolved_to_zero_based():
    """The 1-based tnod table is resolved to 0-based, and the base recorded."""
    connectivity = _read(_DEM).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert int(connectivity.data.min()) >= 0
    assert int(connectivity.data.max()) == 280


def test_node_elevation_becomes_node_z():
    """tnz.adf becomes a node-located node_z variable."""
    node_z = _read(_DEM).get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (281,)


def test_projection_is_read_from_esri_wkt():
    """The Esri-dialect WKT in prj.adf resolves to EPSG:4326."""
    assert _read(_DEM).mesh.crs == 4326


def test_faces_are_counter_clockwise():
    """Esri triangles are clockwise; they come back counter-clockwise after validation."""
    mesh = _read(_DEM).mesh
    for row in mesh.face_node_connectivity.data:
        xs = mesh.node_x[row]
        ys = mesh.node_y[row]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_reached_through_read_file_via_adf_component():
    """read_file dispatches to this driver when pointed at any .adf component."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = UgridDataset.read_file(str(_DEM / "tdenv9.adf"))
    assert dataset.mesh.n_face == 528


def test_reached_through_read_file_with_driver_on_directory():
    """read_file dispatches to this driver for a directory when driver= is given."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = UgridDataset.read_file(str(_DEM), driver="esri_tin")
    assert dataset.mesh.n_face == 528


def test_can_open_accepts_directory_and_component_files():
    """can_open accepts the TIN directory and any .adf file within it."""
    driver = EsriTinDriver()
    assert driver.can_open(str(_DEM)) is True
    assert driver.can_open(str(_DEM / "tnxy.adf")) is True
    assert driver.can_open(str(_DEM / "prj.adf")) is True


def test_can_open_declines_non_tin(tmp_path):
    """can_open declines an .adf whose directory is not a TIN, and a plain file/dir."""
    driver = EsriTinDriver()
    # a lone .adf with no TIN siblings (an Esri GRID-style file, say)
    stray = tmp_path / "w001001.adf"
    stray.write_bytes(b"not a tin component")
    assert driver.can_open(str(stray)) is False
    # an empty directory
    assert driver.can_open(str(tmp_path)) is False
    # a non-.adf regular file
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    assert driver.can_open(str(other)) is False


def test_missing_component_is_a_format_error(tmp_path):
    """A directory missing a required component is a format error."""
    partial = tmp_path / "broken"
    partial.mkdir()
    (partial / "tnxy.adf").write_bytes(b"\x00" * 16)
    (partial / "tdenv9.adf").write_bytes(b"\x00" * 24)
    # no tnod.adf / tnz.adf
    with pytest.raises(FileFormatNotSupportedError):
        EsriTinDriver().read(str(partial))


def test_read_of_a_non_tin_path_is_a_format_error(tmp_path):
    """read() on a path that is neither a TIN directory nor an .adf file is refused."""
    other = tmp_path / "notes.txt"
    other.write_text("not a tin")
    with pytest.raises(FileFormatNotSupportedError, match="not an Esri TIN"):
        EsriTinDriver().read(str(other))


def _tin_without(tmp_path, *, omit=(), prj=None):
    """Build a TIN directory from dem's real components, omitting some / setting prj."""
    built = tmp_path / "tin"
    built.mkdir()
    for component in (
        "tnxy.adf",
        "tnz.adf",
        "tnod.adf",
        "tdenv9.adf",
        "tmsx.adf",
        "tmsk.adf",
        "prj.adf",
    ):
        if component in omit:
            continue
        if component == "prj.adf" and prj is not None:
            (built / component).write_bytes(prj)
            continue
        (built / component).write_bytes((_DEM / component).read_bytes())
    return built


def test_mask_absent_keeps_every_triangle(tmp_path):
    """With no tmsk/tmsx, every triangle is kept -- dem's full 556, not the masked 528."""
    built = _tin_without(tmp_path, omit=("tmsk.adf", "tmsx.adf"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = EsriTinDriver().read(str(built)).mesh
    assert mesh.n_face == 556


@pytest.mark.parametrize(
    "prj",
    [
        None,  # prj.adf absent entirely
        b"",  # empty prj.adf
        b"not valid wkt at all\n",  # unparseable
    ],
)
def test_crs_is_none_without_a_usable_projection(tmp_path, prj):
    """A missing, empty or unparseable prj.adf leaves crs None rather than guessing."""
    omit = ("prj.adf",) if prj is None else ()
    built = _tin_without(tmp_path, omit=omit, prj=prj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert EsriTinDriver().read(str(built)).mesh.crs is None


def test_projection_without_authority_returns_wkt(tmp_path):
    """A valid WKT that names no authority comes back as the normalised WKT string."""
    # a real geographic WKT with its AUTHORITY stripped -> parses, but no EPSG code
    wkt = (
        b'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
        b'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
        b'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    )
    built = _tin_without(tmp_path, prj=wkt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crs = EsriTinDriver().read(str(built)).mesh.crs
    assert isinstance(crs, str) and "GEOGCS" in crs
