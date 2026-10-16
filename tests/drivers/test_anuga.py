"""Tests for the ANUGA SWW reader (N1).

SWW is the first NetCDF-schema driver, read through the vendored GDAL multidimensional
API. The tests cover the connectivity-base assertion (0-based, and silent if wrong), the
location-by-dimension-name rule (length is ambiguous in a tiny file), the CRS-only-when-
real mapping, and the lazy read. Fixtures come from ``write_sww`` so one writer of the
NetCDF layout is exercised across every variant.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers import anuga
from pyramids_mesh.drivers.anuga import (
    SwwDriver,
    build_minimal_fixture_sww,
    write_sww,
)

_NODE_X = [0.0, 1.0, 1.0, 0.0]
_NODE_Y = [0.0, 0.0, 1.0, 1.0]
_VOLUMES = [[0, 1, 2], [0, 2, 3]]
_ELEVATION = [2.0, 2.5, 3.0, 3.5]
_TIMES = [0.0, 3600.0]
_STAGE = [[0.5, 0.25, 0.75, 0.125], [1.5, 1.25, 1.75, 1.125]]


def _write(tmp_path, name="model.sww", **overrides):
    """Write a SWW fixture, overriding any ``write_sww`` keyword."""
    kwargs = dict(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        volumes=_VOLUMES,
        elevation=_ELEVATION,
        times=_TIMES,
        node_variables={"stage": _STAGE},
    )
    kwargs.update(overrides)
    path = str(tmp_path / name)
    write_sww(path, **kwargs)
    return path


def test_sww_reads_topology_and_node_count(tmp_path):
    """A SWW reads into its node count and triangle connectivity."""
    dataset = SwwDriver().read(_write(tmp_path))

    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2
    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_sww_connectivity_is_zero_based(tmp_path):
    """ANUGA's 0-based volumes stay 0-based, with the base recorded as 0 for a writer."""
    dataset = SwwDriver().read(_write(tmp_path))

    connectivity = dataset.mesh.face_node_connectivity
    assert int(connectivity.data.min()) == 0
    assert connectivity.original_start_index == 0


def test_sww_one_based_volumes_are_rebased(tmp_path):
    """A 1-based volumes table is rebased to 0, and the base recorded as 1."""
    one_based = [[1, 2, 3], [1, 3, 4]]
    dataset = SwwDriver().read(_write(tmp_path, volumes=one_based))

    connectivity = dataset.mesh.face_node_connectivity
    assert connectivity.data.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert connectivity.original_start_index == 1


def test_sww_out_of_range_volumes_index_is_refused(tmp_path):
    """A volumes entry beyond the node range is refused rather than read wrong."""
    bad = [[2, 3, 4], [2, 4, 5]]
    with pytest.raises(FileFormatNotSupportedError, match="no node index reaches"):
        SwwDriver().read(_write(tmp_path, volumes=bad))


def test_sww_zero_based_mesh_not_referencing_node_zero(tmp_path):
    """A 0-based mesh whose node 0 is unreferenced is not misdetected as 1-based.

    Keying the base on volumes.min() shifted such a mesh down a node; keying on the max
    (which stays within n_node-1 here) keeps it 0-based. Five nodes, triangles using
    only nodes 1..4, so min()==1 but it is genuinely 0-based.
    """
    volumes = [[1, 2, 3], [1, 3, 4]]
    dataset = SwwDriver().read(
        _write(
            tmp_path,
            node_x=[9.0, 0.0, 1.0, 1.0, 0.0],
            node_y=[9.0, 0.0, 0.0, 1.0, 1.0],
            volumes=volumes,
            elevation=[0.0, 2.0, 2.5, 3.0, 3.5],
            times=[0.0],
            node_variables={"stage": [[0.0, 0.5, 0.25, 0.75, 0.125]]},
        )
    )

    connectivity = dataset.mesh.face_node_connectivity
    assert connectivity.original_start_index == 0
    assert connectivity.data.tolist() == [[1, 2, 3], [1, 3, 4]]


def test_sww_elevation_becomes_node_variable(tmp_path):
    """Elevation comes back as a node-located ``node_z`` variable, not a coordinate."""
    dataset = SwwDriver().read(_write(tmp_path))

    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (4,)
    assert node_z.data.tolist() == _ELEVATION


def test_sww_elevation_read_from_z_name(tmp_path):
    """Elevation stored under ``z`` rather than ``elevation`` is still node_z."""
    dataset = SwwDriver().read(_write(tmp_path, elevation_name="z"))

    assert "node_z" in dataset.data_variable_names
    assert "z" not in dataset.data_variable_names


def test_sww_variable_location_resolved_by_dimension_name(tmp_path):
    """A face-dimensioned variable is face-located even when counts could confuse it.

    The fixture has four nodes and two faces, so lengths differ here; the point is that
    the location follows the dimension name, which a face variable makes explicit.
    """
    dataset = SwwDriver().read(
        _write(tmp_path, face_variables={"friction": [[0.1, 0.2], [0.3, 0.4]]})
    )

    friction = dataset.get("friction")
    assert friction.location == "face"
    assert friction.shape == (2, 2)


def test_sww_variable_on_foreign_dimension_is_skipped_with_warning(tmp_path):
    """A variable on neither the node nor the face dimension is skipped, with a warning.

    The foreign array is written through ``write_sww`` on its own fresh dimension, so
    the file comes from the one proven creation path and the mesh model has an array it
    genuinely cannot place.
    """
    path = _write(
        tmp_path, foreign_variables={"boundary_flux": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )

    with pytest.warns(UserWarning, match="neither the node nor the face dimension"):
        result = SwwDriver().read(path)

    assert "boundary_flux" not in result.data_variable_names


def test_sww_stage_keeps_its_time_axis(tmp_path):
    """A time-varying node variable carries its time coordinate and 2D shape."""
    dataset = SwwDriver().read(_write(tmp_path))

    stage = dataset.get("stage")
    assert stage.location == "node"
    assert stage.shape == (2, 4)
    assert stage.attributes["time_values"].tolist() == _TIMES


def test_sww_without_crs_leaves_crs_none(tmp_path):
    """With no CRS attribute, the dataset carries no CRS -- no UTM zone is guessed."""
    dataset = SwwDriver().read(_write(tmp_path))

    assert dataset.crs_wkt is None
    assert dataset.mesh.crs is None


def test_sww_valid_crs_attribute_is_carried(tmp_path):
    """A valid WKT CRS attribute is validated and carried."""
    from osgeo import osr

    reference = osr.SpatialReference()
    reference.ImportFromEPSG(32756)
    dataset = SwwDriver().read(_write(tmp_path, crs_wkt=reference.ExportToWkt()))

    assert dataset.crs_wkt is not None
    assert "UTM" in dataset.crs_wkt


def test_sww_garbage_crs_attribute_leaves_crs_none(tmp_path):
    """A CRS attribute that does not validate is left as None, not carried blindly."""
    dataset = SwwDriver().read(_write(tmp_path, crs_wkt='PROJCS["not real at all"]'))

    assert dataset.crs_wkt is None


def test_sww_faces_are_ccw(tmp_path):
    """The two triangles come back counter-clockwise, as the mesh model requires."""
    dataset = SwwDriver().read(_write(tmp_path))
    mesh = dataset.mesh

    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_sww_lazy_read_defers_values(tmp_path):
    """A lazy read reports a source for every variable without loading its values."""
    dataset = SwwDriver().read(_write(tmp_path), lazy=True)

    for name in dataset.data_variable_names:
        variable = dataset.get(name)
        assert variable.has_data_source, name
        assert variable._data is None, name

    assert dataset.get("stage").data[0].tolist() == [0.5, 0.25, 0.75, 0.125]


def test_sww_eager_read_holds_values(tmp_path):
    """A non-lazy read carries the values directly rather than behind a loader."""
    dataset = SwwDriver().read(_write(tmp_path), lazy=False)

    assert dataset.get("stage")._data is not None


def test_sww_static_node_variable_without_time(tmp_path):
    """A SWW with no time dimension reads its node variables as static."""
    dataset = SwwDriver().read(
        _write(tmp_path, times=None, node_variables={"friction": [0.1, 0.2, 0.3, 0.4]})
    )

    friction = dataset.get("friction")
    assert friction.location == "node"
    assert friction.shape == (4,)
    assert "time_values" not in friction.attributes


def test_sww_can_open_false_on_junk(tmp_path):
    """``can_open`` declines empty, foreign and directory paths without raising."""
    driver = SwwDriver()
    empty = tmp_path / "empty.sww"
    empty.write_bytes(b"")
    foreign = tmp_path / "other.sww"
    foreign.write_text("plainly not netcdf\n", encoding="utf-8")
    directory = tmp_path / "folder"
    directory.mkdir()

    assert driver.can_open(str(empty)) is False
    assert driver.can_open(str(foreign)) is False
    assert driver.can_open(str(directory)) is False


def test_sww_can_open_false_on_netcdf_without_volumes(tmp_path):
    """A NetCDF file lacking ``volumes`` is not a SWW and is declined."""
    from osgeo import gdal

    path = str(tmp_path / "plain.sww")
    dataset = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    group = dataset.GetRootGroup()
    dim = group.CreateDimension("x", "", "", 3)
    array = group.CreateMDArray(
        "x", [dim], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    )
    array.Write(np.array([1.0, 2.0, 3.0]))
    dataset.FlushCache()
    dataset.Close()

    assert SwwDriver().can_open(path) is False


def test_sww_committed_fixture_reads_and_matches_the_builder(tmp_path):
    """The committed fixture reads, and agrees with ``build_minimal_fixture_sww``.

    The comparison is semantic, not byte-for-byte: a NetCDF-4 file is HDF5, whose byte
    layout is not guaranteed stable across GDAL builds, so the two are compared by what
    they read to rather than by their bytes.
    """
    committed = Path(__file__).parent.parent / "data" / "minimal.sww"
    from_file = SwwDriver().read(str(committed))

    assert from_file.mesh.n_node == 4
    assert from_file.mesh.n_face == 2

    generated = tmp_path / "generated.sww"
    build_minimal_fixture_sww(str(generated))
    from_builder = SwwDriver().read(str(generated))

    assert (
        from_file.mesh.face_node_connectivity.data.tolist()
        == from_builder.mesh.face_node_connectivity.data.tolist()
    )
    assert sorted(from_file.data_variable_names) == sorted(
        from_builder.data_variable_names
    )
    assert (
        from_file.get("stage").data.tolist() == from_builder.get("stage").data.tolist()
    )


def _write_no_volumes(tmp_path):
    """A NetCDF with x, y and a 2D array but no ``volumes`` -- opens, yet is not a SWW.

    The 2D array is what makes GDAL open it as a multidimensional raster; a purely 1-D
    NetCDF does not open that way.
    """
    from osgeo import gdal

    path = str(tmp_path / "novolumes.sww")
    dataset = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    group = dataset.GetRootGroup()
    f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    points = group.CreateDimension("number_of_points", "", "", 4)
    rows = group.CreateDimension("rows", "", "", 2)
    cols = group.CreateDimension("cols", "", "", 2)
    for name in ("x", "y"):
        array = group.CreateMDArray(name, [points], f64)
        array.Write(np.array([0.0, 1.0, 1.0, 0.0]))
    cells = group.CreateMDArray("cells", [rows, cols], f64)
    cells.Write(np.zeros((2, 2)))
    dataset.FlushCache()
    dataset.Close()
    return path


def test_sww_sniff_false_on_netcdf_that_opens_without_volumes(tmp_path):
    """``can_open`` reaches the array-name check and declines a non-SWW NetCDF."""
    assert SwwDriver().can_open(_write_no_volumes(tmp_path)) is False


def test_sww_parse_without_volumes_is_refused(tmp_path):
    """``read`` on a NetCDF lacking ``volumes`` names the missing array."""
    with pytest.raises(FileFormatNotSupportedError, match="missing"):
        SwwDriver().read(_write_no_volumes(tmp_path))


def test_sww_volumes_index_out_of_range_is_refused(tmp_path):
    """A 0-based volumes entry beyond the node range is refused."""
    bad = [[0, 1, 9], [0, 2, 3]]
    with pytest.raises(FileFormatNotSupportedError, match="no node index reaches"):
        SwwDriver().read(_write(tmp_path, volumes=bad))


def test_sww_empty_crs_attribute_leaves_crs_none(tmp_path):
    """A present but empty CRS attribute is ignored rather than carried."""
    dataset = SwwDriver().read(_write(tmp_path, crs_wkt=""))
    assert dataset.crs_wkt is None


def test_sww_without_elevation_has_no_node_z(tmp_path):
    """A SWW carrying no elevation array produces no ``node_z`` variable."""
    dataset = SwwDriver().read(_write(tmp_path, elevation=None))
    assert "node_z" not in dataset.data_variable_names
    assert "stage" in dataset.data_variable_names


def test_sww_elevation_on_foreign_dimension_is_skipped_not_crashed(tmp_path):
    """Elevation on a dimension the mesh cannot place is skipped, not a None crash.

    The node_z assignment is guarded like the result-variable loop; before the guard an
    unplaceable elevation set ``variables['node_z'] = None`` and then crashed in the base
    builder dereferencing ``None.attributes``.
    """
    path = _write(
        tmp_path,
        elevation=None,
        foreign_variables={"elevation": [1.0, 2.0, 3.0, 4.0, 5.0]},
    )
    with pytest.warns(UserWarning, match="neither the node nor the face dimension"):
        dataset = SwwDriver().read(path)

    assert "node_z" not in dataset.data_variable_names


_IMPORTABLE_INVALID_WKT = (
    'PROJCS["x",GEOGCS["y",DATUM["z",SPHEROID["s",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["deg",0.0174]],PROJECTION["Mercator_1SP"],UNIT["m",1]]'
)


def test_sww_importable_but_invalid_crs_falls_back_to_none(tmp_path):
    """A WKT that imports but fails PROJ validation yields crs=None, not a crash.

    Under the scoped ExceptionMgr, Validate() raises for such a WKT; the guard must catch
    it and fall back rather than letting a RuntimeError escape read().
    """
    dataset = SwwDriver().read(_write(tmp_path, crs_wkt=_IMPORTABLE_INVALID_WKT))

    assert dataset.crs_wkt is None


def test_sww_both_elevation_and_z_warns_and_keeps_elevation(tmp_path):
    """A file carrying both ``elevation`` and ``z`` warns and keeps only ``elevation``.

    Both are elevation names, so only the first becomes node_z; the other is dropped,
    which is now announced rather than silent.
    """
    path = _write(
        tmp_path,
        elevation=[2.0, 2.5, 3.0, 3.5],
        extra_node_arrays={"z": [1.0, 1.5, 2.0, 2.5]},
    )
    with pytest.warns(UserWarning, match="only 'elevation' becomes node_z"):
        result = SwwDriver().read(path)

    assert "node_z" in result.data_variable_names
    assert "z" not in result.data_variable_names


def test_sww_negative_volumes_index_is_refused(tmp_path):
    """A negative volumes index is refused outright."""
    with pytest.raises(FileFormatNotSupportedError, match="negative volumes index"):
        SwwDriver().read(_write(tmp_path, volumes=[[-1, 0, 1], [0, 1, 2]]))


def test_sww_volumes_mixing_zero_and_node_count_is_refused(tmp_path):
    """A table using both index 0 and index n_node is refused -- base is inconsistent."""
    # 4 nodes, indices span 0..4: 0 is only 0-based, 4 is only 1-based.
    with pytest.raises(FileFormatNotSupportedError, match="neither 0 nor 1"):
        SwwDriver().read(_write(tmp_path, volumes=[[0, 1, 2], [2, 3, 4]]))
