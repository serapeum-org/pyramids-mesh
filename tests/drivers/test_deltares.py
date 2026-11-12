"""Tests for the Deltares 1D network reader (C1), against real committed samples.

Delft3D FM / D-HYDRO writes 1D hydrodynamic models as UGRID-1.0 + Deltares-extension
NetCDF: a ``network`` of branches and a ``mesh1d`` discretisation of computational nodes,
with ``mesh1d_*`` results over time on a map file. The driver builds a ``Mesh1d`` from
the ``mesh1d`` discretisation, finding both topologies by their ``cf_role`` attributes.

Several genuine samples are committed under ``tests/data/deltares1d/``: a pure-1D topology
(``korte-woerden``), a tiny combined 1D2D topology (``UGrid_1D2D``), and a pure-1D map
file with 97 result steps (``Haskoning``). Three more exercise the branch+offset node
geometry that most D-HYDRO files use instead of explicit x/y: ``dsd-demo`` (reconstructed
from the branch geometry), ``magdalena`` (explicit ``mesh1d_node_x`` present but not
referenced by the topology), and ``f101-boundary`` (the alternate ``branch_id`` /
``branch_offset`` spelling, reconstructed). The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import Mesh1d

from pyramids_mesh.drivers.deltares import (
    DeltaresNetworkDriver,
    _as_str,
    _attr_map,
    _branch_offset_names,
    _DeltaresReader,
    _fill_value,
    _interpolate_along_branches,
    _network_branch_geometry,
    _split,
    _string_attr,
    _topology_dimension,
)

_DATA = Path(__file__).parent.parent / "data" / "deltares1d"
_KORTE = _DATA / "korte-woerden-1d_FlowFM_net.nc"
_UGRID_1D2D = _DATA / "UGrid_1D2D_net.nc"
_HASKONING = _DATA / "Haskoning_example_map.nc"
_DSD = _DATA / "branchoffset_1d_minimal_no-hidden-xy__dsd-demo.nc"
_MAGDALENA = _DATA / "branchoffset_1d_hidden-xy__magdalena.nc"
_F101 = _DATA / "branchoffset-altnames_1d__f101-boundary.nc"
_THREEDI = Path(__file__).parent.parent / "data" / "3di" / "results_3di.nc"


def _read(path):
    """Read a Deltares 1D file, ignoring the expected network-drop warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DeltaresNetworkDriver().read(str(path))


def _parse(path, *, lazy=True):
    """Parse a Deltares 1D file to a validated ParsedMesh, without constructing."""
    from pyramids_mesh._validate import validate_and_normalise

    driver = DeltaresNetworkDriver()
    return validate_and_normalise(driver._parse(str(path), lazy=lazy))


def _reconstructed_reference(path):
    """Independently interpolate node x/y from a file's raw branch, offset and geometry.

    Reproduces the reconstruction from the committed bytes with no reader code beyond the
    pure interpolation helper, so a test can confirm the driver's own result matches.
    """
    dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    try:
        group = dataset.GetRootGroup()
        branch = np.asarray(group.OpenMDArray("mesh1d_node_branch").ReadAsArray())
        branch = branch.astype(np.int64) - 1  # the sample is 1-based
        offset = np.asarray(
            group.OpenMDArray("mesh1d_node_offset").ReadAsArray(), np.float64
        )
        geom_x = np.asarray(
            group.OpenMDArray("network1d_geom_x").ReadAsArray(), np.float64
        )
        geom_y = np.asarray(
            group.OpenMDArray("network1d_geom_y").ReadAsArray(), np.float64
        )
        counts = np.asarray(
            group.OpenMDArray("network1d_geom_node_count").ReadAsArray()
        ).astype(np.int64)
    finally:
        dataset.Close()
    return _interpolate_along_branches(branch, offset, geom_x, geom_y, counts)


def _sibling_node_xy(path):
    """Read a file's explicit ``mesh1d_node_x`` / ``mesh1d_node_y`` arrays."""
    dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    try:
        group = dataset.GetRootGroup()
        node_x = np.asarray(
            group.OpenMDArray("mesh1d_node_x").ReadAsArray(), np.float64
        )
        node_y = np.asarray(
            group.OpenMDArray("mesh1d_node_y").ReadAsArray(), np.float64
        )
    finally:
        dataset.Close()
    return node_x, node_y


def test_pure_1d_network_reads_node_and_edge_counts():
    """The korte-woerden mesh1d discretisation reads as an 86-node, 91-edge network."""
    network = _read(_KORTE)
    assert isinstance(network, Mesh1d)
    assert network.n_node == 86
    assert network.n_edge == 91
    data = network.edge_node_connectivity.data
    assert data.shape == (91, 2)
    assert int(data.min()) == 0
    assert int(data.max()) == 85


def test_both_sample_topologies_parse_to_a_network():
    """Both the pure-1D and the combined-1D2D samples yield a mesh1d Mesh1d.

    The two topologies are found by cf_role, so a file is read the same regardless of
    how it spells its topologies; the combined 1D2D file's 1D half reads via a direct
    ``read`` even though ``can_open`` leaves it to the UGRID reader.
    """
    korte = _read(_KORTE)
    combined = _read(_UGRID_1D2D)
    assert isinstance(korte, Mesh1d)
    assert isinstance(combined, Mesh1d)
    assert (combined.n_node, combined.n_edge) == (4, 3)


def test_crs_comes_from_the_declared_epsg():
    """The network carries the file's EPSG code (Amersfoort / RD New, 28992)."""
    assert _read(_KORTE).crs == 28992


def test_one_based_edges_are_rebased_and_the_base_recorded():
    """The Haskoning map file is 1-based; edges re-base to 0 and the base is kept."""
    network = _read(_HASKONING)
    assert network.n_node == 344
    assert network.n_edge == 342
    data = network.edge_node_connectivity.data
    assert int(data.min()) == 0
    assert int(data.max()) == 343
    assert network.edge_node_connectivity.original_start_index == 1


def test_branch_and_offset_are_carried_as_node_variables():
    """The per-node branch index and offset are parsed as node-located variables."""
    parsed = _parse(_KORTE)
    branch = parsed.variables["mesh1d_node_branch"]
    offset = parsed.variables["mesh1d_node_offset"]
    assert branch.location == "node"
    assert offset.location == "node"
    assert branch.shape == (86,)
    assert offset.shape == (86,)
    # they are static node data, carried eagerly rather than deferred
    assert branch.values is not None


def test_branch_offset_only_file_is_read_by_reconstruction():
    """A file whose nodes are branch+offset (no x/y) reconstructs node coordinates.

    ``dsd-demo`` stores no ``mesh1d_node_x`` anywhere; the node x/y is rebuilt by linear
    referencing each node's offset along its branch polyline. The reconstructed nodes land
    inside the network's geometry bounds and are all finite.
    """
    network = _read(_DSD)
    assert isinstance(network, Mesh1d)
    assert (network.n_node, network.n_edge) == (12, 13)
    assert np.isfinite(network.node_x).all()
    assert np.isfinite(network.node_y).all()
    # the reconstruction matches an independent interpolation from the same raw arrays
    expected_x, expected_y = _reconstructed_reference(_DSD)
    assert np.allclose(network.node_x, expected_x)
    assert np.allclose(network.node_y, expected_y)


def test_unreferenced_node_xy_arrays_are_preferred_over_reconstruction():
    """Explicit ``mesh1d_node_x`` present but unlisted is used verbatim, not reconstructed.

    ``magdalena`` lists only branch+offset in ``node_coordinates`` yet carries genuine
    ``mesh1d_node_x`` / ``_node_y`` arrays; those authoritative values are returned
    exactly, in preference to interpolating.
    """
    network = _read(_MAGDALENA)
    assert network.n_node == 447
    stored_x, stored_y = _sibling_node_xy(_MAGDALENA)
    assert np.array_equal(network.node_x, stored_x)
    assert np.array_equal(network.node_y, stored_y)


def test_alternate_branch_offset_spelling_is_reconstructed():
    """The ``branch_id`` / ``branch_offset`` spelling reconstructs the same way.

    ``f101-boundary`` names its node geometry ``mesh1d_node_branch_id`` /
    ``mesh1d_node_branch_offset``; the reader tells branch from offset by dtype, so the
    spelling does not matter and the nodes reconstruct to finite coordinates.
    """
    network = _read(_F101)
    assert network.n_node == 164
    assert np.isfinite(network.node_x).all()
    assert np.isfinite(network.node_y).all()
    # the alternate-spelling arrays are still carried as node variables
    parsed = _parse(_F101)
    assert parsed.variables["mesh1d_node_branch_id"].location == "node"
    assert parsed.variables["mesh1d_node_branch_offset"].location == "node"


def test_result_variables_attach_over_time():
    """The mesh1d_* results are node/edge variables over the 97-step time axis."""
    parsed = _parse(_HASKONING)
    s1 = parsed.variables["mesh1d_s1"]
    q1 = parsed.variables["mesh1d_q1"]
    assert s1.location == "node"
    assert s1.shape == (97, 344)
    assert s1.standard_name == "sea_surface_height"
    assert s1.time_values is not None
    assert s1.time_values.shape == (97,)
    assert q1.location == "edge"
    assert q1.shape == (97, 342)
    assert q1.standard_name == "discharge"
    # the four results the sample is documented to carry are all present
    for name in ("mesh1d_s1", "mesh1d_waterdepth", "mesh1d_u1", "mesh1d_q1"):
        assert name in parsed.variables


def test_result_values_are_lazy():
    """A lazy parse leaves each result behind a loader, read only on demand."""
    parsed = _parse(_HASKONING, lazy=True)
    s1 = parsed.variables["mesh1d_s1"]
    assert s1.values is None
    assert s1.loader is not None
    loaded = s1.loader()
    assert loaded.shape == (97, 344)


def test_can_open_claims_a_1d_file_and_declines_2d_and_junk(tmp_path):
    """can_open claims a pure-1D file but declines 2D-bearing, other-model and junk."""
    driver = DeltaresNetworkDriver()
    # pure-1D topology and a pure-1D map file are claimed
    assert driver.can_open(str(_KORTE)) is True
    assert driver.can_open(str(_HASKONING)) is True
    # a file that also carries a 2D face_node_connectivity is left to the UGRID reader
    assert driver.can_open(str(_UGRID_1D2D)) is False
    # a 3Di result .nc has no mesh_topology at all, so it is declined
    assert driver.can_open(str(_THREEDI)) is False
    # non-NetCDF junk and a directory are declined without raising
    junk = tmp_path / "plain.nc"
    junk.write_bytes(b"\x00\x01 not netcdf \xff")
    assert driver.can_open(str(junk)) is False
    directory = tmp_path / "folder.nc"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


class _Attr:
    """A stand-in GDAL attribute returning a fixed value, or raising on read."""

    def __init__(self, name, value=None, raises=False):
        self._name = name
        self._value = value
        self._raises = raises

    def GetName(self):
        """Return the attribute name."""
        return self._name

    def Read(self):
        """Return the stored value, or raise ``RuntimeError`` when so configured."""
        if self._raises:
            raise RuntimeError("unreadable attribute")
        return self._value


class _Dim:
    """A stand-in GDAL dimension reporting a name and size."""

    def __init__(self, name, size=0):
        self._name = name
        self._size = size

    def GetName(self):
        """Return the dimension name."""
        return self._name

    def GetSize(self):
        """Return the dimension size."""
        return self._size


class _Array:
    """A stand-in GDAL MD-array with attributes, dimensions and a read value."""

    def __init__(self, attrs=None, value=None, dims=None):
        self._attrs = attrs or []
        self._value = value
        self._dims = dims or []

    def GetAttributes(self):
        """Return the array's attribute handles."""
        return list(self._attrs)

    def ReadAsArray(self):
        """Return the array's stored value."""
        return self._value

    def GetDimensions(self):
        """Return the array's dimension handles."""
        return list(self._dims)


class _Group:
    """A stand-in GDAL group exposing named arrays, ignoring list options."""

    def __init__(self, arrays=None):
        self._arrays = arrays or {}

    def GetMDArrayNames(self, options=None):
        """Return the names of the group's arrays."""
        return list(self._arrays)

    def OpenMDArray(self, name):
        """Return the named array."""
        return self._arrays[name]


def _topo_array(dimension, *, coordinate_space=False):
    """Build a fake ``mesh_topology`` dummy array of the given topology dimension."""
    attrs = [_Attr("cf_role", "mesh_topology"), _Attr("topology_dimension", dimension)]
    if coordinate_space:
        attrs.append(_Attr("coordinate_space", "network"))
    return _Array(attrs=attrs)


def _reader():
    """A reader bound to a dummy path, for exercising its white-box helpers."""
    return _DeltaresReader("dummy.nc", lazy=True)


def test_string_attr_returns_none_when_read_raises():
    """``_string_attr`` returns ``None`` when the attribute read raises ``RuntimeError``."""
    array = _Array(attrs=[_Attr("standard_name", raises=True)])
    assert _string_attr(array, "standard_name") is None


def test_attr_map_skips_an_unreadable_attribute():
    """``_attr_map`` drops an attribute whose read raises and keeps the rest."""
    array = _Array(attrs=[_Attr("good", 1), _Attr("bad", raises=True)])
    assert _attr_map(array) == {"good": 1}


def test_fill_value_reads_a_declared_fill_value():
    """``_fill_value`` returns a declared ``_FillValue`` as a float."""
    array = _Array(attrs=[_Attr("_FillValue", -999.0)])
    assert _fill_value(array) == -999.0


def test_fill_value_none_when_the_sentinel_is_unparseable():
    """A ``_FillValue`` that is not numeric falls through to ``None``."""
    array = _Array(attrs=[_Attr("_FillValue", "not-a-number")])
    assert _fill_value(array) is None


def test_topology_dimension_defaults_to_zero():
    """``_topology_dimension`` is ``0`` for a missing or unparseable value."""
    assert _topology_dimension({}) == 0
    assert _topology_dimension({"topology_dimension": "x"}) == 0


def test_as_str_handles_list_values():
    """``_as_str`` takes the first element of a list and ``None`` for an empty list."""
    assert _as_str(["abc", "def"]) == "abc"
    assert _as_str([]) is None


def test_split_handles_none_and_lists():
    """``_split`` returns ``[]`` for ``None`` and flattens a list of name strings."""
    assert _split(None) == []
    assert _split(["a b", "c"]) == ["a", "b", "c"]


def test_start_index_defaults_to_zero():
    """``_start_index`` is ``0`` for a missing or unparseable ``start_index``."""
    assert _DeltaresReader._start_index(_Array(attrs=[])) == 0
    bad = _Array(attrs=[_Attr("start_index", "x")])
    assert _DeltaresReader._start_index(bad) == 0


def test_discretisation_raises_without_a_1d_topology():
    """``_discretisation`` raises when the file carries no 1D ``mesh_topology``."""
    group = _Group({"mesh2d": _topo_array(2)})
    with pytest.raises(FileFormatNotSupportedError, match="no 1D mesh_topology"):
        _reader()._discretisation(group)


def test_discretisation_falls_back_without_a_coordinate_space():
    """A lone 1D topology with no ``coordinate_space`` is used as the discretisation."""
    group = _Group({"mesh1d": _topo_array(1)})
    name, attrs = _reader()._discretisation(group)
    assert name == "mesh1d"
    assert _as_str(attrs["cf_role"]) == "mesh_topology"


def test_node_coordinates_raise_without_x_or_y():
    """``_node_coordinates`` raises when no entry carries the x/y standard name."""
    group = _Group(
        {
            "a": _Array(attrs=[_Attr("standard_name", "foo")]),
            "b": _Array(attrs=[_Attr("standard_name", "bar")]),
        }
    )
    topo = {"node_coordinates": "a b"}
    with pytest.raises(FileFormatNotSupportedError, match="no x/y node coordinates"):
        _reader()._node_coordinates(group, "mesh1d", topo)


def test_interpolate_along_branches_linear_references_offsets():
    """A node's x/y is the point at its offset along its branch's polyline.

    One L-shaped branch of two segments (0->10 east, then 10->20 north) places a node at
    offset 5 halfway along the first segment and a node at offset 15 halfway up the
    second; offsets are measured as cumulative chord length from the branch start.
    """
    geom_x = np.array([0.0, 10.0, 10.0])
    geom_y = np.array([0.0, 0.0, 10.0])
    counts = np.array([3])
    branch = np.array([0, 0, 0])
    offset = np.array([0.0, 5.0, 15.0])
    node_x, node_y = _interpolate_along_branches(branch, offset, geom_x, geom_y, counts)
    assert np.allclose(node_x, [0.0, 5.0, 10.0])
    assert np.allclose(node_y, [0.0, 0.0, 5.0])


def test_interpolate_along_branches_returns_none_for_a_branchless_node():
    """A node on a branch that has no geometry yields ``None`` rather than ``nan``."""
    geom_x = np.array([0.0, 1.0])
    geom_y = np.array([0.0, 0.0])
    counts = np.array([2, 0])  # branch 1 has no vertices
    branch = np.array([0, 1])  # the second node sits on the empty branch
    offset = np.array([0.0, 0.0])
    assert _interpolate_along_branches(branch, offset, geom_x, geom_y, counts) is None


def test_branch_offset_names_tells_branch_from_offset_by_dtype():
    """The integer entry is the branch index; the float entry is the offset."""
    group = _Group(
        {
            "mesh1d_node_branch": _Array(value=np.array([0, 1, 2], np.int32)),
            "mesh1d_node_offset": _Array(value=np.array([0.0, 1.5], np.float64)),
        }
    )
    branch, offset = _branch_offset_names(
        group, ["mesh1d_node_offset", "mesh1d_node_branch"]
    )
    assert branch == "mesh1d_node_branch"
    assert offset == "mesh1d_node_offset"


def test_reconstruct_returns_none_without_a_coordinate_space():
    """Reconstruction declines when the topology names no coordinate-space network."""
    group = _Group({})
    result = _reader()._reconstruct_node_coordinates(group, {}, ["a", "b"])
    assert result is None


def test_branch_offset_names_ignores_a_non_numeric_entry():
    """A ``node_coordinates`` entry that is neither integer nor float is skipped."""
    group = _Group(
        {
            "label": _Array(value=np.array(["a", "b"], dtype=object)),
            "mesh1d_node_offset": _Array(value=np.array([0.0, 1.0], np.float64)),
            "mesh1d_node_branch": _Array(value=np.array([0, 1], np.int64)),
        }
    )
    branch, offset = _branch_offset_names(
        group, ["label", "mesh1d_node_offset", "mesh1d_node_branch"]
    )
    assert branch == "mesh1d_node_branch"
    assert offset == "mesh1d_node_offset"


def test_reconstruct_returns_none_without_a_branch_offset_pair():
    """A coordinate space but no integer branch index declines reconstruction."""
    group = _Group(
        {
            "x": _Array(value=np.array([0.0], np.float64)),
            "y": _Array(value=np.array([0.0], np.float64)),
        }
    )
    topo = {"coordinate_space": "network", "node_coordinates": "x y"}
    assert _reader()._reconstruct_node_coordinates(group, topo, ["x", "y"]) is None


def test_reconstruct_returns_none_when_network_has_no_geometry():
    """A valid branch+offset pair but a geometry-less network declines reconstruction."""
    group = _Group(
        {
            "mesh1d_node_branch": _Array(value=np.array([0, 1], np.int64)),
            "mesh1d_node_offset": _Array(value=np.array([0.0, 5.0], np.float64)),
            "network": _topo_array(1),
        }
    )
    topo = {
        "coordinate_space": "network",
        "node_coordinates": "mesh1d_node_branch mesh1d_node_offset",
    }
    names = ["mesh1d_node_branch", "mesh1d_node_offset"]
    assert _reader()._reconstruct_node_coordinates(group, topo, names) is None


def _net_topo(edge_geometry=None):
    """Build a fake 1D network ``mesh_topology`` with an optional ``edge_geometry``."""
    attrs = [_Attr("cf_role", "mesh_topology"), _Attr("topology_dimension", 1)]
    if edge_geometry is not None:
        attrs.append(_Attr("edge_geometry", edge_geometry))
    return _Array(attrs=attrs)


def test_network_branch_geometry_none_when_network_absent():
    """No network topology of the given name yields ``None``."""
    assert _network_branch_geometry(_Group({}), "network") is None


def test_network_branch_geometry_none_without_edge_geometry():
    """A network that declares no ``edge_geometry`` yields ``None``."""
    group = _Group({"network": _net_topo()})
    assert _network_branch_geometry(group, "network") is None


def test_network_branch_geometry_none_without_full_geometry():
    """A geometry container missing its vertex x/y or count yields ``None``."""
    geometry = _Array(
        attrs=[_Attr("node_coordinates", "only_x"), _Attr("node_count", None)]
    )
    group = _Group({"network": _net_topo("geom"), "geom": geometry})
    assert _network_branch_geometry(group, "network") is None


def test_network_branch_geometry_falls_back_to_positional_xy():
    """Vertex x/y with no standard name are taken as the first two coordinate entries."""
    group = _Group(
        {
            "network": _net_topo("geom"),
            "geom": _Array(
                attrs=[_Attr("node_coordinates", "gx gy"), _Attr("node_count", "cnt")]
            ),
            "gx": _Array(value=np.array([0.0, 1.0])),
            "gy": _Array(value=np.array([0.0, 0.0])),
            "cnt": _Array(value=np.array([2])),
        }
    )
    geom_x, geom_y, counts = _network_branch_geometry(group, "network")
    assert np.allclose(geom_x, [0.0, 1.0])
    assert np.allclose(geom_y, [0.0, 0.0])
    assert counts.tolist() == [2]


class _RaisingGroup(_Group):
    """A group whose ``OpenMDArray`` raises ``RuntimeError`` for one missing name."""

    def __init__(self, arrays, missing):
        super().__init__(arrays)
        self._missing = missing

    def OpenMDArray(self, name):
        """Raise ``RuntimeError`` for the missing name, else defer to the base group."""
        if name == self._missing:
            raise RuntimeError(f"Array {name} does not exist")
        return super().OpenMDArray(name)


def test_reconstruct_declines_on_a_dangling_geometry_reference():
    """A geometry container named but absent declines, not raises ``RuntimeError``.

    ``build`` documents ``FileFormatNotSupportedError`` for a malformed file; a dangling
    ``edge_geometry`` reference must therefore make reconstruction return ``None`` (the
    caller then declines), never let GDAL's ``RuntimeError`` escape.
    """
    group = _RaisingGroup(
        {
            "mesh1d_node_branch": _Array(value=np.array([0, 1], np.int64)),
            "mesh1d_node_offset": _Array(value=np.array([0.0, 5.0], np.float64)),
            "network": _net_topo("geom"),
        },
        missing="geom",
    )
    topo = {
        "coordinate_space": "network",
        "node_coordinates": "mesh1d_node_branch mesh1d_node_offset",
    }
    names = ["mesh1d_node_branch", "mesh1d_node_offset"]
    assert _reader()._reconstruct_node_coordinates(group, topo, names) is None


def test_sibling_arrays_require_a_name_boundary():
    """A sibling topology's node array (``mesh1d2d_node_x``) is not matched for ``mesh1d``."""
    group = _Group(
        {
            "mesh1d2d_node_x": _Array(
                value=np.array([0.0]),
                attrs=[_Attr("standard_name", "projection_x_coordinate")],
            ),
            "mesh1d2d_node_y": _Array(
                value=np.array([0.0]),
                attrs=[_Attr("standard_name", "projection_y_coordinate")],
            ),
        }
    )
    assert _reader()._xy_from_sibling_arrays(group, "mesh1d") == (None, None)


def test_branch_offset_names_prefers_names_over_an_unstandardised_xy():
    """Name matching keeps a float x/y array from being mistaken for the offset.

    With a float ``mesh1d_node_x`` listed first, a dtype-only rule would take it as the
    offset; matching ``offset`` / ``branch`` by name selects the real arrays instead.
    """
    group = _Group(
        {
            "mesh1d_node_x": _Array(value=np.array([1.0, 2.0], np.float64)),
            "mesh1d_node_offset": _Array(value=np.array([0.0, 5.0], np.float64)),
            "mesh1d_node_branch": _Array(value=np.array([0, 1], np.int64)),
        }
    )
    branch, offset = _branch_offset_names(
        group, ["mesh1d_node_x", "mesh1d_node_offset", "mesh1d_node_branch"]
    )
    assert branch == "mesh1d_node_branch"
    assert offset == "mesh1d_node_offset"


def test_branch_offset_names_handles_the_alternate_spelling_by_name():
    """``..._branch_offset`` is the offset, not the branch, despite containing ``branch``."""
    group = _Group(
        {
            "mesh1d_node_branch_id": _Array(value=np.array([0, 1], np.int64)),
            "mesh1d_node_branch_offset": _Array(value=np.array([0.0, 5.0], np.float64)),
        }
    )
    branch, offset = _branch_offset_names(
        group, ["mesh1d_node_branch_id", "mesh1d_node_branch_offset"]
    )
    assert branch == "mesh1d_node_branch_id"
    assert offset == "mesh1d_node_branch_offset"


def test_network_branch_geometry_none_on_count_vertex_mismatch():
    """A ``node_count`` that does not sum to the vertex-list length yields ``None``."""
    group = _Group(
        {
            "network": _net_topo("geom"),
            "geom": _Array(
                attrs=[_Attr("node_coordinates", "gx gy"), _Attr("node_count", "cnt")]
            ),
            "gx": _Array(value=np.array([0.0, 1.0, 2.0])),
            "gy": _Array(value=np.array([0.0, 0.0, 0.0])),
            "cnt": _Array(value=np.array([2])),  # sums to 2, but 3 vertices exist
        }
    )
    assert _network_branch_geometry(group, "network") is None


def test_edges_raise_without_edge_connectivity():
    """``_edges`` raises when the topology declares no ``edge_node_connectivity``."""
    with pytest.raises(FileFormatNotSupportedError, match="no edge_node_connectivity"):
        _reader()._edges(_Group({}), {})


def test_result_variables_skip_a_non_node_edge_location():
    """A result on the mesh but located on faces is skipped, not mislabelled."""
    array = _Array(attrs=[_Attr("mesh", "mesh1d"), _Attr("location", "face")])
    group = _Group({"mesh1d_face": array})
    assert _reader()._result_variables(group, "mesh1d") == {}


def test_times_is_none_for_a_static_result():
    """``_times`` returns ``None`` when a result array has no dimensions."""
    assert _reader()._times(_Group({}), [], {}) is None


def test_read_time_rejects_a_non_axis_coordinate():
    """``_read_time`` returns ``None`` when the named array is not 1D over its axis."""
    array = _Array(dims=[_Dim("time"), _Dim("extra")])
    group = _Group({"time": array})
    assert _DeltaresReader._read_time(group, "time") is None


def test_crs_is_none_when_no_valid_epsg_is_found():
    """``_crs`` returns ``None`` when the only ``epsg`` attribute is unparseable."""
    group = _Group({"grid": _Array(attrs=[_Attr("epsg", "not-a-code")])})
    assert _reader()._crs(group) is None
