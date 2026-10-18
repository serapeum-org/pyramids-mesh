"""Tests for the Deltares 1D network reader (C1), against real committed samples.

Delft3D FM / D-HYDRO writes 1D hydrodynamic models as UGRID-1.0 + Deltares-extension
NetCDF: a ``network`` of branches and a ``mesh1d`` discretisation of computational nodes,
with ``mesh1d_*`` results over time on a map file. The driver builds a ``Network1d`` from
the ``mesh1d`` discretisation, finding both topologies by their ``cf_role`` attributes.

Three genuine samples are committed under ``tests/data/deltares1d/``: a pure-1D topology
(``korte-woerden``), a tiny combined 1D2D topology (``UGrid_1D2D``), and a pure-1D map
file with 97 result steps (``Haskoning``). The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import Network1d
from pyramids_mesh.drivers.deltares import (
    DeltaresNetworkDriver,
    _DeltaresReader,
    _as_str,
    _attr_map,
    _fill_value,
    _split,
    _string_attr,
    _topology_dimension,
)

_DATA = Path(__file__).parent.parent / "data" / "deltares1d"
_KORTE = _DATA / "korte-woerden-1d_FlowFM_net.nc"
_UGRID_1D2D = _DATA / "UGrid_1D2D_net.nc"
_HASKONING = _DATA / "Haskoning_example_map.nc"
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


def test_pure_1d_network_reads_node_and_edge_counts():
    """The korte-woerden mesh1d discretisation reads as an 86-node, 91-edge network."""
    network = _read(_KORTE)
    assert isinstance(network, Network1d)
    assert network.n_node == 86
    assert network.n_edge == 91
    data = network.edge_node_connectivity.data
    assert data.shape == (91, 2)
    assert int(data.min()) == 0
    assert int(data.max()) == 85


def test_both_sample_topologies_parse_to_a_network():
    """Both the pure-1D and the combined-1D2D samples yield a mesh1d Network1d.

    The two topologies are found by cf_role, so a file is read the same regardless of
    how it spells its topologies; the combined 1D2D file's 1D half reads via a direct
    ``read`` even though ``can_open`` leaves it to the UGRID reader.
    """
    korte = _read(_KORTE)
    combined = _read(_UGRID_1D2D)
    assert isinstance(korte, Network1d)
    assert isinstance(combined, Network1d)
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
        _reader()._node_coordinates(group, topo)


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
