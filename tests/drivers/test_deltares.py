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

from pyramids.netcdf.ugrid import Network1d
from pyramids_mesh.drivers.deltares import DeltaresNetworkDriver

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
