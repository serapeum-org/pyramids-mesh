"""Tests for the template-method base driver and the parsed-mesh intermediate (S2).

Each test drives a stub whose ``_parse`` returns a hand-built :class:`ParsedMesh`, so
the base class's ``read`` pipeline -- parse, validate, construct -- is exercised without
any file format in the way.
"""

import gc

import pyramids  # noqa: F401  # isort:skip

import numpy as np
import pytest

from pyramids.netcdf.ugrid import Network1d, UgridDataset

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable


class _StubDriver(BaseDriver):
    """A driver whose ``_parse`` hands back a pre-built mesh."""

    name = "stub"
    long_name = "Stub format"
    suffixes = (".stub",)
    capabilities = frozenset({"read_mesh", "read_data"})

    def __init__(self, parsed: ParsedMesh) -> None:
        self._parsed = parsed

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        return self._parsed


def _unit_square(**overrides) -> ParsedMesh:
    """A two-triangle unit-square parse, 0-based and counter-clockwise."""
    base = dict(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    base.update(overrides)
    return ParsedMesh(**base)


def test_base_driver_read_is_template_method():
    """A valid stub ``_parse`` flows through to a built ``UgridDataset``."""
    dataset = _StubDriver(_unit_square()).read("mesh.stub")

    assert isinstance(dataset, UgridDataset)
    assert dataset.mesh.n_node == 4
    assert dataset.mesh.n_face == 2


def test_node_z_becomes_variable_not_coordinate():
    """``node_z`` is carried as a node-located variable, never a mesh coordinate."""
    parsed = _unit_square(node_z=np.array([1.0, 2.0, 3.0, 4.0]))

    dataset = _StubDriver(parsed).read("mesh.stub")

    assert "node_z" in dataset.data_variable_names
    assert dataset.get("node_z").location == "node"
    assert not hasattr(dataset.mesh, "node_z")


def test_active_becomes_sibling_variable():
    """A per-step active mask is attached as a sibling ``<var>_active`` variable."""
    parsed = _unit_square(
        variables={
            "wl": ParsedVariable(
                location="face",
                shape=(2,),
                values=np.array([0.5, 0.7]),
                active=np.array([1, 0], dtype=np.int8),
            )
        }
    )

    dataset = _StubDriver(parsed).read("mesh.stub")

    assert "wl_active" in dataset.data_variable_names
    assert dataset.get("wl_active").location == "face"


def test_topology_dimension_1_yields_network1d():
    """A 1D topology builds a ``Network1d`` rather than a ``UgridDataset``."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        faces=None,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        topology_dimension=1,
    )

    result = _StubDriver(parsed).read("channels.stub")

    assert isinstance(result, Network1d)
    assert result.n_node == 3
    assert result.n_edge == 2


def test_loader_survives_driver_garbage_collection(tmp_path):
    """A lazy variable's loader keeps working after the driver object is gone."""
    payload = tmp_path / "wl.npy"
    np.save(payload, np.array([3.0, 4.0]))
    path = str(payload)

    def _load() -> np.ndarray:
        return np.load(path)

    parsed = _unit_square(
        variables={"wl": ParsedVariable(location="face", shape=(2,), loader=_load)}
    )
    driver = _StubDriver(parsed)
    dataset = driver.read("mesh.stub")

    variable = dataset.get("wl")
    assert variable.has_data_source
    del driver, parsed
    gc.collect()

    np.testing.assert_array_equal(variable.data, np.array([3.0, 4.0]))


def test_topology_info_is_none_for_non_netcdf():
    """A non-NetCDF format names no file variables, so ``topology_info`` is ``None``."""
    dataset = _StubDriver(_unit_square()).read("mesh.stub")

    assert dataset._topology_info is None


class _RaisingSniffDriver(_StubDriver):
    """A driver whose content sniff hits an unreadable file."""

    def _sniff(self, path: str) -> bool:
        raise OSError("the file could not be read")


def test_can_open_swallows_os_error_from_sniff(tmp_path):
    """``can_open`` answers False when the sniff cannot read the file, never raises.

    A raising ``can_open`` would abort driver dispatch for every other driver, so the
    base class contains the failure.
    """
    path = tmp_path / "unreadable.stub"
    path.write_text("anything", encoding="utf-8")

    assert _RaisingSniffDriver(_unit_square()).can_open(str(path)) is False


def test_can_open_accepts_a_matching_suffix_by_default(tmp_path):
    """The default ``_sniff`` accepts any file whose suffix the driver claims."""
    path = tmp_path / "mesh.stub"
    path.write_text("content is not inspected by the default sniff", encoding="utf-8")

    assert _StubDriver(_unit_square()).can_open(str(path)) is True


def test_can_open_rejects_a_foreign_suffix(tmp_path):
    """A file the driver does not claim is declined on the suffix alone."""
    path = tmp_path / "mesh.other"
    path.write_text("content", encoding="utf-8")

    assert _StubDriver(_unit_square()).can_open(str(path)) is False


def test_mesh_names_defaults_to_one_entry():
    """A single-mesh format reports exactly one topology name."""
    assert BaseDriver().mesh_names("anything.stub") == ["mesh2d"]


def test_base_parse_is_abstract():
    """``BaseDriver`` is abstract: it is never registered and cannot parse."""
    with pytest.raises(NotImplementedError, match="must implement _parse"):
        BaseDriver()._parse("mesh.stub", lazy=True)


def test_edges_become_edge_connectivity_on_a_2d_mesh():
    """Edges supplied beside faces are carried as ``edge_node_connectivity``.

    Edge-located data needs this; a driver giving both gets both, rather than the
    edges being silently dropped.
    """
    parsed = _unit_square(edges=np.array([[0, 1], [1, 2]], dtype=np.int64))

    dataset = _StubDriver(parsed).read("mesh.stub")

    edge_connectivity = dataset.mesh.edge_node_connectivity
    assert edge_connectivity is not None
    np.testing.assert_array_equal(edge_connectivity.data, [[0, 1], [1, 2]])


def test_temporal_variable_carries_time_values_and_dimensions():
    """A variable with a time coordinate records it and names a time dimension."""
    parsed = _unit_square(
        variables={
            "wl": ParsedVariable(
                location="face",
                shape=(2, 2),
                values=np.array([[0.1, 0.2], [0.3, 0.4]]),
                time_values=np.array([0.0, 1.0]),
                units="m",
                standard_name="sea_surface_height",
                nodata=-999.0,
            )
        }
    )

    variable = _StubDriver(parsed).read("mesh.stub").get("wl")

    assert variable.dimensions == ("time", "n_face")
    assert variable.units == "m"
    assert variable.standard_name == "sea_surface_height"
    assert variable.nodata == -999.0
    np.testing.assert_allclose(variable.attributes["time_values"], [0.0, 1.0])
