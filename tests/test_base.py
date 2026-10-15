"""Tests for the template-method base driver and the parsed-mesh intermediate (S2).

Each test drives a stub whose ``_parse`` returns a hand-built :class:`ParsedMesh`, so
the base class's ``read`` pipeline -- parse, validate, construct -- is exercised without
any file format in the way.
"""

import gc
import warnings

import pyramids  # noqa: F401  # isort:skip

import numpy as np
import pytest

from pyramids.netcdf.ugrid import Network1d, UgridDataset

from pyramids_mesh._base import (
    BaseDriver,
    ParsedMesh,
    ParsedVariable,
    _recorded_start_index,
)


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


def test_active_sibling_inherits_its_parent_location():
    """The mask describes its parent's elements, so it lives where the parent lives."""
    parsed = _unit_square(
        variables={
            "wl": ParsedVariable(
                location="node",
                shape=(4,),
                values=np.zeros(4),
                active=np.array([1, 1, 0, 1], dtype=np.int8),
            )
        }
    )

    sibling = _StubDriver(parsed).read("mesh.stub").get("wl_active")

    assert sibling.location == "node"
    assert sibling.shape == (4,)


def test_wrongly_sized_active_mask_raises():
    """A mask that does not describe its parent's elements is refused.

    The sibling is synthesised after ``check_shapes`` and is not in
    ``parsed.variables``, so nothing downstream would ever question its shape.
    """
    parsed = _unit_square(
        variables={
            "depth": ParsedVariable(
                location="face",
                shape=(2,),
                values=np.zeros(2),
                active=np.array([1, 0, 1, 0, 1], dtype=np.int8),
            )
        }
    )

    with pytest.raises(ValueError, match="active mask of 'depth'"):
        _StubDriver(parsed).read("mesh.stub")


def test_active_sibling_carries_the_parents_dimensions():
    """A per-step mask is labelled with the same dimensions as the variable it masks."""
    parsed = _unit_square(
        variables={
            "wl": ParsedVariable(
                location="face",
                shape=(2, 2),
                values=np.zeros((2, 2)),
                time_values=np.array([0.0, 1.0]),
                active=np.ones((2, 2), dtype=np.int8),
            )
        }
    )

    dataset = _StubDriver(parsed).read("mesh.stub")

    assert dataset.get("wl_active").dimensions == ("time", "n_face")
    assert dataset.get("wl_active").shape == (2, 2)


def test_node_z_colliding_with_a_parsed_variable_raises():
    """A file defining its own ``node_z`` variable is not silently overwritten."""
    parsed = _unit_square(
        node_z=np.array([9.0, 9.0, 9.0, 9.0]),
        variables={
            "node_z": ParsedVariable(
                location="face", shape=(2,), values=np.array([42.0, 43.0])
            )
        },
    )

    with pytest.raises(ValueError, match="both a node_z coordinate"):
        _StubDriver(parsed).read("mesh.stub")


def test_active_sibling_colliding_with_a_parsed_variable_raises():
    """A generated ``<var>_active`` name that the file already uses is an error."""
    parsed = _unit_square(
        variables={
            "wl": ParsedVariable(
                location="face",
                shape=(2,),
                values=np.zeros(2),
                active=np.array([1, 0], dtype=np.int8),
            ),
            "wl_active": ParsedVariable(
                location="face", shape=(2,), values=np.zeros(2)
            ),
        }
    )

    with pytest.raises(ValueError, match="already defines"):
        _StubDriver(parsed).read("mesh.stub")


def test_face_centres_are_converted_to_float_arrays():
    """Face centres given as plain lists reach ``Mesh2d`` as float64 arrays."""
    parsed = _unit_square(face_x=[0.5, 0.5], face_y=[0.25, 0.75])

    mesh = _StubDriver(parsed).read("mesh.stub").mesh

    assert mesh.face_x.dtype == np.float64
    np.testing.assert_allclose(mesh.face_y, [0.25, 0.75])


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


class _BadSniffDriver(_StubDriver):
    """A driver whose content sniff has a bug unrelated to the filesystem."""

    def _sniff(self, path: str) -> bool:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "contrived")


@pytest.mark.parametrize("driver_class", [_RaisingSniffDriver, _BadSniffDriver])
def test_can_open_never_raises_whatever_the_sniff_does(tmp_path, driver_class):
    """``can_open`` declines and warns for any sniff failure, not just ``OSError``.

    A raising ``can_open`` would abort driver dispatch for every other driver, so the
    base class contains the failure -- but names it, since a sniff that raises is a bug.
    """
    path = tmp_path / "unreadable.stub"
    path.write_text("anything", encoding="utf-8")

    with pytest.warns(UserWarning, match="failed to sniff"):
        assert driver_class(_unit_square()).can_open(str(path)) is False


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


def test_recorded_start_index_falls_back_to_start_index():
    """Construction still records a base if it is handed an unvalidated mesh.

    ``validate_and_normalise`` always resolves ``original_start_index``, so this
    fallback is unreachable through ``read``; it keeps a direct ``_construct`` call from
    handing ``None`` to ``Connectivity``.
    """
    parsed = _unit_square(start_index=1, original_start_index=None)

    assert _recorded_start_index(parsed) == 1


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


def test_network_construction_warns_about_what_it_cannot_carry():
    """A 1D build names everything ``Network1d`` has no room for, instead of dropping it.

    ``Network1d`` takes a topology and a CRS only, so variables, node elevation and
    global attributes cannot come along. The loss is unavoidable until upstream grows
    somewhere to put them; being silent about it is not.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        faces=None,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        node_z=np.array([5.0, 6.0, 7.0]),
        variables={
            "wl": ParsedVariable(location="node", shape=(3,), values=np.zeros(3))
        },
        global_attributes={"source": "test"},
        topology_dimension=1,
    )

    with pytest.warns(UserWarning, match="node_z") as record:
        result = _StubDriver(parsed).read("channels.stub")

    assert isinstance(result, Network1d)
    message = str(record[0].message)
    assert "wl" in message
    assert "global_attributes" in message


def test_network_construction_is_quiet_when_there_is_nothing_to_drop():
    """A bare 1D topology carries everything it had, so nothing is warned about."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0]),
        node_y=np.array([0.0, 0.0]),
        faces=None,
        edges=np.array([[0, 1]], dtype=np.int64),
        topology_dimension=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _StubDriver(parsed).read("channels.stub")

    assert result.n_edge == 1


def test_network_drop_warning_is_singular_for_a_single_omission():
    """One omission reads "is not in the result", not "are".

    The plural is chosen by a conditional expression inside the f-string, which no
    branch report can see: the message is built on every 1D read, so a wrong agreement
    would be executed by the whole suite and caught by none of it.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0]),
        node_y=np.array([0.0, 0.0]),
        faces=None,
        edges=np.array([[0, 1]], dtype=np.int64),
        global_attributes={"source": "test"},
        topology_dimension=1,
    )

    with pytest.warns(UserWarning) as record:
        _StubDriver(parsed).read("channels.stub")

    message = str(record[0].message)
    assert "so global_attributes could not be attached" in message
    assert "and is not in the result" in message
    assert "are not in the result" not in message


def test_edge_located_variable_names_the_edge_dimension():
    """An edge-located variable is built against ``n_edge``, not a face or node axis.

    ``_LOCATION_DIMENSION`` carries three entries and the suite only ever built the
    first two, so an edge variable mislabelled as ``n_face`` would have gone unnoticed.
    """
    parsed = _unit_square(
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        variables={
            "flux": ParsedVariable(
                location="edge", shape=(2,), values=np.array([1.0, 2.0])
            )
        },
    )

    variable = _StubDriver(parsed).read("mesh.stub").get("flux")

    assert variable.location == "edge"
    assert variable.dimensions == ("n_edge",)


def test_static_variable_names_only_its_location_dimension():
    """A variable with no time coordinate carries one dimension and no time values."""
    parsed = _unit_square(
        variables={
            "material": ParsedVariable(
                location="face", shape=(2,), values=np.array([1.0, 2.0])
            )
        }
    )

    variable = _StubDriver(parsed).read("mesh.stub").get("material")

    assert variable.dimensions == ("n_face",)
    assert "time_values" not in variable.attributes


def test_mesh_name_argument_names_the_built_topology():
    """``mesh_name`` selects the name every variable is built under, not just a label.

    A variable whose ``mesh_name`` does not match the mesh it belongs to cannot be
    resolved back to its topology, and nothing else in the suite pins the argument to
    the variables it reaches.
    """
    parsed = _unit_square(
        variables={
            "material": ParsedVariable(
                location="face", shape=(2,), values=np.array([1.0, 2.0])
            )
        }
    )

    dataset = _StubDriver(parsed).read("mesh.stub", mesh_name="mesh2d")

    assert dataset.get("material").mesh_name == "mesh2d"


def test_an_unoffered_mesh_name_is_refused():
    """A topology the driver does not offer is refused, not quietly swapped.

    Asking for a network that is not in the file already raised, so silently honouring
    a typo by returning a different topology was the asymmetry -- and for a multi_mesh
    driver ``mesh_name`` is the only way to choose a half.
    """
    with pytest.raises(ValueError, match="does not offer a 'not-a-topology'"):
        _StubDriver(_unit_square()).read("mesh.stub", mesh_name="not-a-topology")


def test_a_driver_whose_mesh_names_lies_is_reported():
    """A driver offering a network it did not parse is named as the bug it is."""

    class _LyingDriver(_StubDriver):
        def mesh_names(self, path: str) -> list[str]:
            return ["mesh2d", "network1d"]

    with pytest.raises(ValueError, match="mesh_names and _parse disagree"):
        _LyingDriver(_unit_square()).read("mesh.stub", mesh_name="network1d")


def test_suffix_matching_ignores_case(tmp_path):
    """A suffix the operating system upper-cased is still claimed.

    ``_has_suffix`` lower-cases the path for exactly this, and every other test feeds
    it an already-lowercase name, so the folding was executed but never proved.
    """
    path = tmp_path / "MESH.STUB"
    path.write_text("content", encoding="utf-8")

    assert _StubDriver(_unit_square()).can_open(str(path)) is True


def test_can_open_propagates_a_bad_argument_type():
    """A value that is not a path at all is the caller's mistake, so it is not caught.

    ``can_open`` promises never to raise *a verdict about a file*, and the path checks
    sit outside the guard on purpose. Pulling ``os.path.isfile`` inside it would turn
    the mistake into a silent ``False`` plus a warning blaming the driver for a file
    that was never named -- and the guard is proved to swallow everything else, so
    nothing pinned the boundary of what it swallows.
    """
    with pytest.raises(TypeError):
        _StubDriver(_unit_square()).can_open(None)


def test_the_lazy_flag_reaches_the_driver_parse():
    """``read`` forwards ``lazy`` to ``_parse``, and defers by default.

    ``_parse`` is the only place a format can honour it, every stub in the suite
    ignores it, and the invariant harness only ever passes ``True`` -- so ``read``
    could have pinned the flag and no test would have noticed.
    """

    class _LazySpy(_StubDriver):
        """A stub recording the ``lazy`` value each read hands to ``_parse``."""

        def __init__(self, parsed: ParsedMesh) -> None:
            super().__init__(parsed)
            self.seen: list[bool] = []

        def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
            self.seen.append(lazy)
            return super()._parse(path, lazy=lazy)

    spy = _LazySpy(_unit_square())
    spy.read("mesh.stub", lazy=False)
    spy.read("mesh.stub", lazy=True)
    spy.read("mesh.stub")

    assert spy.seen == [False, True, True], f"_parse saw {spy.seen}"


_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


@pytest.mark.parametrize(
    "crs, expected_wkt",
    [
        pytest.param(None, None, id="no-crs"),
        pytest.param(4326, None, id="epsg-int"),
        pytest.param(_WKT, _WKT, id="wkt-string"),
    ],
)
def test_crs_reaches_crs_wkt_only_when_it_is_a_string(crs, expected_wkt):
    """``crs_wkt`` carries a WKT CRS and nothing else; an EPSG int is not stringified.

    ``ParsedMesh`` accepts both because Selafin and DFSU carry no EPSG, and the
    dataset's ``crs_wkt`` is filled from the string case alone. Every fixture in the
    suite parses ``crs=None``, so the type test was only ever executed one way and an
    int written into ``crs_wkt`` as ``"4326"`` would have gone unnoticed.
    """
    dataset = _StubDriver(_unit_square(crs=crs)).read("mesh.stub")

    assert dataset.crs_wkt == expected_wkt, f"crs_wkt was {dataset.crs_wkt!r}"
    assert dataset.mesh.crs == crs, f"the mesh lost the CRS: {dataset.mesh.crs!r}"


def test_network_edges_win_over_edges_when_a_parse_carries_both():
    """``network_edges`` is the 1D topology, so it is preferred over ``edges``.

    A real driver sets one or the other, so the precedence in ``_network_edges`` was
    only ever executed one branch at a time. Swapping the two would hand a mixed
    file's surface edges to ``Network1d`` as its channels -- the mislabelling the two
    separate fields exist to prevent.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        faces=None,
        edges=np.array([[0, 1]], dtype=np.int64),
        network_edges=np.array([[1, 2]], dtype=np.int64),
        topology_dimension=1,
    )

    network = _StubDriver(parsed).read("channels.stub")

    np.testing.assert_array_equal(network.edge_node_connectivity.data, [[1, 2]])


def test_a_one_dimensional_parse_builds_a_network_under_any_offered_name():
    """A 1D parse has no surface to build, so the surface name still yields a network.

    ``_wants_network`` ends by re-reading ``topology_dimension`` for a name that is not
    the network's own, and only the ``False`` outcome of that line was ever produced.
    The ``True`` one is what keeps ``mesh_name='mesh2d'`` from constructing a
    ``Mesh2d`` out of a parse that has no faces at all.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        faces=None,
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        topology_dimension=1,
    )

    result = _StubDriver(parsed).read("channels.stub", mesh_name="mesh2d")

    assert isinstance(result, Network1d)
    assert result.n_edge == 2


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
