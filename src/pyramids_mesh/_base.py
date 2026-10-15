"""Driver scaffolding: the template-method base class and the parsed-mesh intermediate.

Holds ``BaseDriver``, whose ``read`` is a template method running ``_parse`` ->
``validate_and_normalise`` -> construct, so a format driver implements ``_parse`` and
nothing structural. Also holds ``ParsedMesh`` / ``ParsedVariable``, this package's
internal intermediate between a file and ``pyramids.netcdf.ugrid`` types.

``ParsedMesh`` is deliberately richer than pyramids' ``construct.MeshArrays`` tuple,
which cannot carry edge connectivity, per-variable time, per-variable nodata or units,
a WKT CRS (Selafin and DFSU have no EPSG), a 1D network, or node elevation.

See ``planning/design.md`` sections 3, 6, 17 and 19. Implemented by task S2.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pyramids.netcdf.ugrid import (
    MESH_DRIVER_API_VERSION,
    Connectivity,
    Mesh2d,
    MeshVariable,
    Network1d,
    UgridDataset,
)

from pyramids_mesh._validate import validate_and_normalise

# The axis-name a one-dimensional variable carries per location, used only to label
# the dataset's dimensions; the element count itself lives in ``shape``.
_LOCATION_DIMENSION = {"node": "n_node", "face": "n_face", "edge": "n_edge"}


def _recorded_start_index(parsed: "ParsedMesh") -> int:
    """Return the connectivity base to record on ``Connectivity`` for a round trip.

    Args:
        parsed: The validated mesh.

    Returns:
        int: ``original_start_index`` when the driver set one, else ``start_index``.
    """
    if parsed.original_start_index is None:
        return parsed.start_index
    return parsed.original_start_index


@dataclass
class ParsedVariable:
    """One data variable as a driver's ``_parse`` produced it, before construction.

    Either ``values`` (eager) or ``loader`` (lazy) carries the array; ``shape`` is
    always set from the file header so the lazy path never forces a read to learn it.

    Attributes:
        location: Mesh location the variable lives on -- ``"node"``, ``"face"`` or
            ``"edge"``.
        shape: The variable's array shape, correct before any lazy load.
        values: The eager array, or ``None`` when the variable is lazy.
        loader: A zero-argument callable returning the array, or ``None`` when eager.
            It must close over a path, never an open file handle (see design section 7).
        units: The variable's units, if the file declares them.
        standard_name: A CF standard name, if known.
        nodata: The fill/no-data value, if the file declares one.
        time_values: The time coordinate, when the variable varies in time.
        active: A per-step active (wet/dry) mask, attached as a sibling variable rather
            than folded into the values (see design section 21.2).
    """

    location: str
    shape: tuple[int, ...]
    values: np.ndarray | None = None
    loader: Callable[[], np.ndarray] | None = None
    units: str | None = None
    standard_name: str | None = None
    nodata: float | None = None
    time_values: np.ndarray | None = None
    active: np.ndarray | None = None


@dataclass
class ParsedMesh:
    """A mesh as a driver's ``_parse`` produced it, before validation and construction.

    Coordinates and connectivity are as the file gave them: ``faces``/``edges`` use the
    file's own base (``start_index``) and pad value (``fill_value``) until
    ``validate_and_normalise`` rewrites them to the 0-based, ``-1``-padded,
    counter-clockwise form pyramids assumes. ``start_index`` is preserved through
    validation so a writer can restore the file's base.

    Attributes:
        node_x: Node x-coordinates, shape ``(n_node,)``.
        node_y: Node y-coordinates, shape ``(n_node,)``.
        faces: Face-node connectivity, ``(n_face, max_nodes)``, fill-padded; ``None``
            for a pure 1D network.
        edges: This mesh's own edge-node connectivity, ``(n_edge, 2)``, as
            ``Mesh2d.edge_node_connectivity`` -- what edge-located data is indexed by.
        network_edges: A **separate 1D topology's** edges, ``(n_edge, 2)``. Kept apart
            from ``edges`` because they are different things: 2DM's ``E2L`` elements are
            channel elements, not the edges of the surface mesh, and feeding them to
            ``Mesh2d.edge_node_connectivity`` would silently mislabel them. A file
            holding both names its topologies separately, and the caller picks one with
            ``mesh_name``.
        fill_value: The pad value the file used for short face rows.
        start_index: The base ``validate_and_normalise`` subtracts from connectivity.
            It is the file's own base for a format whose ids are contiguous; a driver
            that resolves ids through ``_text.native_id_map`` has already produced
            0-based indices and leaves this ``0``. Validation sets it to ``0`` on its
            output, since by then there is nothing left to subtract.
        original_start_index: The base a **writer** should emit, recorded on
            ``Connectivity``. It is deliberately *not* "the base the file used": a
            format whose ids are renumbered or gapped has no single base, so there is
            nothing to round-trip -- what a writer needs is the numbering the format
            specifies. Defaults to ``start_index`` when ``None``, so a driver whose
            ids are contiguous need not set it; an id-mapping driver sets it to the
            base its format writes.
        node_z: Node elevation, carried as a node variable rather than a coordinate
            (see design section 6); ``None`` when the format has none.
        face_x: Face-centre x-coordinates, if the file provides them.
        face_y: Face-centre y-coordinates, if the file provides them.
        crs: The coordinate reference system as an EPSG int or a WKT string; ``None``
            when the format carries none.
        variables: Data variables keyed by name.
        global_attributes: File-level attributes to carry onto the dataset.
        topology_dimension: ``2`` for a mesh, ``1`` for a network.
        mesh_name: The topology name to build under.
    """

    node_x: np.ndarray
    node_y: np.ndarray
    faces: np.ndarray | None = None
    edges: np.ndarray | None = None
    network_edges: np.ndarray | None = None
    fill_value: int = -1
    start_index: int = 0
    original_start_index: int | None = None
    node_z: np.ndarray | None = None
    face_x: np.ndarray | None = None
    face_y: np.ndarray | None = None
    crs: str | int | None = None
    variables: dict[str, ParsedVariable] = field(default_factory=dict)
    global_attributes: dict[str, Any] = field(default_factory=dict)
    topology_dimension: int = 2
    mesh_name: str = "mesh2d"


class BaseDriver:
    """Shared scaffolding for every file-format driver.

    A subclass sets the class attributes and implements ``_parse`` (and usually
    ``_sniff``); ``read`` is a template method that must not be overridden, so the
    0-based / fill-padded / counter-clockwise invariants cannot be skipped. The
    attribute and method names mirror the ``MeshDriver`` protocol pyramids discovers.

    Attributes:
        name: The unique driver name, also the value accepted by ``driver=``.
        long_name: A human-readable name for listings.
        suffixes: The lowercase, dot-prefixed suffixes this driver claims.
        api_version: Must equal pyramids' ``MESH_DRIVER_API_VERSION``.
        capabilities: The capability tokens this driver declares.
    """

    name: str = ""
    long_name: str = ""
    suffixes: tuple[str, ...] = ()
    api_version: int = MESH_DRIVER_API_VERSION
    capabilities: frozenset[str] = frozenset()

    # The name under which a separate 1D topology is offered. A format carrying both a
    # surface and a network names them separately, and the caller selects one through
    # `mesh_name`; this is the 1D half of that pair.
    network_mesh_name: str = "network1d"

    def can_open(self, path: str) -> bool:
        """Report whether this driver recognises the file, without ever raising.

        The default is a suffix match followed by ``_sniff``. An empty file, a file of
        another format, a directory or a missing path all answer ``False``; they never
        raise, so a broken candidate cannot abort driver dispatch.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` only when the suffix matches and ``_sniff`` accepts the file.
        """
        try:
            if not os.path.isfile(path):
                return False
            if not self._has_suffix(path):
                return False
            return self._sniff(path)
        except OSError:
            return False

    def mesh_names(self, path: str) -> list[str]:
        """List the topology names in the file.

        A single-mesh format returns a one-item list; ``multi_mesh`` drivers override
        this to read the names from the file.

        Args:
            path: The filesystem path, unused by the single-mesh default.

        Returns:
            list[str]: The configured mesh name, as a one-item list.
        """
        return [self._default_mesh_name]

    def read(
        self, path: str, *, mesh_name: str | None = None, lazy: bool = True
    ) -> UgridDataset | Network1d:
        """Read the file into a fully-built pyramids object.

        Template method: ``_parse`` turns the file into a :class:`ParsedMesh`,
        ``validate_and_normalise`` enforces pyramids' invariants, and the result is
        constructed into a :class:`~pyramids.netcdf.ugrid.UgridDataset` -- or a
        :class:`~pyramids.netcdf.ugrid.Network1d` when the topology is one-dimensional.
        Do not override it; a driver that bypasses validation loses every invariant.

        Args:
            path: The file to read.
            mesh_name: The topology name to build under; defaults to the parsed name.
            lazy: Forwarded to ``_parse`` so a format with a real deferred loader can
                honour it.

        Returns:
            UgridDataset | Network1d: The built mesh, or a network for a 1D topology.

        Examples:
            - A driver implements ``_parse`` and inherits everything else. Here the
              file is 1-based and its single face is wound clockwise; the template
              method re-bases the connectivity and reverses the face, so neither
              mistake can reach the caller:

                >>> import numpy as np
                >>> from pyramids_mesh._base import BaseDriver, ParsedMesh
                >>> class TinyDriver(BaseDriver):
                ...     name = "tiny"
                ...     suffixes = (".tiny",)
                ...     capabilities = frozenset({"read_mesh"})
                ...
                ...     def _parse(self, path, *, lazy):
                ...         return ParsedMesh(
                ...             node_x=np.array([0.0, 1.0, 1.0]),
                ...             node_y=np.array([0.0, 1.0, 0.0]),
                ...             faces=np.array([[1, 2, 3]]),
                ...             start_index=1,
                ...         )
                >>> dataset = TinyDriver().read("anything.tiny")
                >>> dataset.mesh.n_node, dataset.mesh.n_face
                (3, 1)
                >>> dataset.mesh.face_node_connectivity.data.tolist()
                [[2, 1, 0]]

            - A ``_parse`` reporting a one-dimensional topology builds a network
              instead, which is a separate type with edges rather than faces:

                >>> import numpy as np
                >>> from pyramids_mesh._base import BaseDriver, ParsedMesh
                >>> class TinyNetworkDriver(BaseDriver):
                ...     name = "tiny1d"
                ...     suffixes = (".tiny",)
                ...     capabilities = frozenset({"read_mesh"})
                ...
                ...     def _parse(self, path, *, lazy):
                ...         return ParsedMesh(
                ...             node_x=np.array([0.0, 1.0, 2.0]),
                ...             node_y=np.array([0.0, 0.0, 0.0]),
                ...             edges=np.array([[0, 1], [1, 2]]),
                ...             topology_dimension=1,
                ...         )
                >>> network = TinyNetworkDriver().read("channels.tiny")
                >>> network.n_node, network.n_edge
                (3, 2)
                >>> network.edge_node_connectivity.data.tolist()
                [[0, 1], [1, 2]]
        """
        parsed = self._parse(path, lazy=lazy)
        parsed = validate_and_normalise(parsed)
        return self._construct(parsed, path=path, mesh_name=mesh_name)

    # -- hooks a driver implements -------------------------------------------------

    def _sniff(self, path: str) -> bool:
        """Content check behind ``can_open``; the default accepts any suffix match.

        Args:
            path: The file to inspect; the default ignores it.

        Returns:
            bool: ``True``. Subclasses override to read a cheap header signature.
        """
        return True

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Turn the file into a :class:`ParsedMesh`; the one method a driver writes.

        Args:
            path: The file to parse.
            lazy: When ``True``, defer variable values behind a loader.

        Returns:
            ParsedMesh: The file's topology and variable metadata as plain arrays.

        Raises:
            NotImplementedError: Always, on the base class. ``BaseDriver`` is abstract
                and is never registered; a concrete driver overrides this.
        """
        raise NotImplementedError("a driver must implement _parse")

    # -- shared construction -------------------------------------------------------

    @property
    def _default_mesh_name(self) -> str:
        """The mesh name used when a caller names none."""
        return "mesh2d"

    def _has_suffix(self, path: str) -> bool:
        """Whether the path ends with one of this driver's declared suffixes."""
        lowered = path.lower()
        return any(lowered.endswith(suffix) for suffix in self.suffixes)

    def _construct(
        self, parsed: ParsedMesh, *, path: str, mesh_name: str | None
    ) -> UgridDataset | Network1d:
        """Build pyramids objects from a validated :class:`ParsedMesh`.

        Args:
            parsed: The validated, normalised mesh.
            path: The source path, recorded on the dataset.
            mesh_name: The topology name to build under; defaults to the parsed name.

        Returns:
            UgridDataset | Network1d: A network for a 1D topology, else a dataset.
        """
        node_x = np.asarray(parsed.node_x, dtype=np.float64)
        node_y = np.asarray(parsed.node_y, dtype=np.float64)
        if self._wants_network(parsed, mesh_name):
            return self._construct_network(parsed, node_x, node_y)
        name = mesh_name or parsed.mesh_name
        return self._construct_dataset(parsed, node_x, node_y, name=name, path=path)

    def _wants_network(self, parsed: ParsedMesh, mesh_name: str | None) -> bool:
        """Decide whether this read builds the 1D topology rather than the surface.

        A purely one-dimensional parse always does. A file holding both offers the
        network under :attr:`network_mesh_name`, so the caller selects it explicitly.

        Args:
            parsed: The validated mesh.
            mesh_name: The topology the caller asked for, if any.

        Returns:
            bool: ``True`` to build a ``Network1d``.

        Raises:
            ValueError: If the network was asked for but the file holds no 1D edges.
        """
        if mesh_name == self.network_mesh_name:
            if self._network_edges(parsed) is None:
                raise ValueError(
                    f"{self.name!r} was asked for the {mesh_name!r} topology, but this "
                    f"file holds no 1D elements; it carries only "
                    f"{parsed.mesh_name!r}"
                )
            return True
        return parsed.topology_dimension == 1

    @staticmethod
    def _network_edges(parsed: ParsedMesh) -> np.ndarray | None:
        """Return the 1D topology's edges, whichever field a driver put them in.

        A format that is only ever one-dimensional has nothing to disambiguate and puts
        its edges in ``edges``; one carrying a surface mesh as well uses
        ``network_edges`` so the two are never confused.
        """
        if parsed.network_edges is not None:
            return parsed.network_edges
        if parsed.faces is None:
            return parsed.edges
        return None

    def _construct_network(
        self, parsed: ParsedMesh, node_x: np.ndarray, node_y: np.ndarray
    ) -> Network1d:
        """Build a :class:`~pyramids.netcdf.ugrid.Network1d` from a 1D parse.

        ``Network1d`` carries a topology and a CRS and nothing else -- it has no
        parameter for data variables, node elevation or global attributes -- so anything
        the file carried beside the topology cannot come along. Rather than drop it
        silently, each omission is warned about by name. Carrying it properly needs
        ``Network1d`` to grow somewhere to put it, which is upstream work in
        ``pyramids-gis``, not something to fake here.
        """
        self._warn_network_drops(parsed)
        edge_conn = Connectivity(
            data=np.asarray(self._network_edges(parsed)),
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=_recorded_start_index(parsed),
        )
        return Network1d(
            node_x, node_y, edge_node_connectivity=edge_conn, crs=parsed.crs
        )

    def _construct_dataset(
        self,
        parsed: ParsedMesh,
        node_x: np.ndarray,
        node_y: np.ndarray,
        *,
        name: str,
        path: str,
    ) -> UgridDataset:
        """Build a :class:`~pyramids.netcdf.ugrid.UgridDataset` from a 2D parse.

        When the file also holds a separate 1D topology, this result is only its surface
        half, so the other half is named rather than quietly left out. The warning
        belongs here and not in ``_parse``: only at construction is it known which
        topology the caller actually asked for.
        """
        if parsed.network_edges is not None:
            warnings.warn(
                f"this file also holds a {self.network_mesh_name!r} topology, which is "
                f"not in this result; read it with "
                f"Network1d.read_file(path, mesh_name={self.network_mesh_name!r})",
                stacklevel=4,
            )
        face_conn = Connectivity(
            data=np.asarray(parsed.faces),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=_recorded_start_index(parsed),
        )
        edge_conn = None
        if parsed.edges is not None:
            edge_conn = Connectivity(
                data=np.asarray(parsed.edges),
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=_recorded_start_index(parsed),
            )
        mesh = Mesh2d(
            node_x,
            node_y,
            face_node_connectivity=face_conn,
            edge_node_connectivity=edge_conn,
            face_x=parsed.face_x,
            face_y=parsed.face_y,
            crs=parsed.crs,
        )
        data_variables = self._build_variables(parsed, name)
        crs_wkt = parsed.crs if isinstance(parsed.crs, str) else None
        return UgridDataset(
            mesh=mesh,
            data_variables=data_variables,
            global_attributes=dict(parsed.global_attributes),
            topology_info=None,
            crs_wkt=crs_wkt,
            file_name=str(path),
        )

    @staticmethod
    def _warn_network_drops(parsed: ParsedMesh) -> None:
        """Warn by name about anything a ``Network1d`` cannot carry.

        Args:
            parsed: The validated mesh about to become a network.
        """
        dropped: list[str] = []
        if parsed.node_z is not None:
            dropped.append("node_z")
        dropped.extend(sorted(parsed.variables))
        if parsed.global_attributes:
            dropped.append("global_attributes")
        if not dropped:
            return
        warnings.warn(
            f"Network1d carries a topology and a CRS only, so {', '.join(dropped)} "
            f"could not be attached to this 1D mesh and {'is' if len(dropped) == 1 else 'are'} "
            f"not in the result",
            stacklevel=3,
        )

    def _build_variables(
        self, parsed: ParsedMesh, name: str
    ) -> dict[str, MeshVariable]:
        """Turn parsed variables (and ``node_z``) into ``MeshVariable`` objects.

        A per-step active mask on a variable becomes a sibling face-located variable
        named ``<var>_active`` rather than being folded into the values as NaN, so the
        information survives and stays inspectable (design section 21.2). ``node_z``
        becomes a node-located variable, never a coordinate (design section 6).

        Args:
            parsed: The validated mesh carrying the variables.
            name: The mesh name every variable is built under.

        Returns:
            dict[str, MeshVariable]: The data variables, keyed by name.
        """
        built: dict[str, MeshVariable] = {}
        for var_name, parsed_var in parsed.variables.items():
            built[var_name] = self._build_variable(var_name, parsed_var, name)
            if parsed_var.active is not None:
                active_name = f"{var_name}_active"
                built[active_name] = MeshVariable(
                    name=active_name,
                    location="face",
                    mesh_name=name,
                    shape=tuple(np.shape(parsed_var.active)),
                    _data=np.asarray(parsed_var.active),
                )
        if parsed.node_z is not None:
            built["node_z"] = MeshVariable(
                name="node_z",
                location="node",
                mesh_name=name,
                shape=(len(parsed.node_x),),
                _data=np.asarray(parsed.node_z, dtype=np.float64),
            )
        return built

    @staticmethod
    def _build_variable(
        var_name: str, parsed_var: ParsedVariable, name: str
    ) -> MeshVariable:
        """Build one ``MeshVariable`` from a :class:`ParsedVariable`."""
        attributes: dict[str, Any] = {}
        if parsed_var.time_values is not None:
            attributes["time_values"] = np.asarray(parsed_var.time_values)
        dimension = _LOCATION_DIMENSION.get(parsed_var.location, "n_elem")
        dimensions: tuple[str, ...]
        if parsed_var.time_values is not None:
            dimensions = ("time", dimension)
        else:
            dimensions = (dimension,)
        return MeshVariable(
            name=var_name,
            location=parsed_var.location,
            mesh_name=name,
            shape=tuple(parsed_var.shape),
            attributes=attributes,
            nodata=parsed_var.nodata,
            units=parsed_var.units,
            standard_name=parsed_var.standard_name,
            dimensions=dimensions,
            _data=parsed_var.values,
            _loader=parsed_var.loader,
        )
