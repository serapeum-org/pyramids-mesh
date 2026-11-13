"""Deltares 1D network reader (task C1).

Delft3D FM / D-HYDRO writes its 1D hydrodynamic models as UGRID-1.0 NetCDF with the
Deltares extension. Such a file carries two one-dimensional topologies: a ``network`` of
branches (``network_node_x``/``network_node_y``, ``network_edge_nodes``, branch geometry)
and a ``mesh1d`` discretisation of computational nodes along those branches
(``mesh1d_node_x``/``mesh1d_node_y``, ``mesh1d_edge_nodes``, with a branch index and an
offset per node). A map file adds ``mesh1d_*`` result variables over ``time``.

This reader builds a :class:`~pyramids.netcdf.ugrid.Mesh1d` from the ``mesh1d``
discretisation -- its nodes and its ``edge_node_connectivity`` -- because that is the
computational network results live on. Most files store the node geometry as a branch
index and an along-branch offset rather than explicit x/y; when explicit x/y is absent the
reader reads the branch geometry (the ``network``'s ``edge_geometry``) and derives the
node x/y through :func:`pyramids.netcdf.ugrid.place_nodes_on_network` -- the linear-
referencing primitive lives upstream, not in this driver (ADR-0005). Both topologies are
found by their ``cf_role``
attributes (``mesh_topology`` on the dummy topology variable, ``edge_node_connectivity``
on the edge table), never by a hard-coded name, so the two spelling conventions in the
wild (``network``/``mesh1d`` and ``network1d``/``mesh1d``) read the same way.

It is a hydrodynamic model's native format, so it belongs here (design section 1), read
through the NetCDF substrate the vendored GDAL already provides -- the same
multidimensional route the other NetCDF drivers use, with no declared ``netCDF4`` or
``h5py``. The topology dummy variables are scalar, which GDAL's default listing omits, so
the root group is listed with ``SHOW_ALL=YES`` to surface them.

A Deltares 1D file uses ``.nc`` and is not plain 2D UGRID, so ``read_file`` reaches this
driver through its NetCDF-suffix fallback (``pyramids-core`` 0.69.0).
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np
from osgeo import gdal
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION, place_nodes_on_network

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh.drivers.threedi import _ThreeDiLoader  # reopen-per-read lazy loader

# The cf_role tokens that identify the pieces of a UGRID topology.
_MESH_TOPOLOGY = "mesh_topology"
_FACE_NODE = "face_node_connectivity"

# The standard_name a UGRID node-coordinate carries, used to tell the x and y
# coordinates apart from the branch index and offset in the same node_coordinates list.
_X_STANDARD_NAME = "projection_x_coordinate"
_Y_STANDARD_NAME = "projection_y_coordinate"

# The option that makes GDAL list scalar (dimensionless) arrays too, so the topology
# dummy variables -- which carry the cf_role attributes -- are visible.
_SHOW_ALL = ["SHOW_ALL=YES"]


def _string_attr(array, name: str):
    """Return a GDAL MD-array attribute as a string, or ``None`` when absent.

    Args:
        array: The GDAL multidimensional array.
        name: The attribute name to read.

    Returns:
        str | None: The attribute's value as a string, or ``None``.
    """
    for attribute in array.GetAttributes():
        if attribute.GetName() == name:
            try:
                return str(attribute.Read())
            except RuntimeError:
                return None
    return None


def _attr_map(array) -> dict:
    """Return every attribute of a GDAL MD-array as a ``name -> value`` mapping.

    Args:
        array: The GDAL multidimensional array.

    Returns:
        dict: The array's attributes, each read to its native Python value; an
            attribute that fails to read is omitted.
    """
    attributes: dict = {}
    for attribute in array.GetAttributes():
        try:
            attributes[attribute.GetName()] = attribute.Read()
        except RuntimeError:
            continue
    return attributes


def _fill_value(array):
    """Return a result array's declared no-data value, or ``None``.

    Args:
        array: The GDAL multidimensional array.

    Returns:
        float | None: ``_FillValue`` or ``missing_value`` when declared, else ``None`` --
            the Deltares format declares no sentinel of its own to fall back to.
    """
    attributes = {a.GetName(): a for a in array.GetAttributes()}
    for name in ("_FillValue", "missing_value"):
        attribute = attributes.get(name)
        if attribute is not None:
            try:
                return float(np.asarray(attribute.Read()).ravel()[0])
            except (RuntimeError, ValueError, TypeError, IndexError):
                pass
    return None


class DeltaresNetworkDriver(BaseDriver):
    """Reader for Delft3D FM / D-HYDRO 1D network UGRID NetCDF.

    The driver builds a :class:`~pyramids.netcdf.ugrid.Mesh1d` from the ``mesh1d``
    discretisation: its nodes come from the node x/y coordinates and its edges from the
    ``edge_node_connectivity`` table (re-based to 0 if the file is 1-based, the base kept
    for a writer). When the discretisation stores no explicit node x/y -- the common
    D-HYDRO shape, where nodes are a branch index plus an along-branch offset -- the driver
    reads the network's branch geometry and derives the x/y through
    :func:`pyramids.netcdf.ugrid.place_nodes_on_network` (the linear-referencing primitive
    is upstream, not reimplemented here). The per-node branch index and offset, and every
    ``mesh1d_*`` result variable over ``time``, are parsed as node-located or edge-located
    variables.

    Both topologies are discovered by their ``cf_role`` attributes rather than by name,
    so a file spelling the network ``network`` or ``network1d`` reads identically. The
    discretisation is told from the bare network by its ``coordinate_space`` attribute,
    which only the discretisation carries.

    Documented limitations:
        - **Only the ``mesh1d`` discretisation is built.** The coarser ``network`` of
          branch endpoints and its branch geometry are not returned; results live on the
          discretisation, which is what a :class:`Mesh1d` represents.
        - **A :class:`Mesh1d` carries a topology and a CRS only.** The branch index,
          the offset and the result variables are parsed and validated, but
          :class:`Mesh1d` has nowhere to attach them, so the built network warns that
          they are not in the result (see :class:`~pyramids_mesh._base.BaseDriver`).
          Carrying them needs ``Mesh1d`` to grow a home for node data, which is
          upstream work in ``pyramids-core`` (design section 10).
        - **Edge indices honour the file's ``start_index``.** A 1-based table is re-based
          to 0 and the base recorded for a writer.
        - **A combined 1D2D file is declined by ``can_open``.** A file that also carries a
          2D ``face_node_connectivity`` is left to the UGRID reader; its 1D half can still
          be read by calling :meth:`read` directly.
        - Values are lazy: the loader reopens the file per read, closing over a path.
        - The CRS comes from the file's EPSG code when it declares one, else ``None``.

    Example:
        Read a Deltares 1D map file (shown, not run as a doctest -- it needs a real
        NetCDF file), reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import Mesh1d

            network = Mesh1d.read_file("FlowFM_map.nc")   # mesh1d discretisation
            print(network.n_node, network.n_edge)
    """

    name = "deltares1d"
    long_name = "Deltares 1D network (Delft3D FM / D-HYDRO)"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a Deltares 1D network, by its cf_role topologies.

        Claims a file carrying a one-dimensional ``mesh_topology`` but no 2D
        ``face_node_connectivity``: the 1D topology is what distinguishes it from a plain
        2D UGRID file, and the absence of 2D faces keeps it from stealing one (or a
        combined 1D2D file, left to the UGRID reader). A 3Di or TUFLOW FV ``.nc`` has no
        ``mesh_topology`` variable at all and so is declined. Opening a non-NetCDF file
        raises inside GDAL and is reported as "not Deltares 1D" rather than propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when a 1D topology is present and no 2D faces are.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                topologies = _topology_variables(dataset.GetRootGroup())
            finally:
                dataset.Close()
        has_1d = any(_topology_dimension(attrs) == 1 for attrs in topologies.values())
        has_2d_faces = any(
            _topology_dimension(attrs) == 2 and _FACE_NODE in attrs
            for attrs in topologies.values()
        )
        return has_1d and not has_2d_faces

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Deltares 1D file's ``mesh1d`` discretisation into a :class:`ParsedMesh`.

        Args:
            path: The Deltares 1D ``.nc`` file.
            lazy: When ``True``, leave each result variable behind a loader.

        Returns:
            ParsedMesh: The 1D network, the per-node branch index and offset, and the
                ``mesh1d_*`` result variables.

        Raises:
            FileFormatNotSupportedError: If the file carries no 1D ``mesh_topology``.
        """
        return _DeltaresReader(path, lazy=lazy).build()


def _topology_variables(group) -> dict:
    """Return every ``mesh_topology`` dummy variable in the group, by name.

    The topology variables are scalar, so the group is listed with ``SHOW_ALL=YES`` to
    surface them; each is read for its ``cf_role`` and only the ``mesh_topology`` ones are
    kept.

    Args:
        group: The NetCDF root group.

    Returns:
        dict: ``topology name -> attribute mapping`` for each ``mesh_topology`` variable.
    """
    topologies: dict = {}
    for name in group.GetMDArrayNames(_SHOW_ALL):
        attributes = _attr_map(group.OpenMDArray(name))
        if _as_str(attributes.get("cf_role")) == _MESH_TOPOLOGY:
            topologies[name] = attributes
    return topologies


def _topology_dimension(attributes: dict) -> int:
    """Return a topology's ``topology_dimension`` as an int, or ``0`` when absent."""
    value = attributes.get("topology_dimension")
    if value is None:
        return 0
    try:
        return int(np.asarray(value).ravel()[0])
    except (ValueError, TypeError, IndexError):
        return 0


def _as_str(value):
    """Return a GDAL attribute value as a plain string, or ``None``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


class _DeltaresReader:
    """One Deltares 1D file: build the ``mesh1d`` network and its variables."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Open the file and assemble the 1D network and its variables.

        Returns:
            ParsedMesh: The ``mesh1d`` network with its node and result variables.

        Raises:
            FileFormatNotSupportedError: If the file carries no 1D ``mesh_topology``, or
                if it is openable but malformed -- an attribute naming a UGRID array
                (a ``node_coordinates`` entry, ``edge_geometry``, ``node_count``, a
                connectivity) that the file does not actually contain. GDAL raises
                ``RuntimeError`` for such a dangling reference; it is translated here so a
                file the driver cannot read is declined, never thrown as a raw GDAL error.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                topo_name, topo = self._discretisation(group)
                node_x, node_y = self._node_coordinates(group, topo_name, topo)
                edges, start_index = self._edges(group, topo)
                variables = self._node_attribute_variables(group, topo)
                variables.update(self._result_variables(group, topo_name))
                crs = self._crs(group)
                source = _string_attr(group, "source") or "Deltares 1D network"
            except FileFormatNotSupportedError:
                raise
            except RuntimeError as error:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} is not a readable Deltares 1D network: {error}"
                ) from error
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            edges=edges,
            fill_value=-1,
            start_index=start_index,
            # The file's own base is recorded so a writer re-emits the same numbering.
            original_start_index=start_index,
            crs=crs,
            variables=variables,
            global_attributes={"source": source},
            topology_dimension=1,
            mesh_name=topo_name,
        )

    def _discretisation(self, group) -> tuple[str, dict]:
        """Return the ``mesh1d`` discretisation topology's name and attributes.

        The discretisation is the 1D ``mesh_topology`` carrying a ``coordinate_space``
        attribute -- the one whose nodes are positioned inside a network's coordinate
        space. A file with a single 1D topology and no such marker falls back to that
        one.

        Raises:
            FileFormatNotSupportedError: If no 1D ``mesh_topology`` is present.
        """
        topologies = _topology_variables(group)
        one_d = {
            name: attrs
            for name, attrs in topologies.items()
            if _topology_dimension(attrs) == 1
        }
        if not one_d:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a Deltares 1D network: no 1D mesh_topology"
            )
        for name, attrs in one_d.items():
            if attrs.get("coordinate_space") is not None:
                return name, attrs
        name = next(iter(one_d))
        return name, one_d[name]

    def _node_coordinates(
        self, group, topo_name: str, topo: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the discretisation's node x and y coordinates.

        A ``mesh1d`` discretisation positions its nodes in one of two ways, and modern
        Delft3D FM / D-HYDRO files use the second far more often than the first:

        1. **Explicit x/y.** The ``node_coordinates`` attribute lists an x and a y array,
           told from the branch index and offset in the same list by their
           ``projection_x_coordinate`` / ``projection_y_coordinate`` standard name.
        2. **Branch + offset.** The ``node_coordinates`` attribute lists only a branch
           index and an along-branch offset; the x/y must be reconstructed by linear
           referencing each node's offset along its branch's polyline geometry, which the
           coordinate-space ``network`` carries (``edge_geometry``).

        Three sources are tried in order of fidelity: the x/y named by the topology
        (case 1); failing that, explicit ``<mesh1d>_node_x`` / ``_node_y`` arrays that are
        present in the file but not referenced by ``node_coordinates`` (a common D-HYDRO
        shape); failing that, reconstruction from the branch index and offset (case 2).

        Args:
            group: The NetCDF root group.
            topo_name: The discretisation topology's name, used to find its sibling
                node-coordinate arrays.
            topo: The discretisation topology's attribute mapping.

        Returns:
            tuple[np.ndarray, np.ndarray]: The node x and y coordinates.

        Raises:
            FileFormatNotSupportedError: If no x/y is named, none is present as sibling
                arrays, and none can be reconstructed from the branch geometry.
        """
        names = _split(topo.get("node_coordinates"))
        node_x, node_y = _xy_by_standard_name(group, names)
        if node_x is not None and node_y is not None:
            return node_x, node_y
        node_x, node_y = self._xy_from_sibling_arrays(group, topo_name)
        if node_x is not None and node_y is not None:
            return node_x, node_y
        reconstructed = self._reconstruct_node_coordinates(group, topo, names)
        if reconstructed is not None:
            return reconstructed
        raise FileFormatNotSupportedError(
            f"{self.path!r} mesh1d topology has no x/y node coordinates in {names!r}, "
            f"no sibling {topo_name}_node_x/_node_y arrays, and none could be "
            f"reconstructed from the network branch geometry"
        )

    def _xy_from_sibling_arrays(self, group, topo_name: str):
        """Return explicit node x/y arrays present in the file but not named by the topology.

        D-HYDRO often writes ``<mesh1d>_node_x`` / ``_node_y`` alongside a
        ``node_coordinates`` attribute that references only the branch index and offset.
        When present, those arrays are the authoritative node geometry and are preferred
        over reconstruction. Only node arrays belonging to the discretisation are
        considered -- an array is matched by the ``projection_x`` / ``projection_y``
        standard name, a name beginning with the topology's own name and an underscore
        (so a sibling topology such as ``mesh1d2d`` is not matched), and a ``node`` role
        in the name (so an edge or face coordinate is never mistaken for a node one).

        Args:
            group: The NetCDF root group.
            topo_name: The discretisation topology's name (the array-name prefix).

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: The x and y arrays, or
                ``(None, None)`` when either is absent.
        """
        node_x = node_y = None
        for name in group.GetMDArrayNames(_SHOW_ALL):
            if not name.startswith(f"{topo_name}_") or "node" not in name:
                continue
            array = group.OpenMDArray(name)
            standard_name = _string_attr(array, "standard_name")
            if standard_name == _X_STANDARD_NAME and node_x is None:
                node_x = np.asarray(array.ReadAsArray(), np.float64)
            elif standard_name == _Y_STANDARD_NAME and node_y is None:
                node_y = np.asarray(array.ReadAsArray(), np.float64)
        if node_x is None or node_y is None:
            return None, None
        return node_x, node_y

    def _reconstruct_node_coordinates(self, group, topo: dict, names: list):
        """Reconstruct node x/y by linear referencing along the network branch geometry.

        Each node is given as a branch index (``mesh1d_node_branch`` /
        ``mesh1d_node_branch_id``) and an along-branch distance
        (``mesh1d_node_offset`` / ``mesh1d_node_branch_offset``). The branch index is
        re-based to 0 with the array's ``start_index``; the offset is a chainage along the
        branch. The coordinate-space ``network`` named by the topology's
        ``coordinate_space`` carries the branch polylines through its ``edge_geometry`` and
        a per-branch declared length (``edge_length``). This method only reads those raw
        arrays; the node x/y is derived by
        :func:`~pyramids.netcdf.ugrid.place_nodes_on_network`, the linear-referencing
        primitive -- the coordinate derivation is owned by pyramids, not duplicated here
        (ADR-0005).

        Args:
            group: The NetCDF root group.
            topo: The discretisation topology's attribute mapping.
            names: The topology's ``node_coordinates`` entries (branch and offset arrays).

        Returns:
            tuple[np.ndarray, np.ndarray] | None: The reconstructed node x and y, or
                ``None`` when the file carries no coordinate space, no branch/offset pair,
                or no usable branch geometry (including a dangling array reference).

        Raises:
            FileFormatNotSupportedError: The branch geometry is present but cannot be
                resolved -- a non-numeric or mis-shaped branch/offset/geometry array, or
                :func:`~pyramids.netcdf.ugrid.place_nodes_on_network` rejecting it
                (mismatched lengths, an out-of-range or fractional branch index, a
                non-tiling vertex count, a non-finite value, ...). The underlying cause is
                chained so the real reason is not lost, and no raw ``ValueError`` /
                ``TypeError`` escapes the driver's decline contract.
        """
        coordinate_space = _as_str(topo.get("coordinate_space"))
        if coordinate_space is None:
            return None
        try:
            branch_name, offset_name = _branch_offset_names(group, names)
            if branch_name is None or offset_name is None:
                return None
            branch_array = group.OpenMDArray(branch_name)
            # No int cast here: the branch index is handed to place_nodes_on_network as
            # read (minus the file's start_index), so its _coerce_branch_index_to_int64
            # guard can reject a fractional/non-finite index rather than the driver
            # silently truncating it to a wrong branch.
            branch = np.asarray(branch_array.ReadAsArray()) - self._start_index(
                branch_array
            )
            offset = np.asarray(
                group.OpenMDArray(offset_name).ReadAsArray(), np.float64
            )
            geometry = _network_branch_geometry(group, coordinate_space)
            if geometry is None:
                return None
            geom_x, geom_y, vertices_per_branch, edge_length = geometry
            # The derivation is owned by pyramids (ADR-0005); the driver only reads the raw
            # arrays and hands them to the linear-referencing primitive.
            return place_nodes_on_network(
                branch, offset, geom_x, geom_y, vertices_per_branch, edge_length
            )
        except RuntimeError:
            # A dangling reference (an array named but absent) makes the file unreadable;
            # decline with the generic message so build() reports it.
            return None
        except (ValueError, TypeError) as error:
            # Malformed branch+offset data: a non-numeric / mis-shaped coordinate array, or
            # place_nodes_on_network rejecting the geometry. Decline with the cause chained
            # rather than leaking a raw error, honouring the driver's decline contract.
            # (place_nodes_on_network documents a ValueError-only contract for bad input;
            # TypeError guards a non-numeric array that fails arithmetic before the
            # primitive sees it. A future non-ValueError from the primitive would fall to
            # the RuntimeError clause or escape, so this coupling is deliberate.)
            raise FileFormatNotSupportedError(
                f"{self.path!r} has branch+offset mesh1d geometry that could not be "
                f"resolved to coordinates: {error}"
            ) from error

    def _edges(self, group, topo: dict) -> tuple[np.ndarray, int]:
        """Return the discretisation's edge-node connectivity and its start index.

        Raises:
            FileFormatNotSupportedError: If the topology names no edge connectivity.
        """
        connectivity = _as_str(topo.get("edge_node_connectivity"))
        if connectivity is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} mesh1d topology declares no edge_node_connectivity"
            )
        array = group.OpenMDArray(connectivity)
        edges = np.asarray(array.ReadAsArray(), np.int64)
        start_index = self._start_index(array)
        return edges, start_index

    @staticmethod
    def _start_index(array) -> int:
        """Return a connectivity array's ``start_index``, defaulting to ``0``."""
        value = _attr_map(array).get("start_index")
        if value is None:
            return 0
        try:
            return int(np.asarray(value).ravel()[0])
        except (ValueError, TypeError, IndexError):
            return 0

    def _node_attribute_variables(self, group, topo: dict) -> dict:
        """Carry the per-node branch index and offset as eager node variables.

        Every ``node_coordinates`` entry that is not the x or y geometry -- the branch
        index and the along-branch offset -- becomes a node-located variable, so the
        position of each computational node within its branch is preserved. They are
        small and static, so they are read eagerly rather than deferred.

        Args:
            group: The NetCDF root group.
            topo: The discretisation topology's attribute mapping.

        Returns:
            dict: One eager node :class:`ParsedVariable` per non-geometry node
                coordinate, keyed by its array name.
        """
        variables: dict = {}
        for name in _split(topo.get("node_coordinates")):
            array = group.OpenMDArray(name)
            standard_name = _string_attr(array, "standard_name")
            if standard_name in (_X_STANDARD_NAME, _Y_STANDARD_NAME):
                continue
            values = np.asarray(array.ReadAsArray())
            variables[name] = ParsedVariable(
                location="node",
                shape=values.shape,
                values=values,
                standard_name=standard_name,
                units=_string_attr(array, "units"),
            )
        return variables

    def _result_variables(self, group, topo_name: str) -> dict:
        """Build a variable for each ``mesh1d_*`` result on the discretisation.

        A result array declares which topology it belongs to with a ``mesh`` attribute
        and which location with a ``location`` attribute; an array whose ``mesh`` is the
        discretisation and whose ``location`` is ``node`` or ``edge`` becomes a variable.
        A temporal result carries the ``time`` coordinate; a static one (bed level, cell
        area) carries none.

        Args:
            group: The NetCDF root group.
            topo_name: The discretisation topology's name, matched against ``mesh``.

        Returns:
            dict: One :class:`ParsedVariable` per result array, keyed by its name.
        """
        variables: dict = {}
        times_cache: dict = {}
        for name in sorted(group.GetMDArrayNames()):
            array = group.OpenMDArray(name)
            attributes = _attr_map(array)
            if _as_str(attributes.get("mesh")) != topo_name:
                continue
            location = _as_str(attributes.get("location"))
            if location not in ("node", "edge"):
                continue
            dimensions = [d.GetName() for d in array.GetDimensions()]
            shape = tuple(d.GetSize() for d in array.GetDimensions())
            times = self._times(group, dimensions, times_cache)
            loader = _ThreeDiLoader(self.path, name)
            variables[name] = ParsedVariable(
                location=location,
                shape=shape,
                loader=loader if self.lazy else None,
                values=None if self.lazy else loader(),
                units=_string_attr(array, "units"),
                standard_name=_string_attr(array, "standard_name"),
                nodata=_fill_value(array),
                time_values=times,
            )
        return variables

    def _times(self, group, dimensions: list, cache: dict):
        """Return the time coordinate when a result's first dimension is the time axis.

        A result's leading dimension names the time axis, and the coordinate array shares
        that name; it is read once and cached, so a file of many results reads ``time``
        only once.

        Args:
            group: The NetCDF root group.
            dimensions: The result array's dimension names, outermost first.
            cache: A per-build cache mapping a time-dimension name to its coordinate.

        Returns:
            np.ndarray | None: The time coordinate, or ``None`` for a static result.
        """
        if not dimensions:
            return None
        time_dim = dimensions[0]
        if time_dim not in cache:
            cache[time_dim] = self._read_time(group, time_dim)
        return cache[time_dim]

    @staticmethod
    def _read_time(group, time_dim: str):
        """Read the coordinate array named after the time dimension, or ``None``.

        A result's first dimension is only a time axis when a coordinate variable of the
        same name exists and is itself one-dimensional over that axis; anything else
        (a spatial-only array, a missing coordinate) returns ``None``.
        """
        if time_dim not in set(group.GetMDArrayNames(_SHOW_ALL)):
            return None
        array = group.OpenMDArray(time_dim)
        dimensions = [d.GetName() for d in array.GetDimensions()]
        if dimensions != [time_dim]:
            return None
        return np.asarray(array.ReadAsArray(), np.float64)

    def _crs(self, group):
        """Return the file's EPSG code as an int, or ``None`` when none is declared.

        Delft3D FM records its CRS on a grid-mapping variable carrying an ``epsg``
        attribute; the first such code found is used. No code means no CRS rather than a
        guessed one.

        Args:
            group: The NetCDF root group.

        Returns:
            int | None: The EPSG code, or ``None``.
        """
        for name in group.GetMDArrayNames(_SHOW_ALL):
            attributes = _attr_map(group.OpenMDArray(name))
            if "epsg" not in attributes:
                continue
            try:
                code = int(np.asarray(attributes["epsg"]).ravel()[0])
            except (ValueError, TypeError, IndexError):
                continue
            if code > 0:
                return code
        return None


def _xy_by_standard_name(group, names: list):
    """Return the x and y arrays among ``names``, picked by their standard name.

    The ``node_coordinates`` list mixes geometry with the branch index and offset; the x
    and y are the entries carrying the ``projection_x_coordinate`` /
    ``projection_y_coordinate`` standard name.

    Args:
        group: The NetCDF root group.
        names: The candidate coordinate-array names.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: The x and y arrays, or ``None`` for
            either that is not named.
    """
    node_x = node_y = None
    for name in names:
        standard_name = _string_attr(group.OpenMDArray(name), "standard_name")
        if standard_name == _X_STANDARD_NAME:
            node_x = np.asarray(group.OpenMDArray(name).ReadAsArray(), np.float64)
        elif standard_name == _Y_STANDARD_NAME:
            node_y = np.asarray(group.OpenMDArray(name).ReadAsArray(), np.float64)
    return node_x, node_y


def _branch_offset_names(group, names: list):
    """Return the branch-index and along-branch-offset array names among ``names``.

    A ``mesh1d`` discretisation names its branch index and offset under either spelling
    (``mesh1d_node_branch`` / ``mesh1d_node_offset`` or ``mesh1d_node_branch_id`` /
    ``mesh1d_node_branch_offset``). They are matched by name first -- the entry whose name
    contains ``offset`` is the offset (tested before ``branch`` so the alternate spelling's
    ``..._branch_offset`` is not mistaken for the branch index), the remaining entry whose
    name contains ``branch`` is the branch index. Any entry left unresolved falls back to
    dtype: an integer entry is the branch index, a floating-point entry the offset. The
    name test avoids feeding an unstandardised x/y array (a float that tier 0 missed) into
    reconstruction as if it were a distance.

    Args:
        group: The NetCDF root group.
        names: The topology's ``node_coordinates`` entries.

    Returns:
        tuple[str | None, str | None]: The branch-index and offset array names, each
            ``None`` when neither the name nor the dtype identifies one.
    """
    branch_name = offset_name = None
    for name in names:
        lowered = name.lower()
        if offset_name is None and "offset" in lowered:
            offset_name = name
        elif branch_name is None and "branch" in lowered:
            branch_name = name
    if branch_name is not None and offset_name is not None:
        # Both resolved by name; skip the dtype loop so an unrelated extra entry (a
        # dangling reference among them included) is never opened just to be discarded.
        return branch_name, offset_name
    for name in names:
        if name in (branch_name, offset_name):
            continue
        values = np.asarray(group.OpenMDArray(name).ReadAsArray())
        if branch_name is None and np.issubdtype(values.dtype, np.integer):
            branch_name = name
        elif offset_name is None and np.issubdtype(values.dtype, np.floating):
            offset_name = name
    return branch_name, offset_name


def _network_branch_geometry(group, network_name: str):
    """Return a network's branch polylines: the vertex x, vertex y, and per-branch count.

    The coordinate-space ``network`` topology carries its branch polylines through an
    ``edge_geometry`` container, whose ``node_coordinates`` name the vertex x and y and
    whose ``node_count`` gives the number of vertices on each branch (the branches
    concatenated into one flat vertex list). The vertex x and y are picked by standard
    name, falling back to the first two entries in declared order.

    Args:
        group: The NetCDF root group.
        network_name: The coordinate-space network topology's name.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None: The flat
            vertex x, flat vertex y, per-branch vertex count, and per-branch declared
            length (``edge_length``, or ``None`` when the network declares none), or
            ``None`` when the network declares no usable branch geometry.
    """
    network = _topology_variables(group).get(network_name)
    if network is None:
        return None
    geometry_name = _as_str(network.get("edge_geometry"))
    if geometry_name is None:
        return None
    geometry = _attr_map(group.OpenMDArray(geometry_name))
    coordinate_names = _split(geometry.get("node_coordinates"))
    count_name = _as_str(geometry.get("node_count"))
    if len(coordinate_names) < 2 or count_name is None:
        return None
    vertex_xy = _geometry_vertex_xy(group, coordinate_names)
    if vertex_xy is None:
        return None
    geom_x, geom_y = vertex_xy
    vertices_per_branch = np.asarray(
        group.OpenMDArray(count_name).ReadAsArray()
    ).astype(np.int64, copy=False)
    edge_length = _branch_lengths(group, network, vertices_per_branch.shape[0])
    return geom_x, geom_y, vertices_per_branch, edge_length


def _geometry_vertex_xy(group, coordinate_names: list):
    """Return the geometry's flat vertex x and y arrays, or ``None`` when ambiguous.

    The vertex x and y are picked by their ``projection_x`` / ``projection_y`` standard
    name. When neither carries one they are taken in declared x, y order; when exactly one
    does the match is ambiguous (declared order may disagree with the named axis), so
    ``None`` is returned rather than risk swapping x and y.

    Args:
        group: The NetCDF root group.
        coordinate_names: The geometry container's ``node_coordinates`` entries.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: The vertex x and y, or ``None`` on a partial
            standard-name match.
    """
    geom_x, geom_y = _xy_by_standard_name(group, coordinate_names)
    if geom_x is not None and geom_y is not None:
        return geom_x, geom_y
    if geom_x is not None or geom_y is not None:
        return None
    geom_x = np.asarray(
        group.OpenMDArray(coordinate_names[0]).ReadAsArray(), np.float64
    )
    geom_y = np.asarray(
        group.OpenMDArray(coordinate_names[1]).ReadAsArray(), np.float64
    )
    return geom_x, geom_y


def _branch_lengths(group, network: dict, branch_count: int):
    """Return a network's per-branch declared length (``edge_length``), or ``None``.

    The ``edge_length`` attribute of the network topology names the array of branch
    lengths used to interpret chainage offsets. It is used only when it is present and has
    one length per branch; anything else falls back to chord-length measurement.

    Args:
        group: The NetCDF root group.
        network: The network topology's attribute mapping.
        branch_count: The number of branches the geometry describes.

    Returns:
        np.ndarray | None: The per-branch length, or ``None`` when it is absent or does
            not have one entry per branch.
    """
    length_name = _as_str(network.get("edge_length"))
    if length_name is None:
        return None
    edge_length = np.asarray(group.OpenMDArray(length_name).ReadAsArray(), np.float64)
    if edge_length.shape != (branch_count,):
        return None
    return edge_length


def _split(value) -> list:
    """Split a UGRID space-separated name list attribute into its names.

    Args:
        value: The attribute value -- a single space-separated string, a list already
            split by GDAL, or ``None``.

    Returns:
        list: The individual names, or an empty list when the attribute is absent.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        names: list = []
        for item in value:
            names.extend(str(item).split())
        return names
    return str(value).split()
