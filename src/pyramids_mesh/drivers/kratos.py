"""Kratos Multiphysics ``.mdpa`` reader.

An ``.mdpa`` is Kratos' flat, block-keyed ASCII model part: ``//`` comments (whole-line
or inline -- GiD glues an inline ``// GUI group ...`` onto the element-type token) and
paired ``Begin <Block> ... End <Block>`` sections. ``Begin Nodes`` holds
``<id> <x> <y> <z>`` rows; ``Begin Elements <ElementType>`` holds
``<elem_id> <prop_id> <n1>..<nN>`` rows. The element type usually names the node count
(``Element3D4N`` a four-node tetrahedron, ``Element2D3N`` a three-node surface triangle);
when it does not (``VMS3D``, ``MonolithicPFEM22D``) the count is taken from the width of
the first row. ``Begin Conditions`` is read only when the file has no ``Begin Elements``
with connectivity -- a condition-only mesh (IGA, some shape-optimisation cases) builds
from its conditions; ``Begin Properties`` and ``Begin SubModelPart`` are always read past.

Each element (or, in the fallback, each condition) is mapped straight to a fill-padded
face row -- following the Exodus precedent (``exodus.py``) -- rather than expanded into
bounding polygons: boundary extraction is pyramids' job, not a driver's (ADR-0005). When
the file carries 2D surface elements (``Element2D3N``/``Element2D4N``) those become the
faces; otherwise the 3D volume elements are mapped verbatim, a tetrahedron's four node ids
read as a four-node face. Point elements (one node per row -- DEM particles) are refused:
a point cloud is not a mesh. Node ids are 1-based and resolved through
:func:`~pyramids_mesh._text.native_id_map`. The coordinates carry no datum, so ``crs`` is
``None``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import re

import numpy as np
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records, native_id_map

# Kratos comments open with `//`, C++-style, not `#`.
_COMMENT = "//"

_BEGIN = "BEGIN"
_END = "END"
_NODES = "NODES"
_ELEMENTS = "ELEMENTS"
_CONDITIONS = "CONDITIONS"

# An element type name usually ends in the node count followed by `N`: Element3D4N is
# four nodes, Element2D3N three. Names that do not (VMS3D, MonolithicPFEM22D) take the
# count from the first row instead. The dimension tag (`2D`/`3D`) decides which elements
# become faces.
_NODE_COUNT_RE = re.compile(r"(\d+)N$")
_TWO_DIMENSIONAL = "2D"
# A mesh element needs at least an edge; one node per row is a point cloud, not a mesh.
_MIN_NODES_PER_ELEMENT = 2


class KratosDriver(BaseDriver):
    """Reader for the Kratos Multiphysics model part (``.mdpa``).

    ``Begin Nodes`` rows become ``node_x``/``node_y`` and, when any is non-zero,
    ``node_z``; the elements become the fill-padded ``face_node_connectivity``, one face
    row per element with its leading element and property ids dropped. A file mixing
    element widths pads the short rows to the widest with ``-1``.

    Documented limitations:
        - **2D surface elements are preferred, else 3D volume elements become faces.**
          When the file declares any ``Element2D*`` type those build the mesh; otherwise
          the 3D elements are mapped verbatim -- a ``Element3D4N`` tetrahedron's four node
          ids read as a four-node face (the Exodus precedent), not expanded into its
          bounding triangles. Winding is judged in the ``xy`` plane like any other face.
        - **The element type usually names the node count, else the row width gives it.**
          ``Element2D3N`` is three nodes; a name without a trailing ``<count>N``
          (``VMS3D``, ``MonolithicPFEM22D``) takes its node count from the first row. One
          node per element (DEM ``SphericParticle3D``) is a point cloud, not a mesh, and
          is refused.
        - **Node ids are resolved through an id map.** Kratos numbers from 1 and may gap
          or reorder ids, so connectivity is resolved by position; the ``1`` base is
          recorded for a writer.
        - **Elements build the mesh; Conditions are a fallback.** A file with no
          ``Begin Elements`` connectivity builds from its ``Begin Conditions`` instead
          (IGA / shape-optimisation surface meshes). ``Properties``, ``ModelPartData``
          and ``SubModelPart`` blocks are read past.
        - The format carries no datum, so ``crs`` is ``None``.

    Example:
        Read a Kratos model part (shown, not run as a doctest -- it reads a committed
        data file), reached through ``read_file``::

            from pyramids.netcdf.ugrid import MeshDataset

            mesh = MeshDataset.read_file("model.mdpa")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "kratos"
    long_name = "Kratos Multiphysics model part"
    suffixes = (".mdpa",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file carries ``Begin Nodes`` and Elements or Conditions.

        The file is decoded leniently -- so a binary or empty file answers ``False``
        rather than raising -- and scanned (short-circuiting once both are seen) for a
        ``Begin Nodes`` block and a ``Begin Elements`` *or* ``Begin Conditions`` block.
        Requiring a nodes block plus some connectivity block keeps a FLAC3D ``.f3grid``
        or an unrelated text file from qualifying, while still accepting a condition-only
        mesh (IGA); the whole file is read because the ``Begin Nodes`` block precedes the
        connectivity, so a large node block must not push the opener past a fixed window.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when a nodes block and a connectivity block are both found.
        """
        with open(path, "rb") as handle:
            prefix = handle.read()
        has_nodes = False
        has_connectivity = False
        for line in prefix.decode("utf-8-sig", errors="replace").splitlines():
            tokens = line.strip().split()
            if len(tokens) < 2 or tokens[0].upper() != _BEGIN:
                continue
            block = tokens[1].upper()
            if block == _NODES:
                has_nodes = True
            elif block in (_ELEMENTS, _CONDITIONS):
                has_connectivity = True
            if has_nodes and has_connectivity:
                return True
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Kratos ``.mdpa`` into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The ``.mdpa`` file to read.
            lazy: Ignored -- an ``.mdpa`` is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, the fill-padded face connectivity built from the preferred
                element set (or the conditions, when no elements carry connectivity), and
                the element type recorded as metadata.

        Raises:
            FileFormatNotSupportedError: If the file declares no nodes, or no elements or
                conditions with connectivity; if a block declares fewer than two nodes per
                element (point elements); if an element row is short; or if a numeric field
                is not a number.
        """
        scan = _Scan(path)
        scan.run()
        return scan.build()


class _ElementGroup:
    """One ``Begin Elements``/``Begin Conditions <type>`` block's rows.

    ``node_count`` is ``None`` when the type name carries no ``<count>N`` suffix; it is
    then filled from the width of the first row the block holds.
    """

    def __init__(
        self, element_type: str, node_count: int | None, two_dimensional: bool
    ) -> None:
        self.element_type = element_type
        self.node_count = node_count
        self.two_dimensional = two_dimensional
        self.vertices: list[list[int]] = []


class _Scan:
    """One ``.mdpa`` file, streamed once into nodes and element groups by block."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.groups: list[_ElementGroup] = []
        self.cond_groups: list[_ElementGroup] = []
        # The block currently open: `_NODES`, `_ELEMENTS`, `_CONDITIONS`, or None/other
        # (read past).
        self._block: str | None = None
        self._group: _ElementGroup | None = None

    def run(self) -> None:
        """Stream the file, routing each row by the block it sits in.

        Raises:
            FileFormatNotSupportedError: On a short row or a non-numeric field.
        """
        for keyword, fields in iter_records(self.path, comment=_COMMENT):
            upper = keyword.upper()
            if upper == _BEGIN:
                self._open_block(fields)
            elif upper == _END:
                self._block = None
                self._group = None
            elif self._block == _NODES:
                self._add_node(keyword, fields)
            elif self._block in (_ELEMENTS, _CONDITIONS) and self._group is not None:
                self._add_element(keyword, fields)

    def _open_block(self, fields: list[str]) -> None:
        """Begin a block, tracking ``Nodes``, ``Elements`` and ``Conditions``.

        ``Elements`` build the mesh; ``Conditions`` are collected separately and used only
        as a fallback (see :meth:`build`). ``Properties``/``SubModelPart`` are read past.
        """
        block = fields[0].upper() if fields else ""
        if block == _NODES:
            self._block = _NODES
            self._group = None
        elif block in (_ELEMENTS, _CONDITIONS):
            self._block = block
            element_type = fields[1] if len(fields) > 1 else ""
            self._group = self._start_group(element_type)
            target = self.groups if block == _ELEMENTS else self.cond_groups
            target.append(self._group)
        else:
            # A Properties/SubModelPart block (possibly nested): read past it until its
            # End. A nested opener resets the tracker the same way.
            self._block = None
            self._group = None

    def _start_group(self, element_type: str) -> _ElementGroup:
        """Build the group for an ``Elements``/``Conditions`` block from its type name.

        The node count is read from the name's trailing ``<count>N`` when present
        (``Element2D3N`` -> 3); a name without one (``VMS3D``) leaves it ``None``, to be
        filled from the first row's width in :meth:`_add_element`.
        """
        match = _NODE_COUNT_RE.search(element_type)
        node_count = int(match.group(1)) if match is not None else None
        return _ElementGroup(
            element_type,
            node_count,
            _TWO_DIMENSIONAL in element_type.upper(),
        )

    def _add_node(self, keyword: str, fields: list[str]) -> None:
        """Collect one ``<id> <x> <y> <z>`` node row inside a ``Begin Nodes`` block.

        Raises:
            FileFormatNotSupportedError: If the row is short of an id and three
                coordinates, or a field is not numeric.
        """
        tokens = [keyword, *fields]
        if len(tokens) < 4:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short Nodes row: expected an id and three "
                f"coordinates, found {len(tokens)} field(s)"
            )
        self.node_ids.append(_as_int(self.path, _NODES, tokens[0]))
        self.node_x.append(_as_float(self.path, _NODES, tokens[1]))
        self.node_y.append(_as_float(self.path, _NODES, tokens[2]))
        self.node_z.append(_as_float(self.path, _NODES, tokens[3]))

    def _add_element(self, keyword: str, fields: list[str]) -> None:
        """Collect one element (or condition) row, dropping the leading two ids.

        A row is ``<elem_id> <prop_id> <n1>..<nN>``; the two leading ids are dropped and
        the next ``node_count`` tokens are the connectivity. When the block's type name
        carried no ``<count>N`` the count is unknown until here -- it is taken from this
        row's width and held for the rest of the block.

        Point blocks (one node per row) are recorded, not rejected here: a file may hold
        1-node point-load conditions alongside the real elements that build the mesh, so
        whether a point block matters is decided at selection time (see :meth:`build`).

        Raises:
            FileFormatNotSupportedError: If the row is short of its node ids, or a node id
                is not an integer.
        """
        assert self._group is not None  # guarded by the caller
        tokens = [keyword, *fields]
        count = self._group.node_count
        if count is None:
            # No <count>N in the type name (VMS3D, MonolithicPFEM22D): the row width is
            # the only source. `<id> <prop> <n1>..<nN>` -> count is the rest.
            count = len(tokens) - 2
            self._group.node_count = count
        if len(tokens) < 2 + count:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short {self._group.element_type} row: expected an "
                f"element id, a property id and {count} node ids, found {len(tokens)}"
            )
        self._group.vertices.append(
            [_as_int(self.path, self._block, tok) for tok in tokens[2 : 2 + count]]
        )

    def build(self) -> ParsedMesh:
        """Assemble the parsed mesh from the nodes and the preferred element set.

        Returns:
            ParsedMesh: The mesh, 0-based after id resolution with the ``1`` base
                recorded for a writer, and the element type recorded as metadata.

        Raises:
            FileFormatNotSupportedError: If the file declares no nodes, or no elements or
                conditions with connectivity.
            ValueError: If an element references a node id the file never defined, or two
                nodes share an id.
        """
        if not self.node_ids:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no Nodes, so it is not a readable Kratos mesh"
            )
        # Elements build the mesh; fall back to Conditions only when no element block
        # carries connectivity (a condition-only mesh: IGA, some shape optimisation).
        chosen = self._chosen_groups(self.groups)
        from_conditions = not chosen
        if from_conditions:
            chosen = self._chosen_groups(self.cond_groups)
        if not chosen:
            if self._has_only_point_groups():
                raise FileFormatNotSupportedError(
                    f"{self.path!r} declares only point elements (one node per element -- "
                    f"DEM particles or point loads); a point cloud is not a mesh"
                )
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no Elements or Conditions with connectivity, so "
                f"there is nothing to build a mesh from"
            )
        node_map = native_id_map(self.node_ids)
        faces = self._resolve_faces(chosen, node_map)
        node_z = np.array(self.node_z, dtype=np.float64)
        types = sorted({group.element_type for group in chosen})
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Node ids were resolved through the id map, so the connectivity is already
            # 0-based; Kratos numbers from 1, recorded for a writer to restore.
            start_index=0,
            original_start_index=1,
            node_z=node_z if np.any(node_z != 0.0) else None,
            crs=None,
            global_attributes={
                "source": "Kratos",
                "element_types": types,
                "geometry_from": "Conditions" if from_conditions else "Elements",
            },
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _chosen_groups(self, groups: list[_ElementGroup]) -> list[_ElementGroup]:
        """Return the groups that build the mesh: 2D surface ones, else the rest.

        Only groups with at least :data:`_MIN_NODES_PER_ELEMENT` nodes per element are
        mesh-building candidates; 1-node point blocks (DEM particles, point-load
        conditions) are excluded so they neither build nor shadow a real mesh. Among the
        candidates, any 2D surface groups win -- a file carrying any 2D surface element
        uses only those for the faces; a purely volumetric file maps its 3D elements
        verbatim. Used for the element groups and, as a fallback, the condition groups.
        """
        with_rows = [
            group
            for group in groups
            if group.vertices and (group.node_count or 0) >= _MIN_NODES_PER_ELEMENT
        ]
        two_dimensional = [group for group in with_rows if group.two_dimensional]
        return two_dimensional or with_rows

    def _has_only_point_groups(self) -> bool:
        """Whether every block that held rows was a 1-node point block.

        Distinguishes a DEM particle / point-load file (a point cloud) from one with no
        connectivity at all, so :meth:`build` can name the real reason it is not a mesh.
        """
        groups_with_rows = [
            group for group in (*self.groups, *self.cond_groups) if group.vertices
        ]
        return bool(groups_with_rows) and all(
            (group.node_count or 0) < _MIN_NODES_PER_ELEMENT
            for group in groups_with_rows
        )

    def _resolve_faces(
        self, groups: list[_ElementGroup], node_map: dict[int, int]
    ) -> np.ndarray:
        """Map each chosen element's node ids to 0-based indices, padding short rows.

        Args:
            groups: The element groups that build the mesh.
            node_map: The node-id -> 0-based-index mapping.

        Returns:
            np.ndarray: The ``(n_element, max_nodes)`` fill-padded face connectivity.

        Raises:
            ValueError: If an element references a node id the file never defined.
        """
        rows = [vertices for group in groups for vertices in group.vertices]
        width = max(len(vertices) for vertices in rows)
        faces = np.full((len(rows), width), -1, dtype=np.int64)
        for row, vertices in enumerate(rows):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, self.path)
        return faces


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Return a node id's 0-based index, or raise naming the undefined id.

    Args:
        native: The node id an element referenced.
        node_map: The node-id -> 0-based-index mapping.
        path: The source path, for the error message.

    Returns:
        int: The 0-based node index.

    Raises:
        ValueError: If the id was never defined by a ``Begin Nodes`` row.
    """
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} has an element referencing node id {native}, which no Nodes row "
            f"defines"
        ) from None


def _as_int(path: str, block: str, value: str) -> int:
    """Parse one field as an integer, naming the block and file if it is not one.

    Args:
        path: The source path, for the error message.
        block: The block being read, for the error message.
        value: The field text.

    Returns:
        int: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not an integer.
    """
    try:
        return int(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {block} field: expected an integer, found "
            f"{value!r}"
        ) from error


def _as_float(path: str, block: str, value: str) -> float:
    """Parse one field as a float, naming the block and file if it is not one.

    Args:
        path: The source path, for the error message.
        block: The block being read, for the error message.
        value: The field text.

    Returns:
        float: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not a number.
    """
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {block} field: expected a number, found "
            f"{value!r}"
        ) from error
