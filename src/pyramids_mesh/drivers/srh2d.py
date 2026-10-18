"""SRH-2D reader (task C12).

SRH-2D (the USBR/Aquaveo two-dimensional surface-water model) stores a mesh as an ASCII
trio sharing one basename stem:

- ``<case>.srhgeom`` -- the mesh itself. ``Elem <id> n1 n2 n3[ n4 ...]`` lines carry
  mixed triangle/quad/polygon faces by 1-based node id, and ``Node <id> x y z`` lines the
  node coordinates and bed elevation. ``NodeString`` boundary lists and the ``Name`` /
  ``GridUnit`` header cards are read for metadata only.
- ``<case>.srhhydro`` -- the control deck: ``Grid "<case>.srhgeom"``,
  ``HydroMat "<case>.srhmat"``, ``ManningsN <material-id> <n>``, ``BC`` and ``SimTime``.
- ``<case>.srhmat`` -- the element-to-material-zone map, one ``Material <id> <elem ...>``
  block per zone (element ids continue onto the following unprefixed lines).

This is the geometry/control reader: ``read`` builds the mesh from ``.srhgeom`` and
carries the material-zone id and Manning's n per element as face variables when the
``.srhmat`` / ``.srhhydro`` siblings describe them cleanly. SRH-2D results are written as
XMDF or VTK, which other drivers in this package already read, so no results are read here.

See ``planning/design.md`` section 20.1. Node and element ids are resolved through
:func:`~pyramids_mesh._text.native_id_map`, so a gapped or renumbered file stays wired
correctly rather than assuming ``id == index + 1``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import re
import warnings

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._text import iter_records, native_id_map

# The first significant token of each member file; the sniff keys on these.
_GEOM_HEADER = "SRHGEOM"
_HYDRO_HEADER = "SRHHYDRO"

# Record keywords. ``Node`` and ``NodeString`` share a prefix, so the keyword is matched
# whole -- a ``NodeString`` boundary list is not a node.
_ELEM = "Elem"
_NODE = "Node"
_GRID_UNIT = "GridUnit"

# ``.srhhydro`` cards consulted for metadata.
_GRID = "Grid"
_HYDRO_MAT = "HydroMat"
_MANNINGS = "ManningsN"

# ``.srhmat`` cards. A ``Material`` block's element ids spill onto the following lines,
# which carry no keyword; such a continuation line is recognised by an all-integer first
# token and folded into the open block.
_MATERIAL = "Material"

# A double-quoted value on a ``Grid`` / ``HydroMat`` card, e.g. ``Grid "Muncie.srhgeom"``.
# Matched on the raw line because a path may contain spaces the whitespace split loses.
_QUOTED = re.compile(r'"([^"]+)"')


class Srh2dDriver(BaseDriver):
    """Reader for the SRH-2D (USBR/Aquaveo) ASCII mesh trio.

    The input is one of the ``.srhgeom`` / ``.srhhydro`` sibling files; the driver
    resolves the rest of the trio from it. ``read`` builds the mesh from ``.srhgeom`` and
    attaches the material-zone id and Manning's n as face variables when the ``.srhmat``
    and ``.srhhydro`` siblings describe them cleanly.

    Documented limitations:
        - **The input is a file set.** Point ``read_file`` at either the ``.srhgeom`` or
          the ``.srhhydro``; the ``Grid`` card (or the shared basename stem) resolves the
          geometry, and the ``.srhmat`` / ``.srhhydro`` siblings are read for metadata
          when present. A ``.srhhydro`` whose geometry cannot be resolved is declined by
          ``can_open``.
        - **Faces are mixed-width polygons.** Triangles, quads and larger polygons share
          one fill-padded connectivity array; short rows are padded with ``-1``.
        - **Node and element ids are resolved through an id map,** so a gapped or
          out-of-order file stays correctly wired. The ids are written 1-based, which is
          recorded for a writer.
        - **Bed elevation becomes a ``node_z`` variable,** since the mesh model is planar.
        - **Material and Manning's n are carried only when clean.** The ``material`` face
          variable is attached when every element is assigned exactly one known zone; the
          ``manning_n`` face variable is attached on top of that when every assigned zone
          has a ``ManningsN`` value. An inconsistent map is dropped rather than guessed.
        - **Results are not read here.** SRH-2D writes results as XMDF or VTK, read by the
          ``xmdf`` and VTK drivers; this reader is the geometry/control reader.
        - **SRH-2D carries no CRS,** so ``crs`` is always ``None``; the ``GridUnit`` card
          is recorded as a global attribute but never interpreted as a projection.

    Example:
        Read an SRH-2D mesh by either member file (shown, not run as a doctest -- it needs
        the committed sample trio)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("Muncie.srhgeom", driver="srh2d")
            elevation = mesh.get("node_z")      # node-located bed elevation
            zone = mesh.get("material")         # face-located material-zone id
    """

    name = "srh2d"
    long_name = "SRH-2D mesh"
    suffixes = (".srhgeom", ".srhhydro")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is a readable member of an SRH-2D trio.

        Overrides the base check because the format is a file set: a ``.srhgeom`` is
        accepted when its header is ``SRHGEOM``, and a ``.srhhydro`` when its header is
        ``SRHHYDRO`` *and* its geometry file resolves. Never raises -- a sniff error is
        warned and the file declined, as the base contract requires.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a ``.srhgeom`` with the right header, or a ``.srhhydro``
                with the right header whose geometry sibling exists.
        """
        if not os.path.isfile(path) or not self._has_suffix(path):
            return False
        try:
            lowered = path.lower()
            if lowered.endswith(".srhgeom"):
                return self._first_token(path) == _GEOM_HEADER
            # .srhhydro: the header must match and the geometry must resolve, so a lone
            # control deck with no mesh beside it is declined rather than half-accepted.
            if self._first_token(path) != _HYDRO_HEADER:
                return False
            return self._geom_from_hydro(path) is not None
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            warnings.warn(
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an SRH-2D trio into a :class:`ParsedMesh`, from either member file.

        Args:
            path: The ``.srhgeom`` or ``.srhhydro`` member file.
            lazy: Unused -- an SRH-2D mesh is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, bed elevation, the fill-padded face connectivity and the
                material/Manning face variables when the siblings are clean.

        Raises:
            FileFormatNotSupportedError: If the geometry cannot be resolved, the header is
                wrong, a record is short of its fields, a field is not a number, or the
                file declares no elements.
            ValueError: If an element references an undefined node id, or two nodes or two
                elements share an id.
        """
        geom = self._geom_path(path)
        if geom is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an SRH-2D .srhgeom/.srhhydro file, or its geometry "
                f"could not be resolved"
            )
        hydro = self._hydro_path(path, geom)
        material = self._mat_path(geom, hydro)
        return _Srh2dReader(geom, hydro=hydro, material=material).build()

    # -- resolution across the trio ------------------------------------------------

    def _geom_path(self, path: str) -> str | None:
        """Resolve the ``.srhgeom`` file from either member path, or ``None``."""
        if not os.path.isfile(path):
            return None
        lowered = path.lower()
        if lowered.endswith(".srhgeom"):
            return path
        if lowered.endswith(".srhhydro"):
            return self._geom_from_hydro(path)
        return None

    def _geom_from_hydro(self, hydro: str) -> str | None:
        """Resolve the geometry a ``.srhhydro`` points at, by its ``Grid`` card or stem.

        Prefers the ``Grid "<file>"`` card, resolved against the deck's directory; falls
        back to the ``<stem>.srhgeom`` sibling. Returns the path only when it exists.
        """
        referenced = self._quoted_value(hydro, _GRID)
        directory = os.path.dirname(hydro)
        candidates = []
        if referenced is not None:
            candidates.append(os.path.join(directory, referenced))
        candidates.append(self._sibling(hydro, ".srhgeom"))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _hydro_path(self, path: str, geom: str) -> str | None:
        """Return the ``.srhhydro`` for this trio: the input itself, or a stem sibling."""
        if path.lower().endswith(".srhhydro"):
            return path
        sibling = self._sibling(geom, ".srhhydro")
        return sibling if os.path.isfile(sibling) else None

    def _mat_path(self, geom: str, hydro: str | None) -> str | None:
        """Return the ``.srhmat`` for this trio, from the deck's ``HydroMat`` or the stem."""
        if hydro is not None:
            referenced = self._quoted_value(hydro, _HYDRO_MAT)
            if referenced is not None:
                candidate = os.path.join(os.path.dirname(hydro), referenced)
                if os.path.isfile(candidate):
                    return candidate
        sibling = self._sibling(geom, ".srhmat")
        return sibling if os.path.isfile(sibling) else None

    @staticmethod
    def _sibling(path: str, suffix: str) -> str:
        """Return ``path`` with its final suffix replaced by ``suffix``."""
        return os.path.splitext(path)[0] + suffix

    @staticmethod
    def _first_token(path: str) -> str | None:
        """Return the first significant whitespace token of a file, or ``None``."""
        for keyword, _fields in iter_records(path):
            return keyword
        return None

    @staticmethod
    def _quoted_value(path: str, card: str) -> str | None:
        """Return the double-quoted value of a ``<card> "<value>"`` line, or ``None``.

        The control deck is small, so it is scanned line by line; the first matching card
        wins. The value is taken from the quotes rather than the whitespace split so a
        path containing spaces survives.
        """
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.split()[:1] == [card]:
                    match = _QUOTED.search(stripped)
                    if match is not None:
                        return match.group(1)
        return None


class _Srh2dReader:
    """One SRH-2D trio: read the geometry and fold in the clean material metadata."""

    def __init__(self, geom: str, *, hydro: str | None, material: str | None) -> None:
        self.geom = geom
        self.hydro = hydro
        self.material = material

    def build(self) -> ParsedMesh:
        """Read the geometry and assemble the mesh with its face metadata.

        Returns:
            ParsedMesh: The surface mesh, bed elevation and the material/Manning face
                variables when the siblings describe them cleanly.

        Raises:
            FileFormatNotSupportedError: On a wrong header, a short or non-numeric record,
                or a file with no elements.
            ValueError: On an undefined node reference or a duplicate node/element id.
        """
        node_ids, node_x, node_y, node_z, saw_z, elem_ids, elem_nodes, grid_unit = (
            self._read_geometry()
        )
        if not elem_nodes:
            raise FileFormatNotSupportedError(
                f"{self.geom!r} declares no {_ELEM} records, so it carries no mesh"
            )
        node_map = native_id_map(node_ids)
        # Built for the duplicate-id check: nothing references an element by id, but two
        # elements sharing one means a malformed file.
        native_id_map(elem_ids)
        faces = self._resolve_faces(elem_nodes, node_map)
        variables = self._face_variables(elem_ids, len(elem_nodes))
        attributes: dict[str, object] = {"source": "SRH-2D geometry"}
        if grid_unit is not None:
            attributes["grid_unit"] = grid_unit
        return ParsedMesh(
            node_x=np.array(node_x, dtype=np.float64),
            node_y=np.array(node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map. The
            # file's 1-based numbering is recorded separately for a writer to restore.
            start_index=0,
            original_start_index=1,
            node_z=np.array(node_z, dtype=np.float64) if saw_z else None,
            crs=None,
            variables=variables,
            global_attributes=attributes,
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_geometry(self):
        """Stream the ``.srhgeom`` once, collecting nodes, elements and the grid unit.

        Returns:
            tuple: ``(node_ids, node_x, node_y, node_z, saw_z, elem_ids, elem_nodes,
                grid_unit)`` -- the collected columns in file order, a flag for whether any
                node carried a z column, and the ``GridUnit`` string when present.

        Raises:
            FileFormatNotSupportedError: On a wrong header, or a short or non-numeric
                ``Node`` / ``Elem`` record, or ``Node`` records that disagree about z.
        """
        node_ids: list[int] = []
        node_x: list[float] = []
        node_y: list[float] = []
        node_z: list[float] = []
        elem_ids: list[int] = []
        elem_nodes: list[list[int]] = []
        grid_unit: str | None = None
        saw_z = False
        missing_z = False
        header_seen = False
        for keyword, fields in iter_records(self.geom):
            if not header_seen:
                if keyword.upper() != _GEOM_HEADER:
                    raise FileFormatNotSupportedError(
                        f"{self.geom!r} is not an SRH-2D geometry: expected a "
                        f"{_GEOM_HEADER} header, found {keyword!r}"
                    )
                header_seen = True
                continue
            if keyword == _NODE:
                has_z = self._add_node(fields, node_ids, node_x, node_y, node_z)
                saw_z = saw_z or has_z
                missing_z = missing_z or not has_z
            elif keyword == _ELEM:
                self._add_elem(fields, elem_ids, elem_nodes)
            elif keyword == _GRID_UNIT:
                grid_unit = self._quoted_or_first(fields)
        if saw_z and missing_z:
            raise FileFormatNotSupportedError(
                f"{self.geom!r} has {_NODE} records that disagree about the z column; "
                f"elevation would have to be invented for the ones without it"
            )
        return (
            node_ids,
            node_x,
            node_y,
            node_z,
            saw_z,
            elem_ids,
            elem_nodes,
            grid_unit,
        )

    def _add_node(self, fields, node_ids, node_x, node_y, node_z) -> bool:
        """Collect one ``Node <id> <x> <y> [<z>]`` record; return whether z was present."""
        if len(fields) < 3:
            raise FileFormatNotSupportedError(
                f"{self.geom!r} has a short {_NODE} record: expected an id and x, y, "
                f"found {len(fields)} field(s)"
            )
        node_ids.append(self._to_int(fields[0], _NODE))
        node_x.append(self._to_float(fields[1], _NODE))
        node_y.append(self._to_float(fields[2], _NODE))
        if len(fields) > 3:
            node_z.append(self._to_float(fields[3], _NODE))
            return True
        node_z.append(0.0)
        return False

    def _add_elem(self, fields, elem_ids, elem_nodes) -> None:
        """Collect one ``Elem <id> n1 n2 n3[ n4 ...]`` record, keeping the native ids."""
        if len(fields) < 4:
            raise FileFormatNotSupportedError(
                f"{self.geom!r} has a short {_ELEM} record: expected an id and at least "
                f"three node ids, found {len(fields)} field(s)"
            )
        elem_ids.append(self._to_int(fields[0], _ELEM))
        elem_nodes.append([self._to_int(value, _ELEM) for value in fields[1:]])

    def _resolve_faces(self, elem_nodes, node_map) -> np.ndarray:
        """Map element vertex ids to 0-based indices, padding short rows with ``-1``."""
        max_nodes = max(len(nodes) for nodes in elem_nodes)
        faces = np.full((len(elem_nodes), max_nodes), -1, dtype=np.int64)
        for row, nodes in enumerate(elem_nodes):
            for column, native in enumerate(nodes):
                if native not in node_map:
                    raise ValueError(
                        f"{self.geom!r} element references node id {native}, which no "
                        f"{_NODE} record defines"
                    )
                faces[row, column] = node_map[native]
        return faces

    def _face_variables(self, elem_ids, n_face: int) -> dict[str, ParsedVariable]:
        """Build the ``material`` and ``manning_n`` face variables when they are clean.

        ``material`` is attached when the ``.srhmat`` map assigns every element exactly
        one known zone; ``manning_n`` is attached on top of that when every assigned zone
        has a ``ManningsN`` value in the ``.srhhydro``. Anything less is dropped rather
        than filled with a guess -- the mesh reads fine without it.
        """
        variables: dict[str, ParsedVariable] = {}
        zones = self._element_zones(elem_ids)
        if zones is None:
            return variables
        variables["material"] = ParsedVariable(
            location="face", shape=(n_face,), values=zones.astype(np.float64)
        )
        manning = self._manning_by_material()
        if manning:
            needed = {int(zone) for zone in np.unique(zones)}
            if needed <= set(manning):
                column = np.array(
                    [manning[int(zone)] for zone in zones], dtype=np.float64
                )
                variables["manning_n"] = ParsedVariable(
                    location="face", shape=(n_face,), values=column
                )
        return variables

    def _element_zones(self, elem_ids) -> np.ndarray | None:
        """Return a per-element material-zone id array, or ``None`` when not clean.

        Reads the ``.srhmat`` ``Material`` blocks (whose element ids continue onto the
        following unprefixed lines) into an element-id -> zone-id map, then projects it
        onto the geometry's element order. ``None`` is returned -- and no material
        variable attached -- when the file is absent, an element is assigned to two zones,
        a listed element is unknown to the geometry, or any element is left unassigned.
        """
        if self.material is None:
            return None
        assignment: dict[int, int] = {}
        current: int | None = None
        for keyword, fields in iter_records(self.material):
            if keyword == _MATERIAL:
                if not fields:
                    return None
                current = self._maybe_int(fields[0])
                rest = fields[1:]
            elif self._maybe_int(keyword) is not None:
                # A continuation line of the open block: an all-integer row of element ids.
                rest = [keyword, *fields]
            else:
                continue
            if current is None:
                continue
            for token in rest:
                element = self._maybe_int(token)
                if element is None:
                    return None
                if element in assignment:
                    return (
                        None  # assigned twice: ambiguous, so the whole map is dropped
                    )
                assignment[element] = current
        try:
            return np.array([assignment[int(eid)] for eid in elem_ids], dtype=np.int64)
        except KeyError:
            return None  # an element went unassigned, so the map is incomplete

    def _manning_by_material(self) -> dict[int, float]:
        """Return the ``ManningsN <material-id> <n>`` map from the ``.srhhydro``."""
        mapping: dict[int, float] = {}
        if self.hydro is None:
            return mapping
        for keyword, fields in iter_records(self.hydro):
            if keyword == _MANNINGS and len(fields) >= 2:
                material = self._maybe_int(fields[0])
                value = self._maybe_float(fields[1])
                if material is not None and value is not None:
                    mapping[material] = value
        return mapping

    # -- small numeric helpers -----------------------------------------------------

    def _to_int(self, token: str, keyword: str) -> int:
        """Parse an integer field, or raise naming the record it came from."""
        value = self._maybe_int(token)
        if value is None:
            raise FileFormatNotSupportedError(
                f"{self.geom!r} has a {keyword} record with a non-integer field "
                f"{token!r}"
            )
        return value

    def _to_float(self, token: str, keyword: str) -> float:
        """Parse a float field, or raise naming the record it came from."""
        value = self._maybe_float(token)
        if value is None:
            raise FileFormatNotSupportedError(
                f"{self.geom!r} has a {keyword} record with a non-numeric field "
                f"{token!r}"
            )
        return value

    @staticmethod
    def _maybe_int(token: str) -> int | None:
        """Return ``token`` as an int, or ``None`` when it is not one."""
        try:
            return int(token)
        except ValueError:
            return None

    @staticmethod
    def _maybe_float(token: str) -> float | None:
        """Return ``token`` as a float, or ``None`` when it is not one."""
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def _quoted_or_first(fields: list[str]) -> str | None:
        """Return a card's value: the text inside quotes, else the first field."""
        if not fields:
            return None
        joined = " ".join(fields)
        match = _QUOTED.search(joined)
        if match is not None:
            return match.group(1)
        return fields[0]
