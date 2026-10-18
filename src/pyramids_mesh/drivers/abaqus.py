"""Abaqus keyword-deck reader.

Abaqus ``.inp`` is a keyword deck: every significant line is either a *keyword*
line beginning with ``*`` (``*NODE``, ``*ELEMENT, TYPE=CPS3``), a *comment* line
beginning with ``**``, or a *data* line of comma-separated fields belonging to the
most recent keyword. Keywords are case-insensitive.

Only ``*NODE`` and ``*ELEMENT`` are read. ``*NODE`` data lines are
``label, x, y[, z]`` and define the node coordinates against their file labels (an
Abaqus label may be gapped or out of order, so connectivity is resolved through
:func:`~pyramids_mesh._text.native_id_map`, never by assuming ``label == index + 1``).
``*ELEMENT, TYPE=<code>`` data lines are ``label, n1, n2, ...``; a surface element
(``S3``/``STRI3``/``CPS3``/``CPS4``/``S4``/``CAX3``/``CAX4``) becomes a fill-padded
face, and a volumetric element (``C3D4``/``C3D8``) has its node list mapped straight
to a face row -- following the ``exodus`` precedent -- rather than yielding an empty
mesh. See ``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import native_id_map

# Keyword and comment markers. `**` is a comment and must be tested before `*`,
# since a comment line also starts with `*`.
_COMMENT = "**"
_KEYWORD = "*"

# The two keywords this driver reads; every other block is skipped. Compared
# case-folded, because Abaqus keywords are case-insensitive.
_NODE = "NODE"
_ELEMENT = "ELEMENT"
_TYPE_PARAM = "TYPE"

# Element type codes whose node list becomes a face. The surface types are genuine
# 2-D elements (triangles and quads); the volumetric types are read verbatim as a
# face row rather than expanded into their bounding triangles -- a true volumetric
# mesh model does not exist yet (design section 21.1), and an empty mesh would be the
# only alternative.
_SURFACE_TYPES = frozenset({"S3", "STRI3", "CPS3", "CPS4", "S4", "CAX3", "CAX4"})
_VOLUME_TYPES = frozenset({"C3D4", "C3D8"})
_FACE_TYPES = _SURFACE_TYPES | _VOLUME_TYPES

# Short face rows are padded to the widest element with -1, because the vertex ids
# are already 0-based indices by the time they are stored (resolved through the id
# map), so 0 is a real index and only -1 can mark padding.
_PAD = -1


class AbaqusDriver(BaseDriver):
    """Reader for the Abaqus keyword-deck (``.inp``) finite-element format.

    The mesh comes from the ``*NODE`` coordinates and the ``*ELEMENT`` blocks. Node
    labels are resolved through :func:`~pyramids_mesh._text.native_id_map`, so a
    gapped or reordered deck is wired correctly; the connectivity arrives 0-based with
    the ``1`` base recorded for a writer. Elevation becomes a ``node_z`` variable only
    when a ``z`` column is present and not all zero, matching the other readers.

    Documented limitations:
        - **Only ``*NODE`` and ``*ELEMENT`` are read.** Node and element sets,
          sections, materials, steps, boundary conditions and output requests are
          skipped -- this driver builds a mesh, not a model.
        - **A volumetric element becomes a face.** ``C3D4`` and ``C3D8`` have their
          node list read straight into a face row (the ``exodus`` precedent); a
          tetrahedron is not expanded into its four bounding triangles, so its winding
          is judged in the ``xy`` plane like any other face.
        - **Unsupported element types are ignored.** A deck whose only elements are of
          a type this reader does not model builds no faces and is refused, rather than
          guessing a node mapping.
        - There is no CRS in the format, so ``crs`` is ``None``.

    Example:
        Read an Abaqus deck (shown, not run as a doctest -- it needs a real ``.inp``
        file), reached through ``read_file``::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("part.inp")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "abaqus"
    long_name = "Abaqus keyword deck"
    suffixes = (".inp",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first non-comment line begins a keyword.

        An Abaqus deck opens with a keyword line (``*HEADING``, ``*NODE``, ...), so
        the first line that is neither blank nor a ``**`` comment must start with a
        single ``*``. This is deliberately the inverse of
        :meth:`~pyramids_mesh.drivers.avsucd.AvsUcdDriver._sniff`, whose first
        significant line is five integers, so the two ``.inp`` drivers never both
        accept one file.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first significant line starts a keyword.
        """
        line = _first_significant_line(path, comment=_COMMENT)
        if line is None:
            return False
        return line.startswith(_KEYWORD)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an Abaqus deck into a :class:`ParsedMesh`.

        Args:
            path: The ``.inp`` file to read.
            lazy: Unused -- the deck is plain text and read eagerly; accepted to match
                the driver protocol.

        Returns:
            ParsedMesh: The surface mesh and, for a 3D deck, ``node_z``.

        Raises:
            FileFormatNotSupportedError: If the deck defines no node or no element this
                reader can model.
            ValueError: If an element references an undefined node label, or two nodes
                share a label.
        """
        return _AbaqusReader(path).build()


def _first_significant_line(path: str, *, comment: str) -> str | None:
    """Return the first non-blank, non-comment line of a text file, or ``None``.

    Reads line by line and stops at the first match, so it stays cheap on a large
    deck. Decoded leniently so a binary or empty file answers ``None`` rather than
    raising.

    Args:
        path: The file to read.
        comment: The line-comment marker to skip.

    Returns:
        str | None: The stripped line, or ``None`` at end of file.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith(comment):
                continue
            return stripped
    return None


def _parse_keyword(line: str) -> tuple[str, dict[str, str]]:
    """Split a ``*KEYWORD, PARAM=VALUE, ...`` line into its name and parameters.

    Args:
        line: A keyword line, including the leading ``*``.

    Returns:
        tuple[str, dict[str, str]]: The upper-cased keyword name and a mapping of
            upper-cased parameter names to their upper-cased values.
    """
    parts = line.lstrip(_KEYWORD).split(",")
    keyword = parts[0].strip().upper()
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, _, value = part.partition("=")
            params[key.strip().upper()] = value.strip().upper()
    return keyword, params


class _AbaqusReader:
    """One Abaqus deck: collect nodes and element node lists, then resolve ids."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.element_vertices: list[list[int]] = []
        # The keyword block currently in effect: "node", a supported element type, or
        # None for a block this reader skips.
        self._state: str | None = None
        # An element data line ending in a comma continues on the next line; its
        # tokens accumulate here until a line that does not continue finishes it.
        self._element_buffer: list[str] = []

    def build(self) -> ParsedMesh:
        """Read the deck and assemble the parsed mesh.

        Returns:
            ParsedMesh: Nodes, fill-padded faces, and ``node_z`` for a 3D deck.

        Raises:
            FileFormatNotSupportedError: If the deck defines no node or no supported
                element.
            ValueError: If an element references an undefined node label, or two nodes
                share a label.
        """
        with open(self.path, "r", encoding="utf-8-sig", errors="replace") as handle:
            for raw in handle:
                self._consume(raw.strip())
        self._flush_element()
        if not self.node_ids:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable Abaqus mesh: it defines no *NODE"
            )
        if not self.element_vertices:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable Abaqus mesh: it has no *ELEMENT of a "
                f"type this reader models ({sorted(_FACE_TYPES)})"
            )
        return self._assemble()

    def _consume(self, line: str) -> None:
        """Dispatch one stripped line onto the current keyword block."""
        if not line or line.startswith(_COMMENT):
            return
        if line.startswith(_KEYWORD):
            self._flush_element()
            self._start_block(line)
            return
        if self._state == _NODE:
            self._add_node(line)
        elif self._state is not None:
            self._add_element_line(line)

    def _start_block(self, line: str) -> None:
        """Begin a new keyword block, recording whether it is read or skipped."""
        keyword, params = _parse_keyword(line)
        if keyword == _NODE:
            self._state = _NODE
        elif keyword == _ELEMENT and params.get(_TYPE_PARAM) in _FACE_TYPES:
            self._state = params[_TYPE_PARAM]
        else:
            self._state = None

    def _add_node(self, line: str) -> None:
        """Collect one ``label, x, y[, z]`` node record.

        Raises:
            FileFormatNotSupportedError: If the record is short of a label and both
                coordinates.
        """
        fields = [token.strip() for token in line.split(",") if token.strip()]
        if len(fields) < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short *NODE record {line!r}: expected a label and "
                f"at least x and y"
            )
        self.node_ids.append(_as_int(fields[0], "*NODE", self.path))
        self.node_x.append(_as_float(fields[1], "*NODE", self.path))
        self.node_y.append(_as_float(fields[2], "*NODE", self.path))
        z = _as_float(fields[3], "*NODE", self.path) if len(fields) > 3 else 0.0
        self.node_z.append(z)

    def _add_element_line(self, line: str) -> None:
        """Collect one ``*ELEMENT`` data line, honouring comma continuation."""
        tokens = [token.strip() for token in line.split(",")]
        self._element_buffer.extend(token for token in tokens if token)
        if not line.endswith(","):
            self._flush_element()

    def _flush_element(self) -> None:
        """Finish the buffered element: drop its label, keep the node list.

        Raises:
            FileFormatNotSupportedError: If a value in the element line is not an
                integer.
        """
        if not self._element_buffer:
            return
        tokens = self._element_buffer
        self._element_buffer = []
        # tokens[0] is the element label, which nothing references; the rest are the
        # node labels making up the element.
        vertices = [_as_int(token, "*ELEMENT", self.path) for token in tokens[1:]]
        if vertices:
            self.element_vertices.append(vertices)

    def _assemble(self) -> ParsedMesh:
        """Resolve node labels to indices and build the fill-padded face table."""
        node_map = native_id_map(self.node_ids)
        max_nodes = max(len(vertices) for vertices in self.element_vertices)
        faces = np.full((len(self.element_vertices), max_nodes), _PAD, dtype=np.int64)
        for row, vertices in enumerate(self.element_vertices):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, self.path)
        node_z = np.array(self.node_z, dtype=np.float64)
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            fill_value=_PAD,
            # The connectivity is already 0-based (resolved through the id map), so
            # there is nothing for the validator to subtract; Abaqus labels its nodes
            # from 1, which is recorded for a writer.
            start_index=0,
            original_start_index=1,
            node_z=node_z if np.any(node_z != 0.0) else None,
            crs=None,
            variables={},
            global_attributes={"source": "Abaqus keyword deck"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Map one native node label to its 0-based index, or report the dangling id."""
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} references node {native}, which no *NODE record defines"
        ) from None


def _as_int(text: str, keyword: str, path: str) -> int:
    """Parse an integer field, naming the keyword and file on failure."""
    try:
        return int(text)
    except ValueError:
        raise FileFormatNotSupportedError(
            f"{path!r} has a {keyword} record with a non-integer value {text!r}"
        ) from None


def _as_float(text: str, keyword: str, path: str) -> float:
    """Parse a float field, naming the keyword and file on failure."""
    try:
        return float(text)
    except ValueError:
        raise FileFormatNotSupportedError(
            f"{path!r} has a {keyword} record with a non-numeric value {text!r}"
        ) from None
