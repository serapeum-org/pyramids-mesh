"""Nastran bulk-data reader.

A Nastran bulk-data deck (``.bdf``/``.nas``/``.fem``) is a plain-text finite-element
model: ``GRID`` cards give node coordinates and ``CTRIA3``/``CQUAD4`` cards give the
surface element-to-node lists. Node ids are arbitrary labels, so connectivity is
resolved through :func:`~pyramids_mesh._text.native_id_map` rather than by assuming
``id == index + 1``.

Three card layouts share one deck: **small** fixed-field (ten 8-character columns),
**large** fixed-field (a ``*`` on the card name, 16-character columns spilling onto a
continuation line), and **free** field (comma-separated). Fixed-field reals may embed the
exponent sign without an ``E`` -- ``-2.45-16`` means ``-2.45e-16`` -- and that spelling is
parsed here.

Volumetric elements (``CTETRA``/``CPENTA``/``CHEXA``/``CPYRA``) and anything else are
skipped: this reader builds the surface mesh only, in line with ADR-0005 (parse into
``pyramids.netcdf.ugrid`` objects, no GIS, no analysis). A genuine 3D mesh is out of
scope. The deck carries engineering coordinates with no datum, so ``crs`` is ``None``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import native_id_map

# A bulk-data comment line starts with a dollar sign.
_COMMENT = "$"

_GRID = "GRID"
# Surface element cards and their corner-node counts. Higher-order mid-side nodes
# (CTRIA6/CQUAD8) are deliberately not read: treating a mid-side node as a corner would
# silently distort the element, the same stance the 2DM reader takes on E6T's siblings.
_FACE_CARDS = {"CTRIA3": 3, "CQUAD4": 4}

# A large-field card name carries a '*'; a continuation line opens with '*' or '+', and a
# blank name field is a fixed-field continuation too.
_LARGE = "*"
_CONTINUATION = ("*", "+")

# A DHI FEFLOW mesh shares the '.fem' suffix but opens with these header words and carries
# no GRID/CTRIA3/CQUAD4 cards. The sniff declines it explicitly so the two never collide.
_FEFLOW_MARKERS = ("PROBLEM", "CLASS")

# can_open must not read an unbounded file. The sniff scans a bounded prefix for the
# defining cards; a FEFLOW marker at the top is caught long before this. The bound matches
# the SMS driver's.
_SNIFF_LIMIT = 256 * 1024

# Fixed-field column geometry: ten fields, the first and last 8 columns, the eight data
# fields 8 columns (small) or 16 (large) wide.
_FIELD_SMALL = 8
_FIELD_LARGE = 16
_NAME_WIDTH = 8
_BODY_END = 72


def _parse_real(text: str, path: str) -> float:
    """Parse a Nastran real field, including an exponent with no ``E``.

    A bulk-data real may be written ``1.0``, ``-2.45E-16``, the Fortran-double ``1.5D+3``,
    or -- in a fixed-field column where every character counts -- ``-2.45-16`` with the
    exponent sign abutting the mantissa and no ``E`` at all. The last spelling is rebuilt
    into a form :func:`float` accepts by inserting an ``E`` before a ``+``/``-`` that is
    neither the leading sign nor already an exponent marker.

    Args:
        text: The raw field, with or without surrounding whitespace.
        path: The source path, for the error message.

    Returns:
        float: The parsed value; an empty field is ``0.0``, the bulk-data default.

    Raises:
        FileFormatNotSupportedError: If the field is not a readable real.

    Examples:
        - The fixed-field spelling with an embedded exponent sign is parsed:

            >>> _parse_real("-2.45-16", "deck.bdf")
            -2.45e-16
            >>> _parse_real("2.06-15", "deck.bdf")
            2.06e-15

        - Plain, Fortran-double and bare-decimal spellings all read:

            >>> _parse_real(".7071068", "deck.bdf")
            0.7071068
            >>> _parse_real("1.5D+3", "deck.bdf")
            1500.0
            >>> _parse_real("   ", "deck.bdf")
            0.0
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    candidate = stripped.replace("D", "E").replace("d", "E")
    try:
        return float(candidate)
    except ValueError:
        pass
    rebuilt = [candidate[0]]
    for index in range(1, len(candidate)):
        character = candidate[index]
        if character in "+-" and candidate[index - 1] not in "eE":
            rebuilt.append("E")
        rebuilt.append(character)
    try:
        return float("".join(rebuilt))
    except ValueError:
        raise FileFormatNotSupportedError(
            f"{path!r} has a field that is not a Nastran real: {text!r}"
        ) from None


def _as_int(text: str, card: str, path: str) -> int:
    """Parse an integer id field, naming the card it came from on failure.

    Args:
        text: The raw field.
        card: The card keyword, for the error message.
        path: The source path, for the error message.

    Returns:
        int: The parsed id.

    Raises:
        FileFormatNotSupportedError: If the field is not an integer.
    """
    try:
        return int(text.strip())
    except ValueError:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {card} record: {text!r} is not an integer id"
        ) from None


def _card_name(line: str) -> str:
    """Return a line's card keyword, folded and with the large-field ``*`` removed.

    The keyword is the first comma-delimited token of a free-field line, else the first
    eight columns of a fixed-field one.

    Args:
        line: One physical line, trailing newline already removed.

    Returns:
        str: The upper-cased keyword, e.g. ``GRID`` for ``GRID*`` or ``GRID,1,...``.
    """
    head = line.split(",", 1)[0] if "," in line else line[:_NAME_WIDTH]
    return head.strip().rstrip(_LARGE).upper()


def _fields_of(line: str) -> list[str]:
    """Return a line's data fields (fields 2..9), by its own small/large/free layout.

    Each line declares its own layout: a comma makes it free-field, a ``*`` in the name
    column makes it large-field, otherwise it is small-field. The returned list is the
    raw column text, unstripped, so an abutting fixed-field real such as ``-2.45-160.0``
    is split by column rather than by whitespace.

    Args:
        line: One physical line.

    Returns:
        list[str]: The data fields after the keyword, in file order.
    """
    if "," in line:
        return line.split(",")[1:]
    width = _FIELD_LARGE if _LARGE in line[:_NAME_WIDTH] else _FIELD_SMALL
    return [
        line[start : start + width] for start in range(_NAME_WIDTH, _BODY_END, width)
    ]


def _logical_cards(path: str) -> list[tuple[str, list[str]]]:
    """Read the deck into ``(keyword, fields)`` cards, joining continuation lines.

    Comment (``$``) and blank lines are skipped. A line whose name field opens with
    ``*``/``+`` or is blank continues the previous card, so a large-field ``GRID*`` and
    its continuation line become one card with all its fields in order.

    Args:
        path: The deck to read.

    Returns:
        list[tuple[str, list[str]]]: One ``(keyword, fields)`` pair per logical card.
    """
    cards: list[tuple[str, list[str]]] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith(_COMMENT):
                continue
            name = _card_name(line)
            fields = _fields_of(line)
            opener = (
                line.split(",", 1)[0] if "," in line else line[:_NAME_WIDTH]
            ).strip()
            if (opener == "" or opener.startswith(_CONTINUATION)) and cards:
                cards[-1][1].extend(fields)
            else:
                cards.append((name, fields))
    return cards


class NastranDriver(BaseDriver):
    """Reader for the Nastran bulk-data (``.bdf``/``.nas``/``.fem``) mesh format.

    Builds the surface mesh from ``GRID`` nodes and ``CTRIA3``/``CQUAD4`` faces. Node ids
    are arbitrary labels resolved through an id map, so a gapped or out-of-order deck wires
    correctly; the connectivity arrives 0-based with the bulk-data base recorded for a
    writer. ``CTRIA3`` rows are fill-padded to the four columns of ``CQUAD4``. Node
    elevation (the ``GRID`` ``X3`` column) becomes a ``node_z`` variable when any value is
    non-zero, matching the Exodus and gmsh readers.

    Documented limitations:
        - **Volumetric and line elements are skipped.** ``CTETRA``/``CPENTA``/``CHEXA``/
          ``CPYRA`` solids and ``CBAR``/``CBEAM`` bars are dropped; only ``CTRIA3`` and
          ``CQUAD4`` surface faces are read. A genuine 3D mesh is out of scope (design
          section 21.1), so a deck of only solids is refused for having no surface.
        - **Three field layouts are read** -- small (8-column) and large (16-column)
          fixed-field, and free (comma-separated). A fixed-field real may embed its
          exponent sign without an ``E`` (``-2.45-16`` is ``-2.45e-16``).
        - **Node ids are arbitrary labels**, mapped to contiguous indices; an element
          naming an id no ``GRID`` defines is refused, and a duplicate ``GRID`` id is
          ambiguous and refused.
        - Nastran carries no CRS -- the coordinates are engineering coordinates with no
          datum -- so ``crs`` is always ``None``; it is never guessed.
        - The ``.fem`` suffix is shared with DHI FEFLOW. ``can_open`` requires the Nastran
          signature (a ``GRID`` and a ``CTRIA3``/``CQUAD4`` card, no FEFLOW
          ``PROBLEM:``/``CLASS`` header), so a FEFLOW ``.fem`` is declined.

    Example:
        Read a Nastran deck (shown, not run as a doctest -- it needs a real deck)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("model.bdf")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "nastran"
    long_name = "Nastran bulk data"
    suffixes = (".bdf", ".nas", ".fem")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a Nastran deck, declining a FEFLOW ``.fem``.

        A bounded prefix is scanned for a ``GRID`` card and a ``CTRIA3``/``CQUAD4`` card;
        a FEFLOW ``PROBLEM:``/``CLASS`` header -- the one real collision on the shared
        ``.fem`` suffix -- declines the file outright. Bytes are decoded leniently, so a
        binary or empty file answers ``False`` rather than raising.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` only when both a ``GRID`` and a surface element card are found
                and no FEFLOW header appears.
        """
        with open(path, "rb") as handle:
            prefix = handle.read(_SNIFF_LIMIT)
        has_grid = False
        has_face = False
        first_significant = True
        for raw in prefix.decode("utf-8-sig", errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith(_COMMENT):
                continue
            upper = stripped.upper()
            # A FEFLOW .fem opens with PROBLEM:/CLASS; key the decline on the first
            # significant line only, so a Nastran line merely beginning with those words
            # deeper in the deck does not false-decline.
            if first_significant:
                first_significant = False
                if any(upper.startswith(marker) for marker in _FEFLOW_MARKERS):
                    return False
            keyword = _card_name(raw)
            if keyword == _GRID:
                has_grid = True
            elif keyword in _FACE_CARDS:
                has_face = True
            if has_grid and has_face:
                return True
        return has_grid and has_face

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Nastran deck into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The deck to read.
            lazy: Ignored -- a bulk-data deck is plain text and read eagerly; accepted to
                match the driver protocol.

        Returns:
            ParsedMesh: Nodes, the surface faces, and ``node_z`` when elevations are set.

        Raises:
            FileFormatNotSupportedError: If the deck has no ``GRID`` card, no surface
                element, or a malformed field.
            ValueError: If an element references an undefined node id, or two ``GRID``
                cards share an id.
        """
        return _Deck(path).build()


class _Deck:
    """One Nastran deck's GRID nodes and surface faces, resolved to a parsed mesh."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.face_ids: list[int] = []
        self.face_vertices: list[list[int]] = []

    def build(self) -> ParsedMesh:
        """Read the deck and assemble the surface mesh.

        Returns:
            ParsedMesh: The assembled mesh.

        Raises:
            FileFormatNotSupportedError: If the deck has no ``GRID`` card or no
                ``CTRIA3``/``CQUAD4`` surface element.
            ValueError: If an element references an undefined node id, or two ``GRID``
                cards share an id.
        """
        for keyword, fields in _logical_cards(self.path):
            if keyword == _GRID:
                self._add_grid(fields)
            elif keyword in _FACE_CARDS:
                self._add_face(keyword, fields)
        return self._assemble()

    def _add_grid(self, fields: list[str]) -> None:
        """Collect one ``GRID id cp x1 x2 x3`` card.

        Args:
            fields: The card's data fields, keyword excluded.

        Raises:
            FileFormatNotSupportedError: If the card lacks an id and both planar
                coordinates, or declares a non-zero ``CP`` coordinate frame.
        """
        if len(fields) < 4:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short GRID card: expected at least an id and "
                f"x, y, found {len(fields)} field(s)"
            )
        # X1/X2/X3 are given in the CP coordinate frame; this reader has no frame
        # transform, so a non-zero CP is refused rather than silently read as global.
        cp = fields[1].strip()
        if cp and cp != "0":
            raise FileFormatNotSupportedError(
                f"{self.path!r} GRID card declares a non-zero coordinate frame CP={cp!r}; "
                f"this reader does not transform node coordinates between frames"
            )
        self.node_ids.append(_as_int(fields[0], _GRID, self.path))
        self.node_x.append(_parse_real(fields[2], self.path))
        self.node_y.append(_parse_real(fields[3], self.path))
        self.node_z.append(
            _parse_real(fields[4], self.path) if len(fields) > 4 else 0.0
        )

    def _add_face(self, keyword: str, fields: list[str]) -> None:
        """Collect one ``CTRIA3``/``CQUAD4`` card's element id and corner-node ids.

        Args:
            keyword: The element card keyword.
            fields: The card's data fields, keyword excluded.

        Raises:
            FileFormatNotSupportedError: If the card is short of its ``eid``, ``pid`` and
                corner-node ids.
        """
        count = _FACE_CARDS[keyword]
        if len(fields) < 2 + count:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short {keyword} record: expected an id, a property "
                f"id and {count} node ids"
            )
        self.face_ids.append(_as_int(fields[0], keyword, self.path))
        self.face_vertices.append(
            [_as_int(fields[2 + offset], keyword, self.path) for offset in range(count)]
        )

    def _assemble(self) -> ParsedMesh:
        """Resolve node ids to indices and build the :class:`ParsedMesh`.

        Raises:
            FileFormatNotSupportedError: If no ``GRID`` card or no surface element was
                read.
            ValueError: If an element references an undefined node id, or two ``GRID``
                or two element cards share an id.
        """
        if not self.node_ids:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a Nastran mesh: no GRID card defines any node"
            )
        if not self.face_vertices:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no surface element: no CTRIA3 or CQUAD4 card "
                f"(volumetric elements are not read)"
            )
        node_map = native_id_map(self.node_ids)
        # Built for its duplicate-id check only: nothing references an element by id, but
        # two elements sharing one means a malformed deck, as for the nodes.
        native_id_map(self.face_ids)
        faces = self._resolve_faces(node_map)
        node_z_array = np.array(self.node_z, dtype=np.float64)
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map, so
            # there is nothing for the validator to subtract. Nastran ids are 1-based by
            # convention, recorded for a writer to restore.
            start_index=0,
            original_start_index=1,
            node_z=node_z_array if np.any(node_z_array != 0.0) else None,
            crs=None,
            global_attributes={"source": "Nastran bulk data"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _resolve_faces(self, node_map: dict[int, int]) -> np.ndarray:
        """Map face node ids to 0-based indices, padding short rows with ``-1``.

        Raises:
            ValueError: If a face references a node id no ``GRID`` card defines.
        """
        max_nodes = max(len(vertices) for vertices in self.face_vertices)
        faces = np.full((len(self.face_vertices), max_nodes), -1, dtype=np.int64)
        for row, vertices in enumerate(self.face_vertices):
            for column, native in enumerate(vertices):
                if native not in node_map:
                    raise ValueError(
                        f"{self.path!r} references node id {native} that no GRID card "
                        f"defines"
                    )
                faces[row, column] = node_map[native]
        return faces
