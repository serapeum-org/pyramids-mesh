"""Tecplot finite-element ASCII reader (task C-series).

A Tecplot ASCII dataset is a plain-text header followed by a numeric block:

- ``TITLE = "..."`` -- an optional dataset title.
- ``VARIABLES = "X" "Y" "Z" ...`` -- the variable names, quoted or bare, separated by
  whitespace and/or commas. The first two (``X`` and ``Y``) are the planar coordinates;
  a ``Z`` variable, when present, is the node elevation.
- a finite-element ``ZONE`` header carrying ``N=<nodes>`` and ``E=<elements>``, the
  element type as ``ZONETYPE=FETRIANGLE|FEQUADRILATERAL|FETETRAHEDRON|FEBRICK`` (or the
  legacy ``ET=TRIANGLE|QUADRILATERAL|TETRAHEDRON|BRICK``), the data layout as
  ``DATAPACKING=POINT|BLOCK`` (or the legacy ``F=FEPOINT|FEBLOCK``), and an optional
  ``VARLOCATION([k]=CELLCENTERED)`` marking cell-centred variables.
- the data: in ``POINT`` packing, ``N`` lines each carrying every variable for one node;
  in ``BLOCK``/``FEBLOCK`` packing, each variable as a contiguous block (``N`` values for
  a node-located variable, ``E`` for a cell-centred one).
- the connectivity: ``E`` lines of 1-based node ids, one element per line.

This is a geometry reader: the ``X``/``Y`` (and ``Z``) coordinates and the element
connectivity are read into ``pyramids.netcdf.ugrid`` objects, and extra and cell-centred
variables are skipped. Everything is plain text, so there is no declared dependency
beyond numpy; the driver owns only the file parsing, and the mesh it builds is
pyramids'. See ADR-0005 (the pyramids boundary).
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The keywords that mark a Tecplot header; one of them must lead a significant line for
# ``can_open`` to claim the file. Kept deliberately specific so the generic ``.dat``
# suffix is not stolen from the SMS DAT or FVCOM grid, which carry none of these.
_SIGNATURE = frozenset({"TITLE", "VARIABLES", "ZONE"})

# How ``can_open`` bounds its read: a prefix large enough to clear a long comment or
# variable header, and the first handful of significant lines scanned within it.
_SNIFF_LIMIT = 64 * 1024
_SNIFF_LINES = 16

# The finite-element element types this reader supports, mapped to their node count.
# Legacy (``ET=``) spellings and the modern (``ZONETYPE=``) ``FE``-prefixed ones alike.
# A tetrahedron and a brick are read as a single face of 4 and 8 nodes rather than
# expanded into bounding faces, matching the Exodus reader's precedent: a volumetric
# mesh is not modelled yet, so the element's node list becomes one face row.
_ELEMENT_NODES = {
    "FETRIANGLE": 3,
    "TRIANGLE": 3,
    "FEQUADRILATERAL": 4,
    "QUADRILATERAL": 4,
    "FETETRAHEDRON": 4,
    "TETRAHEDRON": 4,
    "FEBRICK": 8,
    "BRICK": 8,
}

# The canonical (modern) name each legacy element type is reported under.
_CANONICAL = {
    "TRIANGLE": "FETRIANGLE",
    "QUADRILATERAL": "FEQUADRILATERAL",
    "TETRAHEDRON": "FETETRAHEDRON",
    "BRICK": "FEBRICK",
}

_CELL_CENTERED = "CELLCENTERED"
_PUNCTUATION = frozenset("=()[],")


def _strip_comment(line: str) -> str:
    """Return ``line`` with any ``#`` comment removed, respecting quoted strings.

    A ``#`` inside a double-quoted token (a variable name may hold one) is not a comment
    marker, so quote state is tracked rather than cutting at the first ``#`` blindly.

    Args:
        line: One raw line of the file.

    Returns:
        str: The line up to its first unquoted ``#``.
    """
    in_quote = False
    for index, char in enumerate(line):
        if char == '"':
            in_quote = not in_quote
        elif char == "#" and not in_quote:
            return line[:index]
    return line


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Split header-and-data text into ``(kind, value)`` tokens.

    The kinds are ``"quote"`` (a double-quoted string, quotes stripped), ``"punct"`` (one
    of ``= ( ) [ ] ,``) and ``"word"`` (any other run of non-space characters, be it a
    keyword, an element type or a number). Whitespace -- including the newlines that let a
    zone header and its data span many lines -- only separates tokens.

    Args:
        text: The comment-stripped file text.

    Returns:
        list[tuple[str, str]]: The tokens in order.
    """
    tokens: list[tuple[str, str]] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
        elif char == '"':
            end = text.find('"', index + 1)
            if end == -1:
                end = length
            tokens.append(("quote", text[index + 1 : end]))
            index = end + 1
        elif char in _PUNCTUATION:
            tokens.append(("punct", char))
            index += 1
        else:
            end = index
            while (
                end < length
                and not text[end].isspace()
                and text[end] != '"'
                and text[end] not in _PUNCTUATION
            ):
                end += 1
            tokens.append(("word", text[index:end]))
            index = end
    return tokens


def _as_number(token: str) -> float | None:
    """Return ``token`` parsed as a float, or ``None`` when it is not numeric."""
    try:
        return float(token)
    except ValueError:
        return None


def _parse_index_spec(spec: str) -> set[int]:
    """Parse a ``VARLOCATION`` bracket index spec into a set of 1-based indices.

    Accepts a single index, a comma list and a hyphen range, as Tecplot writes them.

    Args:
        spec: The text between the brackets, e.g. ``"4"``, ``"3,5"`` or ``"3-5"``.

    Returns:
        set[int]: The 1-based variable indices the spec names.

    Examples:
        - A single index, a range and a comma list each expand as written:

            >>> sorted(_parse_index_spec("4"))
            [4]
            >>> sorted(_parse_index_spec("3-5"))
            [3, 4, 5]
            >>> sorted(_parse_index_spec("2,4"))
            [2, 4]
    """
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-", 1)
            result.update(range(int(start), int(stop) + 1))
        else:
            result.add(int(part))
    return result


def _peek_is_eq(tokens: list[tuple[str, str]], pos: int) -> bool:
    """Report whether the token at ``pos`` is an ``=`` sign."""
    return pos < len(tokens) and tokens[pos] == ("punct", "=")


def _read_value(
    tokens: list[tuple[str, str]], pos: int
) -> tuple[tuple[str, object], int]:
    """Read one header value, returning ``((kind, value), next_pos)``.

    A value is a quoted string, a bare word, or a parenthesised group (as
    ``VARLOCATION`` carries). A group is returned as ``("group", inner_tokens)`` so the
    caller can interpret it; everything else as ``(kind, text)``.

    Args:
        tokens: The token stream.
        pos: The index of the value's first token.

    Returns:
        tuple: ``((kind, value), next_pos)``.
    """
    kind, value = tokens[pos]
    if kind == "punct" and value == "(":
        depth = 0
        inner: list[tuple[str, str]] = []
        while pos < len(tokens):
            token_kind, token_value = tokens[pos]
            pos += 1
            if token_kind == "punct" and token_value == "(":
                depth += 1
                if depth == 1:
                    continue
            elif token_kind == "punct" and token_value == ")":
                depth -= 1
                if depth == 0:
                    break
            inner.append((token_kind, token_value))
        return ("group", inner), pos
    return (kind, value), pos + 1


def _cell_centered_indices(inner: list[tuple[str, str]]) -> set[int]:
    """Return the 1-based indices a ``VARLOCATION`` group marks cell-centred.

    The group reads ``[<indices>] = CELLCENTERED`` one or more times; any location other
    than ``CELLCENTERED`` (the default ``NODAL``) is left out of the returned set.

    Args:
        inner: The tokens inside the ``VARLOCATION`` parentheses.

    Returns:
        set[int]: The 1-based variable indices located at cell centres.
    """
    cell: set[int] = set()
    index = 0
    while index < len(inner):
        kind, value = inner[index]
        if kind == "punct" and value == "[":
            end = index + 1
            spec_parts: list[str] = []
            while end < len(inner) and inner[end] != ("punct", "]"):
                spec_parts.append(inner[end][1])
                end += 1
            spec = "".join(spec_parts)
            location, end = _value_after_equals(inner, end + 1)
            if location is not None and location.upper() == _CELL_CENTERED:
                cell.update(_parse_index_spec(spec))
            index = end
            continue
        index += 1
    return cell


def _value_after_equals(
    inner: list[tuple[str, str]], pos: int
) -> tuple[str | None, int]:
    """Return the first word after an optional ``=``, and the index past it."""
    while pos < len(inner):
        kind, value = inner[pos]
        if kind == "punct" and value == "=":
            pos += 1
            continue
        return value, pos + 1
    return None, pos


class TecplotDriver(BaseDriver):
    """Reader for the Tecplot finite-element ASCII mesh format.

    The mesh is built from the ``X``/``Y`` coordinate variables and the ``ZONE``
    connectivity; a ``Z`` variable, when it carries any non-zero value, becomes a
    ``node_z`` node variable. Both data packings (``POINT`` and ``BLOCK``/``FEBLOCK``) and
    both keyword generations (modern ``ZONETYPE``/``DATAPACKING`` and legacy
    ``ET``/``F``) are accepted. The 1-based connectivity drops to 0-based and the base is
    recorded for a writer.

    Because ``.dat`` is a generic suffix other formats also use, ``can_open`` requires a
    Tecplot signature (a ``TITLE``/``VARIABLES``/``ZONE`` keyword near the top), so it
    never steals an SMS DAT or an FVCOM ``.dat``.

    Documented limitations:
        - **It is a geometry reader.** Only ``X``/``Y`` (and ``Z``) are read; every other
          variable, and every cell-centred variable, is skipped. ``VARLOCATION`` is parsed
          only so a cell-centred variable's block length (``E`` rather than ``N``) is read
          correctly and the connectivity stays aligned.
        - **A volumetric element becomes one face.** ``FETETRAHEDRON`` (4 nodes) and
          ``FEBRICK`` (8 nodes) are read as a single face of that many nodes rather than
          expanded into bounding faces -- a 3D mesh is not modelled yet (design §21.1) --
          so winding is judged in the ``xy`` plane like any other face, matching the
          Exodus reader's precedent.
        - **The first finite-element zone is read.** A file carrying more than one zone
          yields only the first; the rest are ignored.
        - **Connectivity is 1-based**, resolved to 0-based with the ``1`` base recorded for
          a writer. A single-element-type zone needs no fill padding.
        - **An ordered (structured I/J/K) zone is refused** -- this reader builds
          finite-element meshes only.
        - There is no CRS in the format, so ``crs`` is ``None``; it is never guessed from
          the coordinate magnitudes.

    Example:
        Read a Tecplot FE mesh (shown, not run as a doctest -- it needs a real ``.tec``
        file), reached through ``read_file`` or the driver directly::

            from pyramids_mesh.drivers.tecplot import TecplotDriver

            dataset = TecplotDriver().read("mesh.tec")
            dataset.mesh.n_node, dataset.mesh.n_face
    """

    name = "tecplot"
    long_name = "Tecplot finite-element ASCII"
    suffixes = (".dat", ".tec")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether a Tecplot keyword leads one of the file's first lines.

        A bounded prefix is read and decoded leniently -- a binary or empty file answers
        ``False`` rather than raising -- and the first ``_SNIFF_LINES`` significant
        (non-blank, comment-stripped) lines are scanned. The file is Tecplot when one of
        them begins with ``TITLE``, ``VARIABLES`` or ``ZONE``. The SMS DAT
        (``DATASET``/``BEGSCL``/...) and the FVCOM grid (``Node Number``/``Cell Number``)
        carry none of these, so both are declined even though they share the ``.dat``
        suffix. Never raises.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when a leading Tecplot keyword is found.
        """
        with open(path, "rb") as handle:
            prefix = handle.read(_SNIFF_LIMIT)
        seen = 0
        for raw in prefix.decode("utf-8-sig", errors="replace").splitlines():
            stripped = _strip_comment(raw).strip()
            if not stripped:
                continue
            if stripped.split()[0].upper() in _SIGNATURE:
                return True
            seen += 1
            if seen >= _SNIFF_LINES:
                break
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Tecplot FE ASCII file into a :class:`ParsedMesh`.

        Args:
            path: The ``.dat``/``.tec`` file to read.
            lazy: Unused -- a Tecplot ASCII file is plain text and read whole; accepted to
                match the driver protocol.

        Returns:
            ParsedMesh: Nodes, the 1-based face connectivity and ``node_z`` for a file
                carrying a non-zero ``Z`` variable.

        Raises:
            FileFormatNotSupportedError: If the file carries no ``ZONE`` header, declares
                no supported element type, is an ordered (structured) zone, omits ``N`` or
                ``E``, lacks ``X``/``Y`` variables, or is truncated against the counts its
                header promises.
        """
        return _TecplotReader(path).build()


class _TecplotReader:
    """One Tecplot FE ASCII file: parse the header, then slice the numeric block."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.title: str | None = None
        self.names: list[str] = []
        self.store: dict[str, str | None] = {}
        self.cell: set[int] = set()

    def build(self) -> ParsedMesh:
        """Read the file and assemble the mesh.

        Returns:
            ParsedMesh: The surface mesh with its coordinates, connectivity and optional
                ``node_z``.

        Raises:
            FileFormatNotSupportedError: On any header or truncation problem.
        """
        with open(self.path, "r", encoding="utf-8-sig", errors="replace") as handle:
            cleaned = "\n".join(_strip_comment(line) for line in handle)
        tokens = _tokenize(cleaned)
        pos = self._read_preamble(tokens)
        if pos is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a Tecplot file: no ZONE header was found"
            )
        pos = self._read_zone_header(tokens, pos)
        data = self._collect_data(tokens, pos)
        return self._assemble(data)

    def _read_preamble(self, tokens: list[tuple[str, str]], pos: int = 0) -> int | None:
        """Read ``TITLE``/``VARIABLES`` (and skip any other key) up to the ``ZONE`` word.

        Args:
            tokens: The token stream.
            pos: Where to start.

        Returns:
            int | None: The index just past ``ZONE``, or ``None`` when there is none.
        """
        while pos < len(tokens):
            kind, value = tokens[pos]
            if kind == "word" and value.upper() == "ZONE":
                return pos + 1
            if kind != "word":
                pos += 1
                continue
            keyword = value.upper()
            pos += 1
            if _peek_is_eq(tokens, pos):
                pos += 1
            if keyword == "VARIABLES":
                self.names, pos = self._read_variable_names(tokens, pos)
            elif keyword == "TITLE":
                (vkind, vvalue), pos = _read_value(tokens, pos)
                self.title = vvalue if vkind != "group" else None
            elif pos < len(tokens):
                _value, pos = _read_value(tokens, pos)
        return None

    @staticmethod
    def _read_variable_names(
        tokens: list[tuple[str, str]], pos: int
    ) -> tuple[list[str], int]:
        """Collect the variable names, quoted or bare, up to the ``ZONE`` keyword.

        A bare word immediately followed by ``=`` is a new key rather than a name, so
        collection stops there as well -- a defence against a header with no explicit
        ``ZONE`` line between the variables and the zone keys.
        """
        names: list[str] = []
        while pos < len(tokens):
            kind, value = tokens[pos]
            if kind == "word" and value.upper() == "ZONE":
                break
            if kind == "punct":
                if value == ",":
                    pos += 1
                    continue
                break
            if kind == "word" and _peek_is_eq(tokens, pos + 1):
                break
            names.append(value)
            pos += 1
        return names, pos

    def _read_zone_header(self, tokens: list[tuple[str, str]], pos: int) -> int:
        """Read the ``KEY = VALUE`` pairs of the zone header up to the first data value.

        Each key is an alphabetic word; the data begins at the first numeric word, which
        a key can never be, so the boundary is unambiguous.

        Args:
            tokens: The token stream.
            pos: The index just past the ``ZONE`` keyword.

        Returns:
            int: The index of the first data token.
        """
        while pos < len(tokens):
            kind, value = tokens[pos]
            if kind != "word" or not value or not value[0].isalpha():
                break
            keyword = value.upper()
            pos += 1
            if _peek_is_eq(tokens, pos):
                pos += 1
            (vkind, vvalue), pos = _read_value(tokens, pos)
            if keyword == "VARLOCATION" and vkind == "group":
                self.cell = _cell_centered_indices(vvalue)
            elif vkind != "group":
                self.store[keyword] = vvalue
        return pos

    def _collect_data(self, tokens: list[tuple[str, str]], pos: int) -> list[str]:
        """Collect the contiguous run of numeric tokens that is this zone's data.

        Collection stops at the first non-numeric word (a second ``ZONE`` keyword, say),
        so a multi-zone file yields only the first zone's numbers.
        """
        data: list[str] = []
        while pos < len(tokens):
            kind, value = tokens[pos]
            if kind == "word" and _as_number(value) is not None:
                data.append(value)
                pos += 1
            else:
                break
        return data

    def _assemble(self, data: list[str]) -> ParsedMesh:
        """Slice the numeric block into variables and connectivity and build the mesh."""
        n_node, n_elem, nodes_per_elem, canonical = self._dimensions()
        packing = self._packing()
        columns = self._variable_columns(data, n_node, n_elem, packing)
        faces = self._connectivity(data, n_node, n_elem, nodes_per_elem, packing)
        node_x, node_y, node_z = self._coordinates(columns)
        attributes: dict[str, object] = {
            "source": "Tecplot FE ASCII",
            "element_type": canonical,
            "data_packing": packing,
        }
        if self.title is not None:
            attributes["title"] = self.title
        zone_title = self.store.get("T")
        if zone_title is not None:
            attributes["zone_title"] = zone_title
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # Tecplot FE connectivity is 1-based; the validator re-bases and records the
            # base for a writer. A single-type zone is uniform, so no padding is needed.
            start_index=1,
            original_start_index=1,
            node_z=node_z,
            crs=None,
            variables={},
            global_attributes=attributes,
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _dimensions(self) -> tuple[int, int, int, str]:
        """Return ``(n_node, n_elem, nodes_per_elem, canonical_element_type)``.

        Raises:
            FileFormatNotSupportedError: On a missing or unsupported element type, an
                ordered zone, or a missing/non-integer ``N`` or ``E``.
        """
        raw = self.store.get("ZONETYPE") or self.store.get("ET")
        if raw is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no ZONETYPE/ET element type; this reader builds "
                f"finite-element meshes, not ordered (structured) zones"
            )
        element = raw.upper()
        if element not in _ELEMENT_NODES:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares element type {raw!r}, which is not a supported "
                f"finite-element type (expected one of "
                f"{sorted(set(_CANONICAL.values()))})"
            )
        n_node = self._header_int("N")
        n_elem = self._header_int("E")
        return n_node, n_elem, _ELEMENT_NODES[element], _CANONICAL.get(element, element)

    def _header_int(self, key: str) -> int:
        """Return a zone-header count as an int, or raise naming the missing/bad key."""
        value = self.store.get(key)
        number = None if value is None else _as_number(value)
        if number is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is missing a usable {key}=<count> in its ZONE header"
            )
        return int(number)

    def _packing(self) -> str:
        """Return ``"POINT"`` or ``"BLOCK"`` from ``DATAPACKING`` or the legacy ``F``."""
        packing = self.store.get("DATAPACKING")
        if packing is not None:
            return "BLOCK" if "BLOCK" in packing.upper() else "POINT"
        legacy = self.store.get("F")
        if legacy is not None:
            return "BLOCK" if "BLOCK" in legacy.upper() else "POINT"
        # Tecplot's default layout is POINT when neither keyword is given.
        return "POINT"

    def _variable_columns(
        self, data: list[str], n_node: int, n_elem: int, packing: str
    ) -> dict[str, tuple[str, np.ndarray]]:
        """Slice each variable's values out of the data block, keyed by name.

        In ``POINT`` packing every variable is node-located and the values interleave,
        one row of ``n_var`` values per node. In ``BLOCK`` packing each variable is a
        contiguous block -- ``n_node`` values for a node-located variable, ``n_elem`` for
        a cell-centred one.

        Raises:
            FileFormatNotSupportedError: If the data block is shorter than the header's
                counts promise.
        """
        n_var = len(self.names)
        if packing == "POINT":
            needed = n_node * n_var
            self._require_data(data, needed, "variable values")
            table = np.array(data[:needed], dtype=np.float64).reshape(n_node, n_var)
            return {
                name: ("node", table[:, index]) for index, name in enumerate(self.names)
            }
        columns: dict[str, tuple[str, np.ndarray]] = {}
        offset = 0
        for index, name in enumerate(self.names):
            cell_centered = (index + 1) in self.cell
            count = n_elem if cell_centered else n_node
            self._require_data(data, offset + count, "variable values")
            block = np.array(data[offset : offset + count], dtype=np.float64)
            offset += count
            columns[name] = ("face" if cell_centered else "node", block)
        return columns

    def _connectivity(
        self,
        data: list[str],
        n_node: int,
        n_elem: int,
        nodes_per_elem: int,
        packing: str,
    ) -> np.ndarray:
        """Slice the ``n_elem`` connectivity rows off the tail of the data block."""
        if packing == "POINT":
            offset = n_node * len(self.names)
        else:
            offset = sum(
                n_elem if (index + 1) in self.cell else n_node
                for index in range(len(self.names))
            )
        needed = offset + n_elem * nodes_per_elem
        self._require_data(data, needed, "connectivity")
        flat = np.array(data[offset:needed], dtype=np.int64)
        return flat.reshape(n_elem, nodes_per_elem)

    def _require_data(self, data: list[str], needed: int, what: str) -> None:
        """Raise a truncation error unless the data block holds ``needed`` values."""
        if len(data) < needed:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is truncated: its {what} needs {needed} values but the "
                f"file provides only {len(data)}"
            )

    def _coordinates(
        self, columns: dict[str, tuple[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Pull the ``X``/``Y`` coordinates and the optional ``Z`` elevation.

        A ``Z`` variable becomes ``node_z`` only when it carries a non-zero value,
        matching the gmsh, vtk and Exodus readers: an all-zero column is dropped rather
        than kept as a meaningless elevation.

        Raises:
            FileFormatNotSupportedError: If the file names no ``X`` or no ``Y`` variable.
        """
        coordinate: dict[str, np.ndarray] = {}
        for name, (_location, values) in columns.items():
            upper = name.upper()
            if upper in ("X", "Y", "Z") and upper not in coordinate:
                coordinate[upper] = values
        if "X" not in coordinate or "Y" not in coordinate:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no X/Y coordinate variables to build a mesh from; "
                f"its variables are {self.names!r}"
            )
        node_z = None
        if "Z" in coordinate and np.any(coordinate["Z"] != 0.0):
            node_z = coordinate["Z"]
        return coordinate["X"], coordinate["Y"], node_z
