"""SU2 reader (task C11).

SU2 is the native ASCII mesh of the SU2 open-source CFD/multiphysics suite. A ``.su2``
file is a flat, section-keyed text file: a dimension card ``NDIME= 2|3``, an interior
element table ``NELEM= n`` of VTK-typed connectivity rows, a node table ``NPOIN= m`` of
coordinates, and ``NMARK= k`` boundary markers, each a named group of line/edge elements.

The format is **0-based**: element rows index the node table directly, so -- unlike the
2DM reference driver -- there is no id map and nothing for the validator to subtract.
Points become ``node_x``/``node_y`` (and ``node_z`` for a 3D file); surface elements
(VTK triangle 5 and quadrilateral 9) become the fill-padded ``face_node_connectivity``;
the boundary markers are recorded as metadata (their tags and element counts) rather than
as a separate topology.

This reader targets surface meshes. A volume element type (VTK tetra 10, hexa 12, and the
rest of the 3D cell table) is refused rather than silently flattened, because there is no
committed 3D sample to validate such a read against.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

from typing import Iterator

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records

# SU2 uses `%` for comments, not `#`.
_COMMENT = "%"

# The section cards. A card is written ``KEY= value`` (SU2 allows the ``=`` to carry the
# value directly, ``KEY=value``, or to be followed by it, ``KEY= value``); both are
# handled by splitting the keyword on its ``=``.
_NDIME = "NDIME"
_NELEM = "NELEM"
_NPOIN = "NPOIN"
_NMARK = "NMARK"
_MARKER_TAG = "MARKER_TAG"
_MARKER_ELEMS = "MARKER_ELEMS"

# The cards whose presence marks a file as SU2, used by the header sniff.
_HEADER_CARDS = frozenset({_NDIME, _NELEM, _NPOIN})

# VTK cell codes -> node count, for the element types SU2 writes. Only the number of
# node columns is taken from this; which codes become faces is decided separately.
_VTK_NODE_COUNT = {3: 2, 5: 3, 9: 4, 10: 4, 12: 8}

# The surface (2D) element codes that become mesh faces: VTK triangle and quadrilateral.
_SURFACE_CODES = frozenset({5, 9})

# How many bytes of the head the sniff decodes looking for a section card.
_SNIFF_LIMIT = 64 * 1024


class Su2Driver(BaseDriver):
    """Reader for the SU2 native ASCII mesh (``.su2``).

    Reads the dimension card, the interior element table, the node coordinates and the
    boundary markers of an SU2 surface mesh. Triangle (VTK code 5) and quadrilateral
    (code 9) elements become the fill-padded ``face_node_connectivity``; a mixed-width
    file pads the short rows with ``-1``. Node coordinates become ``node_x``/``node_y``,
    and ``node_z`` for a 3D (``NDIME= 3``) file.

    Documented limitations:
        - **Surface meshes only.** A volume element type (VTK tetra 10, hexa 12, or any
          other 3D cell) in the ``NELEM`` table is refused rather than flattened to a
          face; there is no committed 3D sample to validate such a read against.
        - **0-based, no id map.** SU2 element rows index the node table directly, so the
          connectivity is already 0-based: ``start_index`` and ``original_start_index``
          are both ``0`` and the validator subtracts nothing. The optional trailing
          element/point index column each row may carry is ignored, as SU2 itself
          ignores it on read.
        - **Boundary markers are metadata, not topology.** Each ``MARKER_TAG`` group is
          recorded as ``su2_markers`` on the global attributes (tag -> boundary-element
          count); the marker edges are not built into a separate 1D network.
        - **No CRS.** SU2 carries no coordinate-reference declaration, so ``crs`` is
          ``None`` and is never guessed from the coordinate magnitudes.

    Examples:
        - Read the committed 5x5 quad sample: 36 nodes and 25 quad faces, with the
          0-based connectivity preserved and the boundary markers recorded as metadata.
          (Shown, not run as a doctest -- it reads a committed data file.)::

            from pyramids_mesh.drivers.su2 import Su2Driver

            dataset = Su2Driver().read("tests/data/su2/5x5Mesh.su2")
            dataset.mesh.n_node, dataset.mesh.n_face          # (36, 25)
            dataset.mesh.face_node_connectivity.data.min()    # 0
            dataset.global_attributes["su2_markers"]          # {"bottom": 5, ...}
    """

    name = "su2"
    long_name = "SU2 native mesh"
    suffixes = (".su2",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file head carries an SU2 section card.

        A bounded prefix is decoded leniently -- so a binary or empty file answers
        ``False`` rather than raising -- and scanned for a ``NDIME=``, ``NPOIN=`` or
        ``NELEM=`` card at the start of a significant (non-comment) line. Matching the
        card keyword, not a substring, keeps a file that merely mentions one of those
        words in a comment from qualifying.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when a section card is found in the sniffed prefix.
        """
        with open(path, "rb") as handle:
            prefix = handle.read(_SNIFF_LIMIT)
        for line in prefix.decode("utf-8-sig", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(_COMMENT):
                continue
            name, _value = _split_card(stripped.split()[0])
            if name in _HEADER_CARDS:
                return True
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an SU2 file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The ``.su2`` file to read.
            lazy: Ignored -- an SU2 file is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, the fill-padded face connectivity, and the boundary
                markers recorded as metadata.

        Raises:
            FileFormatNotSupportedError: If the file carries no section card, declares no
                ``NDIME``/``NPOIN``/``NELEM``, uses an unknown or volume element type, or
                a row is short of the fields its element type requires.
        """
        scan = _Scan(path)
        scan.run()
        return scan.build()


def _split_card(token: str) -> tuple[str, str | None]:
    """Split a ``KEY=value`` keyword token into its name and any inline value.

    SU2 writes a card either as ``KEY= value`` (the value a separate field) or
    ``KEY=value`` (glued on), so the inline part is returned when present and the value
    otherwise comes from the record's following fields.

    Args:
        token: The first whitespace-separated token of a line.

    Returns:
        tuple[str, str | None]: The upper-cased card name without its ``=``, and the
            inline value, or ``None`` when the token carried none.
    """
    name, _sep, inline = token.partition("=")
    return name.upper(), (inline or None)


class _Scan:
    """One SU2 file, streamed once into nodes, faces and marker metadata."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.ndime: int | None = None
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.faces: list[list[int]] = []
        self.markers: dict[str, int] = {}
        self.pending_tag: str | None = None

    def run(self) -> None:
        """Stream the file, dispatching each section card onto its reader.

        Raises:
            FileFormatNotSupportedError: On an unknown/volume element type, a short row,
                or a non-numeric field.
        """
        records = iter_records(self.path, comment=_COMMENT)
        for keyword, fields in records:
            name, inline = _split_card(keyword)
            if inline is None and name not in _CARD_READERS:
                # A bare data row reached outside the section that consumes it; the
                # section readers pull their own rows, so anything here is stray.
                continue
            value = inline if inline is not None else (fields[0] if fields else None)
            reader = _CARD_READERS.get(name)
            if reader is not None:
                reader(self, name, value, fields, records)

    def _read_ndime(self, name, value, fields, records) -> None:
        """Record the mesh dimension from an ``NDIME`` card."""
        self.ndime = _card_int(self.path, name, value)

    def _read_nelem(self, name, value, fields, records) -> None:
        """Read the ``NELEM`` count then that many VTK-typed element rows."""
        count = _card_int(self.path, name, value)
        for _ in range(count):
            tokens = _next_row(records, self.path, _NELEM)
            code = _as_int(self.path, _NELEM, tokens[0])
            node_count = _VTK_NODE_COUNT.get(code)
            if node_count is None:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} uses the VTK element code {code}, which this reader "
                    f"does not know; SU2 writes codes 3, 5, 9, 10 and 12"
                )
            if code not in _SURFACE_CODES:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} uses the volume/line element code {code}; this "
                    f"reader builds surface meshes from triangles (5) and quads (9) only"
                )
            if len(tokens) < 1 + node_count:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short NELEM row for VTK code {code}: expected "
                    f"{node_count} node ids, found {len(tokens) - 1}"
                )
            self.faces.append(
                [_as_int(self.path, _NELEM, tok) for tok in tokens[1 : 1 + node_count]]
            )

    def _read_npoin(self, name, value, fields, records) -> None:
        """Read the ``NPOIN`` count then that many coordinate rows.

        Raises:
            FileFormatNotSupportedError: If ``NDIME`` was not seen first (so the z column
                cannot be placed), or a coordinate row is short.
        """
        if self.ndime not in (2, 3):
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares NPOIN before a valid NDIME (2 or 3); the "
                f"coordinate columns cannot be read without the dimension"
            )
        count = _card_int(self.path, name, value)
        for _ in range(count):
            tokens = _next_row(records, self.path, _NPOIN)
            if len(tokens) < self.ndime:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short NPOIN row: expected {self.ndime} "
                    f"coordinates, found {len(tokens)}"
                )
            self.node_x.append(_as_float(self.path, _NPOIN, tokens[0]))
            self.node_y.append(_as_float(self.path, _NPOIN, tokens[1]))
            if self.ndime == 3:
                self.node_z.append(_as_float(self.path, _NPOIN, tokens[2]))

    def _read_nmark(self, name, value, fields, records) -> None:
        """Record the declared marker count; the groups themselves follow as cards."""
        _card_int(self.path, name, value)

    def _read_marker_tag(self, name, value, fields, records) -> None:
        """Record the current marker's tag, carried to its ``MARKER_ELEMS`` group."""
        self.pending_tag = value

    def _read_marker_elems(self, name, value, fields, records) -> None:
        """Read a marker's element count and skip its rows, recording the count.

        The boundary edges are consumed so they do not leak into the next section, but
        only the tag and count are kept -- markers are metadata here, not topology.
        """
        count = _card_int(self.path, name, value)
        for _ in range(count):
            _next_row(records, self.path, _MARKER_ELEMS)
        tag = self.pending_tag or f"marker_{len(self.markers) + 1}"
        self.markers[tag] = count
        self.pending_tag = None

    def build(self) -> ParsedMesh:
        """Assemble the parsed mesh from the collected sections.

        Returns:
            ParsedMesh: The surface mesh, already 0-based, with the markers recorded as
                ``su2_markers`` metadata.

        Raises:
            FileFormatNotSupportedError: If the file declared no dimension, no nodes, or
                no surface elements.
        """
        if self.ndime is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no NDIME, so it is not an SU2 mesh"
            )
        if not self.node_x:
            raise FileFormatNotSupportedError(f"{self.path!r} declares no NPOIN nodes")
        if not self.faces:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no surface elements (VTK triangle 5 or quad 9)"
            )
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=self._padded_faces(),
            fill_value=-1,
            # SU2 rows index the node table directly: the connectivity is already
            # 0-based, so there is nothing to subtract and a writer emits 0-based too.
            start_index=0,
            original_start_index=0,
            node_z=(
                np.array(self.node_z, dtype=np.float64) if self.ndime == 3 else None
            ),
            crs=None,
            global_attributes={
                "source": "SU2",
                "NDIME": self.ndime,
                "su2_markers": dict(self.markers),
            },
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _padded_faces(self) -> np.ndarray:
        """Stack the face rows into one array, padding mixed widths with ``-1``."""
        width = max(len(row) for row in self.faces)
        faces = np.full((len(self.faces), width), -1, dtype=np.int64)
        for row_index, row in enumerate(self.faces):
            faces[row_index, : len(row)] = row
        return faces


# Card name -> the bound reader on `_Scan` that consumes it. Built after the class so the
# methods exist; `run` looks a card up here and a card with no reader is ignored.
_CARD_READERS = {
    _NDIME: _Scan._read_ndime,
    _NELEM: _Scan._read_nelem,
    _NPOIN: _Scan._read_npoin,
    _NMARK: _Scan._read_nmark,
    _MARKER_TAG: _Scan._read_marker_tag,
    _MARKER_ELEMS: _Scan._read_marker_elems,
}


def _next_row(
    records: Iterator[tuple[str, list[str]]], path: str, section: str
) -> list[str]:
    """Pull the next record as a flat token list, or report the file ended mid-section.

    Args:
        records: The file's remaining records.
        path: The source path, for the error message.
        section: The section being read, for the error message.

    Returns:
        list[str]: The record's keyword followed by its fields.

    Raises:
        FileFormatNotSupportedError: If the record stream is exhausted mid-section.
    """
    try:
        keyword, fields = next(records)
    except StopIteration:
        raise FileFormatNotSupportedError(
            f"{path!r} ends inside the {section} section: a declared row is missing"
        ) from None
    return [keyword, *fields]


def _card_int(path: str, name: str, value: str | None) -> int:
    """Parse a section card's integer value, naming the card if it is missing or bad.

    Args:
        path: The source path, for the error message.
        name: The card name, for the error message.
        value: The card's value, or ``None`` when the card carried none.

    Returns:
        int: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the value is absent or not an integer.
    """
    if value is None:
        raise FileFormatNotSupportedError(f"{path!r} has a {name} card with no value")
    return _as_int(path, name, value)


def _as_int(path: str, section: str, value: str) -> int:
    """Parse one field as an integer, naming the section and file if it is not one.

    Args:
        path: The source path, for the error message.
        section: The section being read, for the error message.
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
            f"{path!r} has a malformed {section} field: expected an integer, found "
            f"{value!r}"
        ) from error


def _as_float(path: str, section: str, value: str) -> float:
    """Parse one field as a float, naming the section and file if it is not one.

    Args:
        path: The source path, for the error message.
        section: The section being read, for the error message.
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
            f"{path!r} has a malformed {section} field: expected a number, found "
            f"{value!r}"
        ) from error
