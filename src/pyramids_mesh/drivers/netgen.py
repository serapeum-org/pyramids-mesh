"""Netgen ``.vol`` reader (task C12).

Netgen writes its tetrahedral/triangular meshes to a flat, section-keyed ASCII file: a
``mesh3d`` preamble with a ``dimension`` and a ``geomtype`` setting, then one section per
topology kind -- each a keyword line, a count line, and that many rows. The ``.vol`` file
is read with the standard library alone (a line-record walk, optionally through
:mod:`gzip` for a ``.vol.gz`` file), so the driver declares no dependency and never
imports a mesh library.

Only the surface of the mesh is turned into a :class:`~pyramids.netcdf.ugrid.UgridDataset`:
the ``surfaceelements`` triangles become the fill-padded ``face_node_connectivity`` (the
same "keep the surface" choice the VTK reader makes for a volumetric grid), their node
coordinates come from the ``points`` section, and the ``volumeelements`` tetrahedra are
dropped rather than boundary-extracted -- a faithful volume read needs the layered mesh
model pyramids does not have yet (design section 21.1). The optional trailing sections
(``materials``, ``bcnames``, ``identifications`` and the rest of the Netgen table) are
skipped.

See ``planning/design.md`` section 20.3 for the plain-text driver family this sits beside.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import gzip
from typing import Iterator

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# Netgen comments start with `#`; blank and comment lines carry no record.
_COMMENT = "#"

# Section keywords this reader acts on. `mesh3d` is the file's opening marker, and
# `surfaceelements` the only element section that becomes faces. Everything else
# (`volumeelements`, `edgesegmentsgi2`, `materials`, `bcnames`, ...) is skipped, so its
# numeric rows fall through the dispatch as unrecognised and are ignored.
_MESH3D = "mesh3d"
_DIMENSION = "dimension"
_GEOMTYPE = "geomtype"
_SURFACE_ELEMENTS = "surfaceelements"
_POINTS = "points"

# The tokens whose presence marks a file as Netgen, used by the header sniff and the
# parser's final check. Either the `mesh3d` preamble or a `surfaceelements` section is
# enough to recognise the format.
_SIGNATURE = frozenset({_MESH3D, _SURFACE_ELEMENTS})

# A `surfaceelements` row is `surfnr bcnr domin domout np p1..pnp`: four leading columns,
# then the node count, then that many 1-based node ids. The ids are the trailing columns.
_SURFACE_LEADING = 4
_NP_COLUMN = 4

# How many bytes of the head the sniff decodes looking for the signature. Bounded so a
# large (or a decompressing) file is not read whole just to recognise it.
_SNIFF_LIMIT = 64 * 1024


class NetgenDriver(BaseDriver):
    """Reader for the Netgen ``.vol`` / ``.vol.gz`` mesh format.

    Reads the ``points`` node coordinates and the ``surfaceelements`` triangles of a
    Netgen 3D mesh into a surface :class:`~pyramids.netcdf.ugrid.UgridDataset`. Node ids
    are 1-based and contiguous, so the connectivity arrives 0-based with the base
    recorded for a writer. The node ``z`` column becomes a ``node_z`` variable, since the
    pyramids mesh model is planar and carries no per-node elevation coordinate.

    A plain ``.vol`` file is read as text; a ``.vol.gz`` file is decompressed through the
    standard-library :mod:`gzip` first. Both go through the same line-record walk.

    Documented limitations:
        - **Only the surface becomes a mesh.** The ``surfaceelements`` section (the mesh
          surface's triangles) is turned into faces; the ``volumeelements`` tetrahedra
          are dropped rather than boundary-extracted. A faithful volume read needs the
          layered mesh model pyramids does not have yet (design section 21.1), and
          reconstructing the boundary from the tetrahedra could not be verified against a
          sample, so the surface the file already lists is kept instead.
        - **The optional trailing sections are metadata, not topology.** ``materials``,
          ``bcnames``, ``identifications``, ``edgesegmentsgi2`` and the rest of the Netgen
          table are skipped; only ``points`` and ``surfaceelements`` are read.
        - **The node ``z`` column is carried as a ``node_z`` variable**, not as a
          coordinate -- the mesh model is planar. Unlike the 2DM and VTK readers, a flat
          ``z = 0`` is kept rather than omitted: Netgen is a 3D format whose points always
          declare a real elevation.
        - Netgen carries no CRS, so ``crs`` is always ``None``; it is never guessed from
          the coordinate magnitudes.

    Examples:
        - Read the committed Netgen sample: 65 nodes and 108 surface triangles, with the
          1-based connectivity arriving 0-based and the file's base recorded for a writer.
          (Shown, not run as a doctest -- it reads a committed data file.)::

            from pyramids_mesh.drivers.netgen import NetgenDriver

            dataset = NetgenDriver().read("tests/data/netgen/periodic_3d.vol")
            dataset.mesh.n_node, dataset.mesh.n_face              # (65, 108)
            dataset.mesh.face_node_connectivity.original_start_index  # 1
            dataset.get("node_z").location                       # 'node'
    """

    name = "netgen"
    long_name = "Netgen volume mesh"
    suffixes = (".vol", ".vol.gz")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file head carries the Netgen signature.

        A bounded prefix is read -- through :mod:`gzip` for a ``.vol.gz`` file -- and
        decoded leniently, so a binary, empty or truncated file answers ``False`` rather
        than raising. The prefix is scanned for a standalone ``mesh3d`` or
        ``surfaceelements`` line; matching the keyword rather than a substring keeps a
        file that merely mentions one in a comment from qualifying.

        This never raises: a decompression or decoding failure is reported as a decline,
        so a corrupt candidate cannot abort driver dispatch.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the signature is found in the sniffed prefix.
        """
        try:
            with _open_text(path) as handle:
                read = 0
                for line in handle:
                    read += len(line)
                    if read > _SNIFF_LIMIT:
                        break
                    stripped = line.strip()
                    if not stripped or stripped.startswith(_COMMENT):
                        continue
                    if stripped.split()[0].lower() in _SIGNATURE:
                        return True
        except Exception:  # noqa: BLE001 - the contract is never to raise
            # A non-gzip file under a `.gz` name, a corrupt stream, or any decode trouble
            # is a decline, not an error: `can_open` must answer for every candidate.
            return False
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Netgen ``.vol`` file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The ``.vol`` / ``.vol.gz`` file to read.
            lazy: Ignored -- a Netgen file is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, the fill-padded surface-triangle connectivity, and the
                node elevation as a ``node_z`` variable.

        Raises:
            FileFormatNotSupportedError: If the file carries no Netgen signature, declares
                no ``points`` or ``surfaceelements`` section, a section's count line is not
                an integer, a row is short of the columns it declares, or a numeric field
                is not a number.
        """
        scan = _Scan(path)
        scan.run()
        return scan.build()


def _open_text(path: str):
    """Open a Netgen file for text reading, decompressing a ``.vol.gz`` file.

    Both openers decode as ``utf-8-sig`` -- dropping a leading byte-order mark a Windows
    editor may have written -- and replace undecodable bytes rather than raising.

    Args:
        path: The ``.vol`` or ``.vol.gz`` file.

    Returns:
        A text-mode file handle, to be used as a context manager.
    """
    opener = gzip.open if path.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", errors="replace")


def _iter_records(path: str) -> Iterator[tuple[str, list[str]]]:
    """Yield one ``(keyword, fields)`` pair per significant line of a Netgen file.

    A significant line is one that is neither blank nor a comment. The line is split on
    runs of whitespace; the first token is the keyword and the rest the fields. This is
    the gzip-aware sibling of :func:`pyramids_mesh._text.iter_records`, used because a
    Netgen file may arrive compressed.

    Args:
        path: The ``.vol`` / ``.vol.gz`` file to read.

    Yields:
        tuple[str, list[str]]: The keyword and the remaining fields of each line.
    """
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(_COMMENT):
                continue
            tokens = stripped.split()
            yield tokens[0], tokens[1:]


class _Scan:
    """One Netgen file, streamed once into nodes, surface faces and preamble metadata."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.faces: list[list[int]] = []
        self.dimension: int | None = None
        self.geomtype: int | None = None
        self.saw_signature = False

    def run(self) -> None:
        """Stream the file, dispatching each section keyword onto its reader.

        The preamble settings (``dimension``, ``geomtype``) carry their value on the same
        line or the next; the element sections carry a count line then that many rows. A
        keyword this reader does not act on is ignored, and its numeric rows fall through
        the dispatch unmatched.

        Raises:
            FileFormatNotSupportedError: On a non-integer count, a short row, or a
                non-numeric field.
        """
        records = _iter_records(self.path)
        for keyword, fields in records:
            lowered = keyword.lower()
            if lowered in _SIGNATURE:
                self.saw_signature = True
            if lowered == _DIMENSION:
                self.dimension = _maybe_int(self._scalar(fields, records))
            elif lowered == _GEOMTYPE:
                self.geomtype = _maybe_int(self._scalar(fields, records))
            elif lowered == _SURFACE_ELEMENTS:
                self._read_surface_elements(records)
            elif lowered == _POINTS:
                self._read_points(records)

    def _scalar(
        self, fields: list[str], records: Iterator[tuple[str, list[str]]]
    ) -> str | None:
        """Return a preamble setting's value, inline on its line or on the next.

        Netgen writes ``dimension 3`` on one line in some files and ``dimension`` then
        ``3`` on the next in others; both are accepted.

        Args:
            fields: The keyword line's remaining fields.
            records: The file's remaining records, to pull the value line from.

        Returns:
            str | None: The value token, or ``None`` at end of file.
        """
        if fields:
            return fields[0]
        try:
            keyword, _ = next(records)
        except StopIteration:
            return None
        return keyword

    def _count(self, records: Iterator[tuple[str, list[str]]], section: str) -> int:
        """Read a section's count line, the single integer before its rows.

        Args:
            records: The file's remaining records.
            section: The section being read, for the error message.

        Returns:
            int: The declared row count.

        Raises:
            FileFormatNotSupportedError: If the file ends before the count, or the count
                is not an integer.
        """
        tokens = _next_row(records, self.path, section)
        return _as_int(self.path, section, tokens[0])

    def _read_surface_elements(self, records: Iterator[tuple[str, list[str]]]) -> None:
        """Read the ``surfaceelements`` count then that many triangle rows.

        Each row is ``surfnr bcnr domin domout np p1..pnp``; the trailing ``np`` columns
        are the 1-based node ids kept as a face. The leading columns (surface number,
        boundary condition, the two adjacent domains) are read past.

        Raises:
            FileFormatNotSupportedError: On a short row or a non-integer field.
        """
        count = self._count(records, _SURFACE_ELEMENTS)
        for _ in range(count):
            tokens = _next_row(records, self.path, _SURFACE_ELEMENTS)
            if len(tokens) < _SURFACE_LEADING + 1:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short {_SURFACE_ELEMENTS} row: expected at "
                    f"least {_SURFACE_LEADING + 1} columns, found {len(tokens)}"
                )
            np_count = _as_int(self.path, _SURFACE_ELEMENTS, tokens[_NP_COLUMN])
            start = _SURFACE_LEADING + 1
            if len(tokens) < start + np_count:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short {_SURFACE_ELEMENTS} row: declares "
                    f"np {np_count} but carries {len(tokens) - start} node id(s)"
                )
            self.faces.append(
                [
                    _as_int(self.path, _SURFACE_ELEMENTS, tok)
                    for tok in tokens[start : start + np_count]
                ]
            )

    def _read_points(self, records: Iterator[tuple[str, list[str]]]) -> None:
        """Read the ``points`` count then that many ``x y z`` coordinate rows.

        Raises:
            FileFormatNotSupportedError: On a short row or a non-numeric coordinate.
        """
        count = self._count(records, _POINTS)
        for _ in range(count):
            tokens = _next_row(records, self.path, _POINTS)
            if len(tokens) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short {_POINTS} row: expected 3 coordinates, "
                    f"found {len(tokens)}"
                )
            self.node_x.append(_as_float(self.path, _POINTS, tokens[0]))
            self.node_y.append(_as_float(self.path, _POINTS, tokens[1]))
            self.node_z.append(_as_float(self.path, _POINTS, tokens[2]))

    def build(self) -> ParsedMesh:
        """Assemble the parsed mesh from the collected sections.

        Returns:
            ParsedMesh: The surface mesh, 1-based connectivity recorded for a writer, with
                the node elevation carried as ``node_z``.

        Raises:
            FileFormatNotSupportedError: If the file carried no Netgen signature, declared
                no ``points``, or declared no ``surfaceelements``.
        """
        if not self.saw_signature:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a Netgen file: it carries neither a {_MESH3D!r} "
                f"preamble nor a {_SURFACE_ELEMENTS!r} section"
            )
        if not self.node_x:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no {_POINTS} nodes"
            )
        if not self.faces:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no {_SURFACE_ELEMENTS} surface triangles"
            )
        attributes: dict[str, object] = {"source": "Netgen"}
        if self.dimension is not None:
            attributes["dimension"] = self.dimension
        if self.geomtype is not None:
            attributes["geomtype"] = self.geomtype
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=self._padded_faces(),
            fill_value=-1,
            # Node ids are 1-based and contiguous, so the validator subtracts the base;
            # the file's own 1-based numbering is recorded for a writer to restore.
            start_index=1,
            original_start_index=1,
            node_z=np.array(self.node_z, dtype=np.float64),
            crs=None,
            global_attributes=attributes,
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _padded_faces(self) -> np.ndarray:
        """Stack the surface rows into one array, padding mixed widths with ``-1``.

        Netgen surface elements are triangles, so the array is usually a clean ``(n, 3)``;
        padding keeps a file mixing triangle widths rectangular rather than ragged.
        """
        width = max(len(row) for row in self.faces)
        faces = np.full((len(self.faces), width), -1, dtype=np.int64)
        for row_index, row in enumerate(self.faces):
            faces[row_index, : len(row)] = row
        return faces


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


def _maybe_int(value: str | None) -> int | None:
    """Parse a preamble setting as an int, or return ``None`` when it is absent or not one.

    A bad ``dimension`` or ``geomtype`` is metadata, not structure, so it is dropped
    rather than failing the read.

    Args:
        value: The setting's value token, or ``None``.

    Returns:
        int | None: The parsed integer, or ``None``.
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


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
