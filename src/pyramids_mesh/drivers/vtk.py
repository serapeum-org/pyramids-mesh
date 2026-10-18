"""VTK / VTU reader (task C4).

Reads the VTK unstructured-grid format in both of its spellings: the legacy ``.vtk``
ASCII file (``# vtk DataFile Version`` magic, ``DATASET UNSTRUCTURED_GRID``) and the XML
``.vtu`` file (``<VTKFile type="UnstructuredGrid">``). Both are parsed with the standard
library alone -- a text/token walk for the legacy form and ``xml.etree.ElementTree`` for
the XML form -- so the driver declares no dependency and never imports ``vtk``,
``meshio`` or any other mesh library.

Only the 2D surface cells of a grid are turned into a mesh: VTK triangle (type ``5``) and
quadrilateral (type ``9``), plus the quadratic triangle (``22``) and quadratic quad
(``23``) reduced to their corner face. Volumetric cells -- tetrahedron (``10``),
hexahedron (``12``), quadratic tetrahedron (``24``) and the rest -- carry a vertical
structure this package does not model yet (design section 21.1); they are dropped, and a
grid with no surface cells at all is refused rather than returned empty.

See ``planning/design.md`` section 20.3 for the XML-index family this sits beside.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# The legacy ASCII magic and the XML root marker the sniff looks for.
_LEGACY_MAGIC = b"# vtk DataFile Version"
_XML_ROOT = "VTKFile"
_UNSTRUCTURED = "UnstructuredGrid"

# Only an ``ascii`` data encoding is read; ``appended`` and ``binary`` base64 payloads
# are refused for v1 (documented) rather than silently mis-read.
_ASCII = "ascii"

# VTK cell type codes this reader recognises as a 2D surface cell, mapped to the number
# of corner nodes that make up the planar face. A quadratic cell (22/23) lists its
# corners first, then its mid-edge nodes, so taking the leading corners linearises it
# without distorting the corner positions -- and keeps a planar polygon rather than a
# self-overlapping ring of corner-plus-midside vertices.
_SURFACE_CORNERS = {5: 3, 9: 4, 22: 3, 23: 4}


class VtkDriver(BaseDriver):
    """Reader for the VTK / VTU unstructured-grid mesh format.

    The legacy ``.vtk`` file is plain ASCII: a magic line, a title, the ``ASCII`` marker,
    ``DATASET UNSTRUCTURED_GRID``, then ``POINTS``, a ``CELLS`` block (either the classic
    ``count id...`` rows or the VTK 5.1 ``OFFSETS``/``CONNECTIVITY`` pair), ``CELL_TYPES``
    and optional ``POINT_DATA``/``CELL_DATA`` attribute sections. The XML ``.vtu`` file
    carries the same information as a ``<Piece>`` with ``<Points>``, ``<Cells>``
    (``connectivity``/``offsets``/``types`` DataArrays) and ``<PointData>``/``<CellData>``.

    Documented limitations:
        - **Only 2D surface cells become a mesh** -- triangle (``5``), quad (``9``), and
          the quadratic triangle (``22``) / quadratic quad (``23``) reduced to their
          corner face. A volumetric cell (``10``/``12``/``24``/...) is dropped, and a grid
          with no surface cell at all is refused rather than returned as an empty mesh.
        - **Only ASCII DataArrays are read.** An ``appended`` or ``binary`` (base64)
          ``.vtu`` payload is refused for v1, documented, rather than mis-decoded.
        - **Only single-component ``SCALARS`` / DataArrays become variables.** A
          multi-component scalar, and the ``VECTORS``/``NORMALS``/``TENSORS``/``FIELD``
          attribute kinds, are consumed so parsing stays aligned but left out of the
          result -- the 2D scalar case ships first, as the sibling XML-index driver does.
        - **Point ``(x, y, z)`` with any non-zero z contributes a ``node_z`` variable.** A
          grid flat at ``z = 0`` gets no ``node_z`` -- a column of zeros could not be told
          from a real datum, the same reason the 2DM reader omits it.
        - **Values are eager.** VTK is read as plain text/XML with no record index to seek,
          so there is nothing to defer and the driver does not declare ``lazy_data``.
        - There is no CRS in the format, so ``crs`` is ``None``, never guessed.

    Examples:
        - A legacy ASCII ``.vtk`` of two triangles reads into a 2D surface mesh, its
          connectivity already 0-based:

            >>> import os
            >>> import tempfile
            >>> import warnings
            >>> from pyramids_mesh.drivers.vtk import VtkDriver
            >>> text = '''# vtk DataFile Version 3.0
            ... example
            ... ASCII
            ... DATASET UNSTRUCTURED_GRID
            ... POINTS 4 float
            ... 0 0 0 1 0 0 1 1 0 0 1 0
            ... CELLS 2 8
            ... 3 0 1 2
            ... 3 0 2 3
            ... CELL_TYPES 2
            ... 5 5
            ... '''
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "tris.vtk")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write(text)
            ...     with warnings.catch_warnings():
            ...         warnings.simplefilter("ignore")
            ...         dataset = VtkDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.original_start_index
            0

        - The suffix is claimed and the content sniffed, so a file that is not a VTK
          unstructured grid is declined even under a ``.vtk`` name:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.vtk import VtkDriver
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "junk.vtk")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write("not a vtk file at all")
            ...     VtkDriver().can_open(path)
            False
    """

    name = "vtk"
    long_name = "VTK / VTU unstructured grid"
    suffixes = (".vtk", ".vtu")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a VTK or VTU unstructured grid.

        Reads only the head of the file: the legacy form is recognised by its
        ``# vtk DataFile Version`` magic, and the XML form by a ``VTKFile`` root carrying
        the ``UnstructuredGrid`` type. A foreign or non-VTK file is declined cheaply
        without parsing the whole document.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` for a legacy VTK magic, or an XML ``VTKFile`` of
                ``UnstructuredGrid`` type.
        """
        with open(path, "rb") as handle:
            head = handle.read(4096)
        if _LEGACY_MAGIC in head:
            return True
        return _XML_ROOT.encode() in head and _UNSTRUCTURED.encode() in head

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a VTK or VTU file into a :class:`ParsedMesh`.

        The spelling is chosen from the content, not the suffix: a file beginning with
        the legacy magic is read as ASCII VTK, otherwise as XML VTU.

        Args:
            path: The ``.vtk`` / ``.vtu`` file to parse.
            lazy: Ignored. VTK is plain text/XML read eagerly; the driver does not declare
                ``lazy_data``.

        Returns:
            ParsedMesh: The 2D surface mesh, ``node_z`` for a non-flat grid, and the
                single-component point/cell scalars as variables.

        Raises:
            FileFormatNotSupportedError: If the file is neither spelling, uses a non-ASCII
                payload, declares no ``UNSTRUCTURED_GRID`` dataset, or carries no 2D
                surface cell to build a mesh from.
        """
        with open(path, "rb") as handle:
            head = handle.read(len(_LEGACY_MAGIC) + 8)
        if _LEGACY_MAGIC in head:
            raw = _LegacyReader(path).read()
        else:
            raw = _read_vtu(path)
        return _raw_to_parsed(raw, path)


@dataclass
class _RawMesh:
    """A VTK grid as parsed, before surface cells are selected and the mesh built.

    This is the common intermediate both spellings produce, so the surface-cell
    selection and variable construction in :func:`_raw_to_parsed` are written once.

    Attributes:
        points: Node coordinates, shape ``(n_node, 3)``.
        cells: One array of node indices per cell, in file order, every cell type kept.
        cell_types: The VTK cell type code of each cell, aligned with ``cells``.
        point_data: Single-component point scalars, keyed by name, each length ``n_node``.
        cell_data: Single-component cell scalars, keyed by name, each length ``n_cell``.
    """

    points: np.ndarray
    cells: list[np.ndarray]
    cell_types: np.ndarray
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)


def _cells_from_offsets(
    connectivity: np.ndarray, boundaries: np.ndarray
) -> list[np.ndarray]:
    """Split a flat connectivity array into cells at the given boundary offsets.

    ``boundaries`` holds the ``n_cell + 1`` cumulative indices, starting at ``0`` and
    ending at ``len(connectivity)``, so cell ``i`` is
    ``connectivity[boundaries[i]:boundaries[i + 1]]``. The two offset conventions -- the
    legacy 5.1 ``OFFSETS`` array (already ``n_cell + 1`` with a leading ``0``) and the
    XML ``offsets`` array (``n_cell`` end offsets) -- are both normalised to this form by
    the caller.

    Args:
        connectivity: The flat node-index stream.
        boundaries: The cumulative cell boundaries, length ``n_cell + 1``.

    Returns:
        list[np.ndarray]: One node-index array per cell.
    """
    return [
        np.asarray(connectivity[boundaries[i] : boundaries[i + 1]], dtype=np.int64)
        for i in range(len(boundaries) - 1)
    ]


def _raw_to_parsed(raw: _RawMesh, source: str) -> ParsedMesh:
    """Select the surface cells of a parsed grid and build a :class:`ParsedMesh`.

    Args:
        raw: The parsed grid, every cell type still present.
        source: The source path, for the error message.

    Returns:
        ParsedMesh: The 2D surface mesh, with the point scalars as node variables and the
            surface cells' slice of the cell scalars as face variables.

    Raises:
        FileFormatNotSupportedError: If the grid holds no 2D surface cell.
    """
    surface = [
        (index, int(code))
        for index, code in enumerate(raw.cell_types)
        if int(code) in _SURFACE_CORNERS
    ]
    if not surface:
        raise FileFormatNotSupportedError(
            f"{source!r} holds no 2D surface cell (triangle/quad); a volumetric or "
            f"line-only VTK grid is refused rather than built as an empty mesh"
        )
    faces = _build_faces(raw.cells, surface)
    points = np.asarray(raw.points, dtype=np.float64)
    node_z = points[:, 2] if points.shape[1] > 2 and np.any(points[:, 2]) else None
    variables = _build_variables(raw, [index for index, _ in surface], len(faces))
    return ParsedMesh(
        node_x=points[:, 0].copy(),
        node_y=points[:, 1].copy(),
        faces=faces,
        fill_value=-1,
        start_index=0,
        original_start_index=0,  # VTK connectivity is 0-based
        node_z=None if node_z is None else node_z.copy(),
        crs=None,
        variables=variables,
        global_attributes={"source": "VTK"},
        topology_dimension=2,
        mesh_name="mesh2d",
    )


def _build_faces(cells: list[np.ndarray], surface: list[tuple[int, int]]) -> np.ndarray:
    """Build the fill-padded face-node connectivity from the selected surface cells.

    Each surface cell contributes its leading corner nodes (3 for a triangle, 4 for a
    quad); rows are padded with ``-1`` to the widest face.

    Args:
        cells: Every parsed cell's node indices.
        surface: The ``(cell index, type code)`` pairs of the surface cells.

    Returns:
        np.ndarray: The ``(n_face, max_corners)`` connectivity, ``-1``-padded.
    """
    corner_counts = [_SURFACE_CORNERS[code] for _, code in surface]
    width = max(corner_counts)
    faces = np.full((len(surface), width), -1, dtype=np.int64)
    for row, ((index, _code), corners) in enumerate(zip(surface, corner_counts)):
        faces[row, :corners] = cells[index][:corners]
    return faces


def _build_variables(
    raw: _RawMesh, surface_indices: list[int], n_face: int
) -> dict[str, ParsedVariable]:
    """Turn the single-component point and cell scalars into parsed variables.

    Point scalars are node-located and carried whole. Cell scalars are face-located and
    sliced to the surface cells, so a value survives for exactly the cells that became
    faces -- a grid mixing surface and volumetric cells keeps the right column.

    Args:
        raw: The parsed grid carrying the scalar arrays.
        surface_indices: The positions, among all cells, of the cells kept as faces.
        n_face: The face count, for the cell-variable shape.

    Returns:
        dict[str, ParsedVariable]: The point scalars as node variables and the cell
            scalars as face variables, keyed by name.
    """
    variables: dict[str, ParsedVariable] = {}
    n_node = len(raw.points)
    for name, values in raw.point_data.items():
        variables[name] = ParsedVariable(
            location="node", shape=(n_node,), values=np.asarray(values)
        )
    for name, values in raw.cell_data.items():
        column = np.asarray(values)[surface_indices]
        variables[name] = ParsedVariable(
            location="face", shape=(n_face,), values=column
        )
    return variables


def _read_vtu(path: str) -> _RawMesh:
    """Read an XML ``.vtu`` unstructured grid into a :class:`_RawMesh`.

    Args:
        path: The ``.vtu`` file.

    Returns:
        _RawMesh: The parsed grid.

    Raises:
        FileFormatNotSupportedError: If the root is not a ``VTKFile`` of
            ``UnstructuredGrid`` type, a required element is missing, or any DataArray
            uses a non-ASCII (``appended``/``binary``) payload.
    """
    root = ET.parse(path).getroot()
    if root.tag != _XML_ROOT or root.get("type") != _UNSTRUCTURED:
        raise FileFormatNotSupportedError(
            f"{path!r} is not a VTKFile of type {_UNSTRUCTURED!r}"
        )
    piece = _require_child(_require_child(root, _UNSTRUCTURED, path), "Piece", path)
    points = _vtu_points(piece, path)
    cells = _vtu_cells(piece, path)
    cell_types = cells[1]
    return _RawMesh(
        points=points,
        cells=cells[0],
        cell_types=cell_types,
        point_data=_vtu_attributes(piece.find("PointData"), path),
        cell_data=_vtu_attributes(piece.find("CellData"), path),
    )


def _require_child(element: ET.Element, tag: str, path: str) -> ET.Element:
    """Return a required child element, or refuse the file naming what is missing.

    Args:
        element: The parent element.
        tag: The child tag that must be present.
        path: The source path, for the error message.

    Returns:
        ET.Element: The child element.

    Raises:
        FileFormatNotSupportedError: If the child is absent.
    """
    child = element.find(tag)
    if child is None:
        raise FileFormatNotSupportedError(f"{path!r} has no <{tag}> element")
    return child


def _vtu_points(piece: ET.Element, path: str) -> np.ndarray:
    """Read the ``<Points>`` DataArray into an ``(n_node, 3)`` coordinate array.

    A 2-component geometry is padded with a zero z-column so the array is always 3-wide.

    Args:
        piece: The ``<Piece>`` element.
        path: The source path, for error messages.

    Returns:
        np.ndarray: The node coordinates, shape ``(n_node, 3)``.
    """
    data_array = _require_child(
        _require_child(piece, "Points", path), "DataArray", path
    )
    _require_ascii(data_array, path)
    components = int(data_array.get("NumberOfComponents", "3"))
    coords = _floats(data_array.text).reshape(-1, components)
    if components == 3:
        return coords
    padded = np.zeros((coords.shape[0], 3), dtype=np.float64)
    padded[:, :components] = coords
    return padded


def _vtu_cells(piece: ET.Element, path: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Read the ``<Cells>`` connectivity/offsets/types DataArrays into cells.

    The XML ``offsets`` array holds the ``n_cell`` end offsets, so a leading ``0`` is
    prepended to turn it into the ``n_cell + 1`` boundaries :func:`_cells_from_offsets`
    expects.

    Args:
        piece: The ``<Piece>`` element.
        path: The source path, for error messages.

    Returns:
        tuple[list[np.ndarray], np.ndarray]: The cells and their type codes.

    Raises:
        FileFormatNotSupportedError: If a required DataArray is missing or non-ASCII.
    """
    cells = _require_child(piece, "Cells", path)
    by_name = {array.get("Name"): array for array in cells.findall("DataArray")}
    for name in ("connectivity", "offsets", "types"):
        if name not in by_name:
            raise FileFormatNotSupportedError(
                f"{path!r} <Cells> has no {name!r} DataArray"
            )
        _require_ascii(by_name[name], path)
    connectivity = _ints(by_name["connectivity"].text)
    offsets = _ints(by_name["offsets"].text)
    boundaries = np.concatenate(([0], offsets))
    cell_list = _cells_from_offsets(connectivity, boundaries)
    return cell_list, _ints(by_name["types"].text)


def _vtu_attributes(section: ET.Element | None, path: str) -> dict[str, np.ndarray]:
    """Read a ``<PointData>`` / ``<CellData>`` section's single-component scalars.

    A multi-component DataArray is skipped (the 2D scalar case ships first), and a
    non-ASCII one refuses the whole file rather than being mis-decoded.

    Args:
        section: The ``<PointData>`` or ``<CellData>`` element, or ``None`` when absent.
        path: The source path, for error messages.

    Returns:
        dict[str, np.ndarray]: The single-component scalars, keyed by DataArray name.

    Raises:
        FileFormatNotSupportedError: If a DataArray uses a non-ASCII payload.
    """
    data: dict[str, np.ndarray] = {}
    if section is None:
        return data
    for data_array in section.findall("DataArray"):
        _require_ascii(data_array, path)
        if int(data_array.get("NumberOfComponents", "1")) != 1:
            continue
        name = data_array.get("Name")
        if name:
            data[name] = _floats(data_array.text)
    return data


def _require_ascii(data_array: ET.Element, path: str) -> None:
    """Refuse a DataArray whose format is not ``ascii``.

    Args:
        data_array: The ``<DataArray>`` element.
        path: The source path, for the error message.

    Raises:
        FileFormatNotSupportedError: If the ``format`` is ``appended`` or ``binary``.
    """
    fmt = (data_array.get("format") or _ASCII).lower()
    if fmt != _ASCII:
        raise FileFormatNotSupportedError(
            f"{path!r} has a {fmt!r} DataArray; only ascii VTU payloads are read "
            f"(appended/binary base64 is a documented v1 limitation)"
        )


def _floats(text: str | None) -> np.ndarray:
    """Parse whitespace-separated text into a float64 array."""
    return np.array((text or "").split(), dtype=np.float64)


def _ints(text: str | None) -> np.ndarray:
    """Parse whitespace-separated text into an int64 array."""
    return np.array((text or "").split(), dtype=np.int64)


class _Cursor:
    """A forward cursor over a flat token list, with count-driven numeric reads."""

    def __init__(self, tokens: list[str], path: str) -> None:
        self.tokens = tokens
        self.path = path
        self.index = 0

    def done(self) -> bool:
        """Whether every token has been consumed."""
        return self.index >= len(self.tokens)

    def peek(self) -> str:
        """Return the next token without consuming it, or ``""`` at the end."""
        return "" if self.done() else self.tokens[self.index]

    def take(self) -> str:
        """Consume and return the next token, refusing a truncated file."""
        if self.done():
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends mid-record: the VTK stream is truncated"
            )
        token = self.tokens[self.index]
        self.index += 1
        return token

    def take_int(self) -> int:
        """Consume the next token as an int, refusing a non-numeric one."""
        token = self.take()
        try:
            return int(token)
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} expected an integer, found {token!r}"
            ) from error

    def take_block(self, count: int, dtype) -> np.ndarray:
        """Consume ``count`` tokens and return them as an array of ``dtype``.

        Args:
            count: The number of tokens to consume.
            dtype: The numpy dtype to read them as.

        Returns:
            np.ndarray: The parsed values.

        Raises:
            FileFormatNotSupportedError: If fewer than ``count`` tokens remain, or one is
                not a number.
        """
        end = self.index + count
        if end > len(self.tokens):
            raise FileFormatNotSupportedError(
                f"{self.path!r} is truncated: expected {count} more value(s)"
            )
        chunk = self.tokens[self.index : end]
        self.index = end
        try:
            return np.array(chunk, dtype=dtype)
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a non-numeric value in a data block"
            ) from error


class _LegacyReader:
    """One legacy ASCII ``.vtk`` file: the token walk that fills a :class:`_RawMesh`."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.points: np.ndarray | None = None
        self.cells: list[np.ndarray] = []
        self.cell_types: np.ndarray | None = None
        self.point_data: dict[str, np.ndarray] = {}
        self.cell_data: dict[str, np.ndarray] = {}
        # The active attribute section: "node" after POINT_DATA, "face" after CELL_DATA,
        # with the element count the section declared.
        self.location: str | None = None
        self.count = 0

    def read(self) -> _RawMesh:
        """Parse the file and return the raw grid.

        Returns:
            _RawMesh: The parsed points, cells, types and single-component scalars.

        Raises:
            FileFormatNotSupportedError: On a non-ASCII encoding, a dataset that is not
                ``UNSTRUCTURED_GRID``, a missing POINTS/CELLS/CELL_TYPES block, or an
                attribute kind this reader does not read.
        """
        with open(self.path, "r", encoding="utf-8-sig", errors="replace") as handle:
            lines = handle.read().splitlines()
        # Line 1 is the magic and line 2 is a free-text title that may itself contain a
        # keyword word, so both are dropped by line before the rest is tokenised.
        body = " ".join(lines[2:]).split()
        cursor = _Cursor(body, self.path)
        self._header(cursor)
        self._walk(cursor)
        if self.points is None or self.cell_types is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is missing a POINTS or CELL_TYPES block"
            )
        return _RawMesh(
            points=self.points,
            cells=self.cells,
            cell_types=self.cell_types,
            point_data=self.point_data,
            cell_data=self.cell_data,
        )

    def _header(self, cursor: _Cursor) -> None:
        """Consume the ``ASCII`` marker and the ``DATASET UNSTRUCTURED_GRID`` line.

        Raises:
            FileFormatNotSupportedError: On a ``BINARY`` encoding or a dataset type other
                than ``UNSTRUCTURED_GRID``.
        """
        encoding = cursor.take().upper()
        if encoding != "ASCII":
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a {encoding!r} VTK file; only ASCII is read "
                f"(binary is a documented v1 limitation)"
            )
        if cursor.take().upper() != "DATASET":
            raise FileFormatNotSupportedError(f"{self.path!r} has no DATASET line")
        dataset = cursor.take().upper()
        if dataset != "UNSTRUCTURED_GRID":
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a {dataset!r} dataset; only UNSTRUCTURED_GRID is read"
            )

    def _walk(self, cursor: _Cursor) -> None:
        """Dispatch each top-level keyword until the token stream is exhausted."""
        while not cursor.done():
            keyword = cursor.take().upper()
            if keyword == "POINTS":
                self._points(cursor)
            elif keyword == "CELLS":
                self._cells(cursor)
            elif keyword == "CELL_TYPES":
                self._cell_types(cursor)
            elif keyword == "POINT_DATA":
                self.location, self.count = "node", cursor.take_int()
            elif keyword == "CELL_DATA":
                self.location, self.count = "face", cursor.take_int()
            elif keyword == "SCALARS":
                self._scalars(cursor)
            else:
                self._skip_attribute(keyword, cursor)

    def _points(self, cursor: _Cursor) -> None:
        """Read ``POINTS n datatype`` and the ``n`` xyz triples that follow it."""
        n_point = cursor.take_int()
        cursor.take()  # the datatype word (float/double/...), not needed
        self.points = cursor.take_block(3 * n_point, np.float64).reshape(n_point, 3)

    def _cells(self, cursor: _Cursor) -> None:
        """Read a ``CELLS`` block, in either the classic or the VTK 5.1 spelling.

        The classic form is ``CELLS n total`` then one ``count id...`` row per cell. The
        5.1 form is ``CELLS n_offsets n_conn`` then an ``OFFSETS`` array and a
        ``CONNECTIVITY`` array, distinguished by the ``OFFSETS`` keyword that follows the
        two counts.
        """
        first = cursor.take_int()
        second = cursor.take_int()
        if cursor.peek().upper() == "OFFSETS":
            cursor.take()  # OFFSETS
            cursor.take()  # its datatype word
            offsets = cursor.take_block(first, np.int64)
            if cursor.take().upper() != "CONNECTIVITY":
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has OFFSETS without a CONNECTIVITY array"
                )
            cursor.take()  # its datatype word
            connectivity = cursor.take_block(second, np.int64)
            self.cells = _cells_from_offsets(connectivity, offsets)
            return
        for _ in range(first):
            size = cursor.take_int()
            self.cells.append(cursor.take_block(size, np.int64))

    def _cell_types(self, cursor: _Cursor) -> None:
        """Read ``CELL_TYPES n`` and the ``n`` type codes."""
        n_cell = cursor.take_int()
        self.cell_types = cursor.take_block(n_cell, np.int64)

    def _scalars(self, cursor: _Cursor) -> None:
        """Read a ``SCALARS name datatype [numComp]`` array and its lookup table.

        A single-component scalar is kept as a variable on the active section's location;
        a multi-component one is consumed and skipped (documented v1 limitation).

        Raises:
            FileFormatNotSupportedError: If a SCALARS block appears before any
                POINT_DATA/CELL_DATA section has set the element count.
        """
        if self.location is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a SCALARS block outside POINT_DATA/CELL_DATA"
            )
        name = cursor.take()
        cursor.take()  # the datatype word
        components = 1
        if cursor.peek().upper() != "LOOKUP_TABLE":
            components = cursor.take_int()
        cursor.take()  # LOOKUP_TABLE
        cursor.take()  # the lookup-table name
        values = cursor.take_block(self.count * components, np.float64)
        if components == 1:
            target = self.point_data if self.location == "node" else self.cell_data
            target[name] = values

    def _skip_attribute(self, keyword: str, cursor: _Cursor) -> None:
        """Consume a non-scalar attribute array so the walk stays aligned.

        ``VECTORS``/``NORMALS`` carry three components per element, ``TENSORS`` nine,
        ``COLOR_SCALARS`` a declared number, and ``FIELD`` a list of sub-arrays. Each is
        read past and dropped (the 2D scalar case ships first). A keyword this reader does
        not know is refused rather than mis-parsed.

        Raises:
            FileFormatNotSupportedError: On an unknown keyword, or a known attribute that
                appears before any POINT_DATA/CELL_DATA section.
        """
        if keyword == "FIELD":
            self._skip_field(cursor)
            return
        fixed = {"VECTORS": 3, "NORMALS": 3, "TENSORS": 9}
        if keyword in fixed or keyword == "COLOR_SCALARS":
            if self.location is None:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a {keyword} block outside POINT_DATA/CELL_DATA"
                )
            cursor.take()  # the array name
            per = fixed[keyword] if keyword in fixed else cursor.take_int()
            if keyword in fixed:
                cursor.take()  # the datatype word
            cursor.take_block(self.count * per, np.float64)
            return
        raise FileFormatNotSupportedError(
            f"{self.path!r} uses the {keyword!r} section, which this reader does not read"
        )

    def _skip_field(self, cursor: _Cursor) -> None:
        """Consume a ``FIELD name numArrays`` block and all of its sub-arrays."""
        cursor.take()  # the field name
        for _ in range(cursor.take_int()):
            cursor.take()  # the sub-array name
            components = cursor.take_int()
            tuples = cursor.take_int()
            cursor.take()  # the datatype word
            cursor.take_block(components * tuples, np.float64)
