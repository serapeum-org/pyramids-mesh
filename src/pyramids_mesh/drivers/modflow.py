"""MODFLOW 6 DISV reader (task C5).

MODFLOW 6's DISV package describes a *discretization by vertices*: a layered groundwater
grid whose horizontal footprint is an unstructured set of polygonal cells. This driver
reads the recommended ASCII ``.disv`` input file -- one file of a MODFLOW 6 model
directory -- and builds the single horizontal cell layer as a ``pyramids.netcdf.ugrid``
surface mesh.

The ASCII grammar is three keyword blocks (the intervening ``GRIDDATA`` block, which
carries ``TOP``/``BOTM``/``IDOMAIN``, is skipped)::

    BEGIN DIMENSIONS
      NLAY <nlay>        number of layers
      NCPL <ncpl>        number of cells per layer -- the face count
      NVERT <nvert>      number of vertices -- the node count
    END DIMENSIONS

    BEGIN VERTICES
      <iv> <x> <y>       one 1-based vertex per line
    END VERTICES

    BEGIN CELL2D
      <icell2d> <xc> <yc> <ncvert> <icvert...>   1-based vertex ids, clockwise
    END CELL2D

``VERTICES`` becomes ``node_x`` / ``node_y``; each ``CELL2D`` row's ``icvert`` list
becomes one fill-padded, 1-based face row (pyramids re-bases it to 0 and rewinds the
clockwise order to counter-clockwise); ``xc`` / ``yc`` become the face centres. A DISV
grid repeats this one horizontal footprint over ``NLAY`` layers, so the mesh is that
footprint of ``NCPL`` polygons -- the vertical layering is not modelled here (see the
limitations below).

The binary companion grid (``.disv.grb``, header ``GRID DISV``) is **not read in v1**:
``.disv`` is the recommended, self-describing target and the driver claims only that
suffix. See ``docs/reference/modflow6.md``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# Block keywords. A DISV file opens each section with ``BEGIN <name>`` and closes it with
# ``END <name>``; everything between a block this driver does not read is ignored.
_BEGIN = "BEGIN"
_END = "END"
_DIMENSIONS = "DIMENSIONS"
_VERTICES = "VERTICES"
_CELL2D = "CELL2D"
_COMMENT = "#"

# The dimension keywords the reader needs; others in the block (e.g. ``NLAY``) are read
# but only ``NVERT`` and ``NCPL`` size the mesh.
_NVERT = "NVERT"
_NCPL = "NCPL"

# 1-based vertex ids never take 0, so 0 is a safe pad for a short (polygonal) face row;
# pyramids rewrites it to -1 during normalisation.
_FILL = 0


class Modflow6DisvDriver(BaseDriver):
    """Reader for the MODFLOW 6 DISV (discretization by vertices) ASCII grid.

    The ``.disv`` file is one file of a MODFLOW 6 model directory. ``read_file`` dispatches
    to this driver on the ``.disv`` suffix, and ``can_open`` confirms the file carries the
    DISV blocks rather than only matching the name.

    Documented limitations:
        - **One horizontal layer.** A DISV grid is ``NLAY`` layers sharing a single
          horizontal footprint of ``NCPL`` cells; the mesh is that footprint. The vertical
          layering and the ``TOP`` / ``BOTM`` / ``IDOMAIN`` arrays in ``GRIDDATA`` are not
          modelled -- this package reads the mesh file format, not the full groundwater
          model.
        - **ASCII only.** The binary ``.disv.grb`` grid is not read in v1; point the reader
          at the ASCII ``.disv`` input, the recommended self-describing form.
        - **Vertex ids are 1-based.** They are resolved to 0-based and the base recorded for
          a writer. A short (polygonal) face row is padded; a triangular grid like the test
          fixture has no padding.
        - **Cells are clockwise.** DISV lists ``icvert`` clockwise; pyramids reverses each
          face to counter-clockwise on read.
        - There is no CRS or node elevation in a ``.disv`` file, so ``crs`` and ``node_z``
          are ``None``; the cell centres are carried as the face centres.

    Example:
        Read a MODFLOW 6 DISV grid by its ``.disv`` file (shown, not run as a doctest -- it
        needs a real DISV file)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("model/tri_model.disv", driver="modflow6_disv")
            footprint = mesh.mesh      # NCPL polygons, one horizontal layer
    """

    name = "modflow6_disv"
    long_name = "MODFLOW 6 DISV"
    suffixes = (".disv",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file carries the DISV blocks, not only the suffix.

        Reads the whole file -- a DISV grid is small -- and accepts it only when the
        ``DIMENSIONS``, ``VERTICES`` and ``CELL2D`` blocks are all present with the vertex
        and cell-count dimensions, so a stray or empty ``.disv`` is declined.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the DISV blocks and dimensions are all present.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read().upper()
        except OSError:
            return False
        has_blocks = all(
            f"{_BEGIN} {block}" in text for block in (_DIMENSIONS, _VERTICES, _CELL2D)
        )
        has_dims = _NVERT in text and _NCPL in text
        return has_blocks and has_dims

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a ``.disv`` file into a :class:`ParsedMesh`.

        Args:
            path: The ``.disv`` file.
            lazy: Unused -- a DISV grid is small and read whole.

        Returns:
            ParsedMesh: The horizontal cell footprint, 1-based and clockwise as written.

        Raises:
            FileFormatNotSupportedError: If the path is not a file, or the DISV blocks are
                missing, malformed, or disagree with the declared dimensions.
        """
        if not os.path.isfile(path):
            raise FileFormatNotSupportedError(f"{path!r} is not a MODFLOW 6 DISV file")
        return _Modflow6DisvReader(path).build()


class _Modflow6DisvReader:
    """One ``.disv`` file: read the keyword blocks and assemble the surface mesh."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Read the blocks and build the horizontal cell footprint.

        Returns:
            ParsedMesh: Nodes from ``VERTICES`` and 1-based, fill-padded faces from
                ``CELL2D``, with the cell centres as face centres.

        Raises:
            FileFormatNotSupportedError: If a required block is missing or malformed.
        """
        blocks = self._read_blocks()
        dimensions = self._parse_dimensions(blocks.get(_DIMENSIONS, []))
        node_x, node_y = self._parse_vertices(
            blocks.get(_VERTICES, []), dimensions.get(_NVERT)
        )
        faces, face_x, face_y = self._parse_cells(
            blocks.get(_CELL2D, []), dimensions.get(_NCPL)
        )
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=_FILL,
            start_index=1,  # icvert is 1-based; pyramids re-bases and records the base
            original_start_index=1,
            node_z=None,
            face_x=face_x,
            face_y=face_y,
            crs=None,
            variables={},  # the .disv file carries geometry only
            global_attributes={"source": "MODFLOW 6 DISV"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_blocks(self) -> dict[str, list[str]]:
        """Group the significant lines by the ``BEGIN``/``END`` block they sit in.

        Comments (``#``) and blank lines are dropped, and a line outside any block is
        ignored -- a DISV file's content all lives inside the keyword blocks.

        Returns:
            dict[str, list[str]]: The data lines of each block, keyed by upper-case name.

        Raises:
            FileFormatNotSupportedError: If the file cannot be read.
        """
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
                raw_lines = handle.readlines()
        except OSError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} could not be read as a MODFLOW 6 DISV file: {error}"
            ) from error
        blocks: dict[str, list[str]] = {}
        current: str | None = None
        for raw in raw_lines:
            line = raw.split(_COMMENT, 1)[0].strip()
            if not line:
                continue
            keyword = line.upper().split()
            if keyword[0] == _BEGIN and len(keyword) >= 2:
                current = keyword[1]
                blocks.setdefault(current, [])
            elif keyword[0] == _END:
                current = None
            elif current is not None:
                blocks[current].append(line)
        return blocks

    def _parse_dimensions(self, lines: list[str]) -> dict[str, int]:
        """Read the ``DIMENSIONS`` block into an upper-case keyword-to-int mapping.

        Args:
            lines: The data lines of the ``DIMENSIONS`` block.

        Returns:
            dict[str, int]: Every ``<KEYWORD> <int>`` entry, e.g. ``NVERT`` and ``NCPL``.
        """
        dimensions: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    dimensions[parts[0].upper()] = int(parts[1])
                except ValueError:
                    continue
        return dimensions

    def _parse_vertices(
        self, lines: list[str], declared: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the ``VERTICES`` block into node coordinate arrays.

        Each line is ``<iv> <x> <y>`` with ``iv`` a 1-based id; the node arrays are
        ordered by that id, so a face's 1-based ``icvert`` indexes straight into them.

        Args:
            lines: The data lines of the ``VERTICES`` block.
            declared: ``NVERT`` from the dimensions, used to size and cross-check.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``node_x`` and ``node_y``, float64.

        Raises:
            FileFormatNotSupportedError: If the block is empty, a line is malformed, an id
                is out of range, or the count disagrees with ``NVERT``.
        """
        if not lines:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no VERTICES block entries"
            )
        count = declared if declared is not None else len(lines)
        node_x = np.full(count, np.nan, dtype=np.float64)
        node_y = np.full(count, np.nan, dtype=np.float64)
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} VERTICES line is not '<iv> <x> <y>': {line!r}"
                )
            try:
                index = int(parts[0]) - 1
                node_x[index] = float(parts[1])
                node_y[index] = float(parts[2])
            except (ValueError, IndexError) as error:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} VERTICES line {line!r} is out of range or "
                    f"non-numeric: {error}"
                ) from error
        if np.isnan(node_x).any():
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares NVERT {count} but some vertex ids are missing"
            )
        return node_x, node_y

    def _parse_cells(
        self, lines: list[str], declared: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the ``CELL2D`` block into fill-padded faces and face centres.

        Each line is ``<icell2d> <xc> <yc> <ncvert> <icvert...>``; the ``icvert`` ids are
        kept 1-based and clockwise as written (pyramids re-bases and rewinds them). Rows
        are ordered by ``icell2d`` and padded to the widest polygon with :data:`_FILL`.

        Args:
            lines: The data lines of the ``CELL2D`` block.
            declared: ``NCPL`` from the dimensions, used to size and cross-check.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The ``(n_face, max_nodes)`` face
                connectivity, and the ``face_x`` / ``face_y`` centre arrays.

        Raises:
            FileFormatNotSupportedError: If the block is empty, a line is malformed, or the
                count disagrees with ``NCPL``.
        """
        if not lines:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no CELL2D block entries"
            )
        parsed = [self._parse_cell_line(line) for line in lines]
        count = declared if declared is not None else len(parsed)
        if len(parsed) != count:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares NCPL {count} but has {len(parsed)} CELL2D rows"
            )
        max_nodes = max(len(icvert) for _, _, _, icvert in parsed)
        faces = np.full((count, max_nodes), _FILL, dtype=np.int64)
        face_x = np.empty(count, dtype=np.float64)
        face_y = np.empty(count, dtype=np.float64)
        for icell, xc, yc, icvert in parsed:
            row = icell - 1
            faces[row, : len(icvert)] = icvert
            face_x[row] = xc
            face_y[row] = yc
        return faces, face_x, face_y

    def _parse_cell_line(self, line: str) -> tuple[int, float, float, list[int]]:
        """Parse one ``CELL2D`` line into ``(icell2d, xc, yc, icvert)``.

        Args:
            line: A single ``CELL2D`` data line.

        Returns:
            tuple[int, float, float, list[int]]: The 1-based cell id, its centre, and the
                1-based vertex ids making the polygon.

        Raises:
            FileFormatNotSupportedError: If the line is too short, non-numeric, or its
                vertex count does not match the declared ``ncvert``.
        """
        parts = line.split()
        if len(parts) < 5:
            raise FileFormatNotSupportedError(
                f"{self.path!r} CELL2D line is not "
                f"'<icell2d> <xc> <yc> <ncvert> <icvert...>': {line!r}"
            )
        try:
            icell = int(parts[0])
            xc, yc = float(parts[1]), float(parts[2])
            ncvert = int(parts[3])
            icvert = [int(value) for value in parts[4 : 4 + ncvert]]
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} CELL2D line {line!r} is non-numeric: {error}"
            ) from error
        if len(icvert) != ncvert or ncvert < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} CELL2D line {line!r} declares ncvert {ncvert} but gives "
                f"{len(icvert)} vertex id(s); a face needs at least three"
            )
        return icell, xc, yc, icvert
