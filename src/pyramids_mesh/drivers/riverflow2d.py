"""RiverFlow2D ``.FED`` reader (task C29, issue #67).

RiverFlow2D (Hydronia) stores its triangular finite-element mesh as a single ASCII
``.FED`` file. The layout, confirmed against the 16 meshes shipped in Hydronia's public
``ExampleProjects`` package (Hoh River, Brumadinho tailings, DamBreach, …), is::

    NELEM NNODES 5 5                                                   <- 4 integers
    IN  X  Y  ZB  INITWSE  MINERODELEV  BCTYPE  BCFILENAME             <- NNODES lines
    IE  N1 N2 N3  MANNINGN  ELZB  ELINITWSE  ELMINERODELEV  <reserved> <- NELEM lines

Node and element ids are 1-based and consecutive; every element is a counter-clockwise
triangle (``N1 N2 N3``). The header's trailing two fields are unused dummies, and a
trailing element column is reserved (``0.0`` in every shipped mesh).

What becomes what: ``X``/``Y`` are the node coordinates, ``ZB`` the node bed elevation
(a ``node_z`` variable), and the connectivity the triangular faces. The per-cell data in
a ``.FED`` is **element-located**: ``MANNINGN`` (``manning_n``), ``ELINITWSE``
(``init_wse``) and ``ELMINERODELEV`` (``min_erode_elev``) are carried as face variables.
The per-node ``INITWSE``/``MINERODELEV``/``BCTYPE``/``BCFILENAME`` columns are sentinel
in every shipped mesh and are not surfaced; boundary conditions live in the separate
``.TBA`` file, not here.

This is a mesh reader only. RiverFlow2D writes results as VTK/HDF5, read by other drivers
in this package, so no results are read here. No CRS: a ``.FED`` carries projected metric
coordinates but declares no authority, so ``crs`` is ``None`` rather than guessed.

See ``planning/design.md`` and issue #67. The format boundary (ADR-0005) is respected:
this only parses the file into ``pyramids.netcdf.ugrid`` objects -- no GIS or analysis.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# Minimum token counts for a well-formed line. The header names four integers but only the
# first two carry meaning; a node line names eight fields but we read the first four; an
# element line names nine but we read through the eighth.
_MIN_HEADER_FIELDS = 2
_MIN_NODE_FIELDS = 4  # IN X Y ZB (the columns this reader consumes)
_MIN_ELEM_FIELDS = 8  # IE N1 N2 N3 MANNINGN ELZB ELINITWSE ELMINERODELEV
_TRIANGLE = 3

# RiverFlow2D's "no value" sentinel, carried as nodata on the variables that use it.
_FILL = -9999.0


class RiverFlow2dDriver(BaseDriver):
    """Reader for the RiverFlow2D (Hydronia) ``.FED`` triangular mesh.

    ``read`` builds the surface mesh from the single ``.FED`` file: node coordinates, the
    bed elevation as ``node_z``, the 1-based triangular connectivity, and the element
    Manning's n, initial water-surface elevation and minimum erodible elevation as face
    variables.

    Documented limitations:
        - **Triangles only.** A ``.FED`` is a triangulation; a line that does not carry
          three node ids is refused rather than guessed.
        - **Ids are 1-based and consecutive.** The format numbers nodes ``1..NNODES`` in
          order; a file that breaks that is refused rather than silently mis-wired. The
          base is recorded for a writer.
        - **Per-cell data is element-located.** ``manning_n``, ``init_wse`` and
          ``min_erode_elev`` are face variables; the per-node columns of the same name
          are sentinel in every shipped mesh and are not surfaced. ``-9999`` is recorded
          as nodata on ``min_erode_elev``.
        - **Boundary conditions are not here.** They live in the companion ``.TBA`` file,
          which this reader does not consult.
        - **Results are not read here.** RiverFlow2D writes VTK/HDF5, read by other
          drivers in this package.
        - **No CRS.** The coordinates are projected metres with no declared authority, so
          ``crs`` is ``None``.

    Example:
        Read a RiverFlow2D mesh (shown, not run as a doctest -- it needs the committed
        sample)::

            from pyramids.netcdf.ugrid import MeshDataset

            mesh = MeshDataset.read_file("model.fed", driver="riverflow2d")
            bed = mesh.get("node_z")        # node-located bed elevation
            n = mesh.get("manning_n")       # face-located Manning's n
    """

    name = "riverflow2d"
    long_name = "RiverFlow2D mesh"
    suffixes = (".fed",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Cheaply confirm a ``.FED`` by its header and first node line.

        Discriminating without reading the whole file: line 1 is ``NELEM NNODES ...``
        (two positive integers) and the first node line is ``IN X Y ZB ...`` (an integer
        id followed by three floats). Never raises for a parse failure -- a malformed
        candidate simply fails one of the checks and is declined.

        Args:
            path: The ``.fed`` file to inspect; only its first two significant lines are
                read.

        Returns:
            bool: ``True`` when the header and first node line are FED-shaped, else
                ``False``.
        """
        rows = _first_rows(path, count=2)
        if len(rows) < 2:
            return False
        header, node = rows[0], rows[1]
        if len(header) < _MIN_HEADER_FIELDS or len(node) < _MIN_NODE_FIELDS:
            return False
        nelem = _maybe_int(header[0])
        nnodes = _maybe_int(header[1])
        if nelem is None or nnodes is None or nelem <= 0 or nnodes <= 0:
            return False
        if _maybe_int(node[0]) is None:
            return False
        return all(_maybe_float(token) is not None for token in node[1:4])

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a ``.FED`` file into a :class:`ParsedMesh`.

        Args:
            path: The ``.FED`` file.
            lazy: Unused -- a ``.FED`` is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, bed elevation, the 1-based triangular connectivity and the
                element Manning/WSE/erode face variables.

        Raises:
            FileFormatNotSupportedError: On a missing or malformed header, a short or
                non-numeric node/element line, a non-triangular element, non-consecutive
                node ids, or a file declaring no elements.
            ValueError: If an element references a node id outside ``1..NNODES``.
        """
        return _FedReader(path).build()


class _FedReader:
    """One ``.FED`` file, parsed into a :class:`ParsedMesh`."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Parse the file and assemble the mesh with its face variables.

        Returns:
            ParsedMesh: Node coordinates, bed elevation (``node_z``), the 1-based
                triangular connectivity, and the ``manning_n`` / ``init_wse`` /
                ``min_erode_elev`` face variables.

        Raises:
            FileFormatNotSupportedError: On an empty file, a short or non-integer header,
                a header that declares more nodes/elements than the file holds, a file
                declaring no elements, a short or non-numeric node/element line, or
                non-consecutive node ids.
            ValueError: If an element references a node id outside ``1..NNODES``.
        """
        rows = _significant_rows(self.path)
        if not rows:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is empty, so it carries no RiverFlow2D mesh"
            )
        nelem, nnodes = self._header(rows[0])
        node_rows = rows[1 : 1 + nnodes]
        elem_rows = rows[1 + nnodes : 1 + nnodes + nelem]
        if len(node_rows) < nnodes or len(elem_rows) < nelem:
            raise FileFormatNotSupportedError(
                f"{self.path!r} header declares {nnodes} nodes and {nelem} elements but "
                f"the file holds {len(node_rows)} node and {len(elem_rows)} element lines"
            )
        if nelem <= 0:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no elements, so it carries no mesh"
            )
        node_x, node_y, node_z = self._nodes(node_rows)
        faces, manning, init_wse, min_erode = self._elements(elem_rows, nnodes)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # The connectivity holds the file's own 1-based ids; the template method
            # re-bases it to 0-based and records the base for a writer.
            start_index=1,
            node_z=node_z,
            crs=None,
            variables={
                "manning_n": ParsedVariable(
                    location="face", shape=(nelem,), values=manning
                ),
                "init_wse": ParsedVariable(
                    location="face", shape=(nelem,), values=init_wse
                ),
                "min_erode_elev": ParsedVariable(
                    location="face", shape=(nelem,), values=min_erode, nodata=_FILL
                ),
            },
            global_attributes={"source": "RiverFlow2D .FED"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _header(self, row: list[str]) -> tuple[int, int]:
        """Read ``NELEM NNODES`` from the header line; the trailing dummies are ignored."""
        if len(row) < _MIN_HEADER_FIELDS:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short header: expected 'NELEM NNODES ...', "
                f"found {row!r}"
            )
        nelem = _maybe_int(row[0])
        nnodes = _maybe_int(row[1])
        if nelem is None or nnodes is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} header is not two integers: found {row[:2]!r}"
            )
        return nelem, nnodes

    def _nodes(self, node_rows: list[list[str]]):
        """Read the node block into x, y and bed-elevation arrays.

        The node ids must be consecutive ``1..NNODES`` -- the format's guarantee -- so
        the face connectivity can index node ``k`` at array position ``k - 1``. A file
        that breaks this is refused rather than silently mis-wired.

        Args:
            node_rows: The node block, each row already split into whitespace tokens.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The ``node_x``, ``node_y`` and
                ``node_z`` (bed-elevation) arrays, each of length ``len(node_rows)``.

        Raises:
            FileFormatNotSupportedError: On a node line with fewer than four fields, a
                non-numeric coordinate, or a node id that is not its 1-based position.
        """
        node_x = np.empty(len(node_rows), dtype=np.float64)
        node_y = np.empty(len(node_rows), dtype=np.float64)
        node_z = np.empty(len(node_rows), dtype=np.float64)
        for index, row in enumerate(node_rows):
            if len(row) < _MIN_NODE_FIELDS:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short node line: expected 'IN X Y ZB ...', "
                    f"found {len(row)} field(s)"
                )
            native = _maybe_int(row[0])
            if native != index + 1:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} node ids are not consecutive from 1: expected "
                    f"{index + 1}, found {row[0]!r}"
                )
            node_x[index] = self._float(row[1], "node")
            node_y[index] = self._float(row[2], "node")
            node_z[index] = self._float(row[3], "node")
        return node_x, node_y, node_z

    def _elements(self, elem_rows: list[list[str]], nnodes: int):
        """Read the element block into connectivity and the three face-variable columns.

        Args:
            elem_rows: The element block, each row already split into whitespace tokens.
            nnodes: The node count, used to bound-check each vertex id.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The ``(n_elem, 3)``
                1-based face connectivity and the per-element ``manning_n``, ``init_wse``
                and ``min_erode_elev`` columns.

        Raises:
            FileFormatNotSupportedError: On an element line with fewer than eight fields
                or a non-numeric Manning/WSE/erode field.
            ValueError: If a vertex id is not an integer in ``1..NNODES``.
        """
        faces = np.empty((len(elem_rows), _TRIANGLE), dtype=np.int64)
        manning = np.empty(len(elem_rows), dtype=np.float64)
        init_wse = np.empty(len(elem_rows), dtype=np.float64)
        min_erode = np.empty(len(elem_rows), dtype=np.float64)
        for index, row in enumerate(elem_rows):
            if len(row) < _MIN_ELEM_FIELDS:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short element line: expected 'IE N1 N2 N3 "
                    f"MANNINGN ELZB ELINITWSE ELMINERODELEV', found {len(row)} field(s)"
                )
            for column in range(_TRIANGLE):
                vertex = _maybe_int(row[1 + column])
                if vertex is None or not 1 <= vertex <= nnodes:
                    raise ValueError(
                        f"{self.path!r} element {index + 1} references node id "
                        f"{row[1 + column]!r}, outside 1..{nnodes}"
                    )
                faces[index, column] = vertex
            manning[index] = self._float(row[4], "element")
            init_wse[index] = self._float(row[6], "element")
            min_erode[index] = self._float(row[7], "element")
        return faces, manning, init_wse, min_erode

    def _float(self, token: str, kind: str) -> float:
        """Parse a float field, or raise naming the record it came from."""
        value = _maybe_float(token)
        if value is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a {kind} line with a non-numeric field {token!r}"
            )
        return value


def _significant_rows(path: str) -> list[list[str]]:
    """Return every non-blank line of the file as its whitespace-split tokens."""
    with open(path, encoding="latin-1") as handle:
        return [tokens for line in handle if (tokens := line.split())]


def _first_rows(path: str, *, count: int) -> list[list[str]]:
    """Return the first ``count`` non-blank lines as token lists, for a cheap sniff."""
    rows: list[list[str]] = []
    with open(path, encoding="latin-1") as handle:
        for line in handle:
            tokens = line.split()
            if tokens:
                rows.append(tokens)
                if len(rows) >= count:
                    break
    return rows


def _maybe_int(token: str) -> int | None:
    """Return ``token`` as an int, or ``None`` when it is not one."""
    try:
        return int(token)
    except ValueError:
        return None


def _maybe_float(token: str) -> float | None:
    """Return ``token`` as a float, or ``None`` when it is not one."""
    try:
        return float(token)
    except ValueError:
        return None
