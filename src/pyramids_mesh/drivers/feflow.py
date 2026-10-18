"""DHI FEFLOW classic ASCII ``.fem`` mesh reader.

FEFLOW is DHI's finite-element groundwater model. Its classic ASCII ``.fem`` file is a
keyword deck: a sequence of all-caps section headers (``PROBLEM:``, ``CLASS``, ``DIMENS``,
``SCALE``, ``NODE``, ``COOR`` and many more) each followed by the section's numeric rows,
up to a trailing ``END``. This driver reads the mesh those sections describe -- the node
coordinates and the element-to-node connectivity -- and builds a ``pyramids.netcdf.ugrid``
surface mesh. The physics, boundary-condition, material and time-series sections that
follow the geometry are not interpreted.

The geometry lives in four sections:

- ``DIMENS`` -- a single row ``<n_nodes> <n_elements> <nodes_per_element> ...``. The
  third field is the element valence: ``3`` triangle, ``4`` quad or tetrahedron, ``6``
  prism, ``8`` hexahedron.
- ``SCALE`` -- six numbers, a ``(scale, offset)`` pair per axis, used to map the stored
  coordinates to real-world ones: ``real = stored * scale + offset`` for x, then y, then
  z. A 2D mesh leaves the z pair at zero.
- ``NODE`` -- ``n_elements`` rows, each ``nodes_per_element`` 1-based node ids; the
  element-to-node connectivity.
- ``COOR`` -- a single flat, comma-separated stream (wrapped ~12 values per line) of all
  node x-coordinates, then all y, then -- for a 3D mesh -- all z. Whether z is present is
  decided by the stream length: ``len(COOR) / n_nodes`` is ``2`` or ``3``.

The ``.fem`` suffix is **shared** with the Nastran bulk-data format, so ``can_open``
requires the FEFLOW signature -- the first two non-blank statements being ``PROBLEM:`` then
``CLASS`` -- and declines a Nastran deck (which opens on ``$`` comment lines and
``BEGIN BULK``). See ``docs/reference/feflow.md``.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# Section headers. A FEFLOW deck opens each section with an all-caps keyword line and the
# section runs until the next keyword; only these four carry the geometry this reader
# builds, the rest (physics, boundary conditions, materials, time series) are ignored.
_PROBLEM = "PROBLEM"
_CLASS = "CLASS"
_DIMENS = "DIMENS"
_SCALE = "SCALE"
_NODE = "NODE"
_COOR = "COOR"

# 1-based node ids never take 0, so 0 is a safe pad for a short element row (a mixed
# triangle/quad deck); pyramids rewrites it to -1 during normalisation. A single-valence
# deck pads nothing.
_FILL = 0


class FeflowDriver(BaseDriver):
    """Reader for the DHI FEFLOW classic ASCII ``.fem`` finite-element mesh.

    ``read_file`` dispatches to this driver on the ``.fem`` suffix. Because that suffix is
    also Nastran's, ``can_open`` confirms the FEFLOW signature -- the first two non-blank
    statements being ``PROBLEM:`` then ``CLASS`` -- and declines a Nastran bulk-data deck.

    The mesh comes from the ``NODE`` (element-to-node connectivity, 1-based) and ``COOR``
    (the flat coordinate stream) sections, with ``DIMENS`` giving the counts and element
    valence and ``SCALE`` the per-axis ``(scale, offset)`` transform applied to the
    coordinates. A 3D deck (``COOR`` three coordinates per node) carries the elevation as
    ``node_z``.

    Documented limitations:
        - **Geometry only.** Only ``DIMENS`` / ``SCALE`` / ``NODE`` / ``COOR`` are read;
          the physics, boundary-condition, material and time-series sections are not
          interpreted -- this package reads the mesh file format, not the full model.
        - **Node ids are 1-based.** Connectivity is resolved to 0-based and the base
          recorded for a writer. A mixed-valence deck pads short rows; a single-valence
          deck (the triangular test fixture) pads nothing.
        - **No CRS.** A ``.fem`` file carries no spatial-reference authority, so ``crs`` is
          ``None`` rather than guessed. The ``SCALE`` transform is applied to the
          coordinates as the file specifies.
        - **``.fem`` is shared with Nastran.** ``can_open`` keys on the ``PROBLEM:`` /
          ``CLASS`` signature so a Nastran deck sharing the suffix is declined rather than
          mis-parsed.

    Example:
        Read a FEFLOW mesh by its ``.fem`` file (shown, not run as a doctest -- it needs a
        real FEFLOW file)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("model/disv_nodes.fem", driver="feflow")
            surface = mesh.mesh      # the finite-element mesh
    """

    name = "feflow"
    long_name = "DHI FEFLOW (classic ASCII)"
    suffixes = (".fem",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens as a FEFLOW deck, never raising.

        Reads only the first two non-blank lines and accepts the file when their leading
        tokens are ``PROBLEM`` then ``CLASS`` (the ``PROBLEM:`` colon is tolerated). A
        Nastran deck opens on ``$`` comment lines and ``BEGIN BULK``, so it fails this
        check and is declined -- which is the whole point of the signature, since the two
        formats share the ``.fem`` suffix.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` only when the first two statements are ``PROBLEM:`` then
                ``CLASS``.
        """
        try:
            statements: list[str] = []
            with open(path, "r", encoding="latin-1") as handle:
                for raw in handle:
                    stripped = raw.strip()
                    if stripped:
                        statements.append(stripped)
                        if len(statements) >= 2:
                            break
        except OSError:
            return False
        if len(statements) < 2:
            return False
        first = statements[0].split()[0].upper().rstrip(":")
        second = statements[1].split()[0].upper().rstrip(":")
        return first == _PROBLEM and second == _CLASS

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a ``.fem`` file into a :class:`ParsedMesh`.

        Args:
            path: The ``.fem`` file.
            lazy: Unused -- a FEFLOW mesh is read whole.

        Returns:
            ParsedMesh: The finite-element surface mesh, 1-based as written, with
                ``node_z`` when the deck is 3D.

        Raises:
            FileFormatNotSupportedError: If the path is not a file, or the ``DIMENS`` /
                ``SCALE`` / ``NODE`` / ``COOR`` sections are missing, malformed, or
                disagree with the declared dimensions.
        """
        if not os.path.isfile(path):
            raise FileFormatNotSupportedError(f"{path!r} is not a FEFLOW .fem file")
        return _FeflowReader(path).build()


class _FeflowReader:
    """One ``.fem`` file: scan the keyword sections and assemble the mesh."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Read the sections and build the finite-element surface mesh.

        Returns:
            ParsedMesh: Nodes from ``COOR`` (``SCALE`` applied), 1-based, fill-padded
                faces from ``NODE``, and ``node_z`` when ``COOR`` carries three
                coordinates per node.

        Raises:
            FileFormatNotSupportedError: If a required section is missing or malformed.
        """
        sections = self._read_sections()
        n_nodes, n_elements, nodes_per_element = self._parse_dimens(
            sections.get(_DIMENS, [])
        )
        scale = self._parse_scale(sections.get(_SCALE, []))
        faces = self._parse_nodes(
            sections.get(_NODE, []), n_elements, nodes_per_element
        )
        node_x, node_y, node_z = self._parse_coor(
            sections.get(_COOR, []), n_nodes, scale
        )
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=_FILL,
            start_index=1,  # NODE ids are 1-based; pyramids re-bases and records the base
            original_start_index=1,
            node_z=node_z,
            crs=None,
            variables={},  # the geometry sections carry no data variables
            global_attributes={"source": "DHI FEFLOW .fem"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_sections(self) -> dict[str, list[str]]:
        """Group the data lines by the keyword section they sit in.

        A line whose first character is a letter or underscore opens a section (its first
        token, upper-cased and with a trailing colon stripped, is the key); any other
        non-blank line is a data row of the current section. The first occurrence of each
        keyword wins, so a header seen twice extends the same section.

        Returns:
            dict[str, list[str]]: The data rows of each section, keyed by upper-case name.

        Raises:
            FileFormatNotSupportedError: If the file cannot be read.
        """
        try:
            with open(self.path, "r", encoding="latin-1") as handle:
                raw_lines = handle.readlines()
        except OSError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} could not be read as a FEFLOW .fem file: {error}"
            ) from error
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for raw in raw_lines:
            line = raw.strip()
            if not line:
                continue
            if line[0].isalpha() or line[0] == "_":
                current = line.split()[0].upper().rstrip(":")
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)
        return sections

    def _parse_dimens(self, lines: list[str]) -> tuple[int, int, int]:
        """Read the ``DIMENS`` row into ``(n_nodes, n_elements, nodes_per_element)``.

        Args:
            lines: The data rows of the ``DIMENS`` section.

        Returns:
            tuple[int, int, int]: The node count, element count and element valence.

        Raises:
            FileFormatNotSupportedError: If the section is missing or its first row has
                fewer than three integers.
        """
        if not lines:
            raise FileFormatNotSupportedError(f"{self.path!r} has no DIMENS section")
        parts = lines[0].replace(",", " ").split()
        try:
            n_nodes = int(parts[0])
            n_elements = int(parts[1])
            nodes_per_element = int(parts[2])
        except (IndexError, ValueError) as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} DIMENS row {lines[0]!r} is not "
                f"'<n_nodes> <n_elements> <nodes_per_element> ...': {error}"
            ) from error
        if n_nodes <= 0 or n_elements <= 0 or nodes_per_element < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} DIMENS declares n_nodes {n_nodes}, n_elements "
                f"{n_elements}, nodes_per_element {nodes_per_element}; all must be "
                f"positive and an element needs at least three nodes"
            )
        return n_nodes, n_elements, nodes_per_element

    def _parse_scale(self, lines: list[str]) -> np.ndarray:
        """Read the ``SCALE`` row into six ``(scale, offset)`` numbers per axis.

        A missing or short ``SCALE`` is treated as identity -- scale ``1``, offset ``0`` --
        for the axes it does not cover, so a deck that omits it reads its coordinates
        verbatim.

        Args:
            lines: The data rows of the ``SCALE`` section.

        Returns:
            np.ndarray: ``[sx, ox, sy, oy, sz, oz]`` as float64, padded with the identity
                transform where the file is short.
        """
        scale = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        if not lines:
            return scale
        values = [
            float(token)
            for token in lines[0].replace(",", " ").split()
            if token.strip()
        ]
        for index, value in enumerate(values[:6]):
            scale[index] = value
        return scale

    def _parse_nodes(
        self, lines: list[str], n_elements: int, nodes_per_element: int
    ) -> np.ndarray:
        """Read the ``NODE`` section into a 1-based, fill-padded face table.

        Each row is ``nodes_per_element`` 1-based node ids. Rows are kept 1-based (pyramids
        re-bases and rewinds them); a row shorter than the widest is padded with
        :data:`_FILL`, so a mixed triangle/quad deck comes back rectangular.

        Args:
            lines: The data rows of the ``NODE`` section.
            n_elements: ``n_elements`` from ``DIMENS``, used to cross-check the row count.
            nodes_per_element: The element valence from ``DIMENS``; at most this many ids
                are read per row.

        Returns:
            np.ndarray: The ``(n_elements, max_width)`` int64 connectivity table.

        Raises:
            FileFormatNotSupportedError: If the section is empty, the row count disagrees
                with ``n_elements``, or a row is non-numeric or carries no ids.
        """
        if not lines:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no NODE section entries"
            )
        if len(lines) != n_elements:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares {n_elements} elements but the NODE section has "
                f"{len(lines)} rows"
            )
        rows: list[list[int]] = []
        for line in lines:
            try:
                ids = [
                    int(token)
                    for token in line.replace(",", " ").split()[:nodes_per_element]
                ]
            except ValueError as error:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} NODE row {line!r} is non-numeric: {error}"
                ) from error
            if not ids:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} NODE row {line!r} carries no node ids"
                )
            rows.append(ids)
        width = max(len(ids) for ids in rows)
        faces = np.full((n_elements, width), _FILL, dtype=np.int64)
        for index, ids in enumerate(rows):
            faces[index, : len(ids)] = ids
        return faces

    def _parse_coor(
        self, lines: list[str], n_nodes: int, scale: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Read the flat ``COOR`` stream into scaled node coordinates.

        ``COOR`` is one comma-separated stream of every node's x, then every node's y, then
        -- for a 3D mesh -- every node's z, wrapped across lines. The number of coordinates
        per node is inferred from the stream length (``len / n_nodes``), and the ``SCALE``
        transform ``real = stored * scale + offset`` is applied per axis.

        Args:
            lines: The data rows of the ``COOR`` section.
            n_nodes: ``n_nodes`` from ``DIMENS``, used to split and cross-check the stream.
            scale: The six ``(scale, offset)`` numbers from :meth:`_parse_scale`.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray | None]: ``node_x``, ``node_y`` and
                ``node_z`` (``None`` for a 2D mesh), all float64 with ``SCALE`` applied.

        Raises:
            FileFormatNotSupportedError: If the section is empty, non-numeric, or its
                length is not a 2D or 3D multiple of ``n_nodes``.
        """
        if not lines:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no COOR section entries"
            )
        try:
            stream = np.array(
                [
                    float(token)
                    for line in lines
                    for token in line.replace(",", " ").split()
                    if token.strip()
                ],
                dtype=np.float64,
            )
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} COOR stream is non-numeric: {error}"
            ) from error
        if n_nodes == 0 or stream.size % n_nodes != 0:
            raise FileFormatNotSupportedError(
                f"{self.path!r} COOR carries {stream.size} values, not a whole multiple "
                f"of the {n_nodes} declared nodes"
            )
        n_dims = stream.size // n_nodes
        if n_dims not in (2, 3):
            raise FileFormatNotSupportedError(
                f"{self.path!r} COOR implies {n_dims} coordinates per node; only 2D and "
                f"3D meshes are read"
            )
        node_x = stream[0:n_nodes] * scale[0] + scale[1]
        node_y = stream[n_nodes : 2 * n_nodes] * scale[2] + scale[3]
        node_z = None
        if n_dims == 3:
            node_z = stream[2 * n_nodes : 3 * n_nodes] * scale[4] + scale[5]
        return node_x, node_y, node_z
