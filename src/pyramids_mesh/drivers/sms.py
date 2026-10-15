"""SMS-family drivers: 2DM (mesh frame) and DAT (results-only).

2DM is the reference driver every later one copies (task D1). DAT carries values but
no topology, so it declares ``needs_mesh`` and implements ``read_data`` rather than
``read`` (task D2). See ``planning/design.md`` section 20.1.

The 2DM structure here was verified against MDAL's ``mdal_2dm.cpp``: a plain-text file
whose first significant line is ``MESH2D``, then one record per line.

``topology_info`` is ``None`` for this driver -- a 2DM file has no NetCDF variables to
name, and nothing is synthesised that ``to_file`` would not write.
"""

from __future__ import annotations

import numpy as np

# Imported from a private module because there is no public one: as of pyramids-gis
# 0.68.0 nothing named *Error is re-exported from `pyramids` or `pyramids.base`, while
# the published driver contract ("Exceptions -- reuse, do not invent") tells a driver to
# raise exactly this type. ADR-0005 requires a local workaround to name the upstream
# gap and be removed when it closes: the ask upstream is a public export of
# FileFormatNotSupportedError (and a public eager/lazy MeshVariable constructor, which
# _base.py needs for the same reason). Until then a patch release renaming the private
# module would break this import, so `tests/test_package.py` keeps a canary on it.
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._text import iter_records, native_id_map

_HEADER = "MESH2D"

# The node count comes from the keyword's second character (MDAL reads it as
# ``line[1]`` and asserts 3, 4 or 6), but the full keyword is matched instead: E2L and
# E3L carry a digit too and are line elements, not faces.
_FACE_NODE_COUNTS = {"E3T": 3, "E4Q": 4, "E6T": 6}

# MDAL refuses the whole file on these (``Err_UnsupportedElement``) rather than guess a
# mapping, and so does this driver. E3L is a 1D element MDAL also rejects.
_UNSUPPORTED_ELEMENTS = ("E3L", "E8Q", "E9Q")

_LINE_ELEMENT = "E2L"
_NODE = "ND"
_MATERIALS_TAG = "NUM_MATERIALS_PER_ELEM"

# Kept in step with ``iter_records``' default: the header sniff and the parser must
# skip the same lines, or one accepts a file the other refuses.
_COMMENT = "#"

# ``can_open`` must not read the whole file, so the header sniff reads in blocks and
# stops at the first significant line. A single fixed prefix was not enough: a file
# whose comment header ran past it was declined by ``can_open`` and then read fine by
# ``read``, which is the asymmetry the sniff exists to avoid. The limit still bounds the
# work for a file that is all comments, or binary -- which decodes to replacement
# characters rather than raising.
_SNIFF_BLOCK = 4096
_SNIFF_LIMIT = 256 * 1024

# The smallest valid 2DM: the unit square split into two counter-clockwise triangles.
_MINIMAL_FIXTURE = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
E3T 1 1 2 3 1
E3T 2 1 3 4 1
"""


class TwoDmDriver(BaseDriver):
    """Reader for the SMS/TUFLOW 2DM mesh-frame format.

    Reads nodes, triangular/quadrilateral/six-node faces and 1D line elements. Node
    elevation becomes a ``node_z`` variable and the material id a face-located
    ``material`` variable, since the pyramids mesh model is planar and carries no
    per-node elevation coordinate.

    Documented limitations:
        - ``E3L``, ``E8Q`` and ``E9Q`` are refused outright, as MDAL refuses them;
          treating a mid-side or centre node as a corner would silently distort the
          element.
        - ``E6T`` is read as a real six-node face, not demoted to its corner triangle.
        - A file may hold a surface mesh, an ``E2L`` network, or both. They are named
          separately (``mesh_names`` -> ``["mesh2d", "network1d"]``), so one read
          returns one topology and the caller picks with ``mesh_name``. Reading either
          half warns about what the other leaves behind. ``E2L`` elements are a
          separate 1D topology, never folded into the surface's edge connectivity.
        - A 1D read carries no ``node_z``, variables or global attributes:
          ``Network1d`` has no parameter for them. Each is named in a warning rather
          than dropped quietly. That is a ``pyramids-gis`` limitation to fix upstream.
        - ``node_z`` is omitted entirely when no ``ND`` record carries a z column,
          rather than fabricated as zeros.
        - 2DM carries no CRS, so ``crs`` is always ``None``; it is never guessed from
          the coordinate magnitudes.

    Examples:
        - Write the smallest valid 2DM and read it back. The file numbers its nodes
          from 1, and the connectivity arrives 0-based with the file's base still
          recorded for a writer to restore:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import (
            ...     TwoDmDriver,
            ...     build_minimal_fixture,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.2dm")
            ...     build_minimal_fixture(path)
            ...     dataset = TwoDmDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1

        - Elevation and the material id come back as variables, because the mesh
          model is planar and carries neither as a coordinate:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import (
            ...     TwoDmDriver,
            ...     build_minimal_fixture,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.2dm")
            ...     build_minimal_fixture(path)
            ...     dataset = TwoDmDriver().read(path)
            >>> sorted(dataset.data_variable_names)
            ['material', 'node_z']
            >>> dataset.get("node_z").location
            'node'
            >>> dataset.get("material").data.tolist()
            [1.0, 1.0]

        - Dispatch checks the suffix and sniffs the header, so the right content
          under a suffix this driver does not claim is still declined:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import (
            ...     TwoDmDriver,
            ...     build_minimal_fixture,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     claimed = os.path.join(directory, "domain.2dm")
            ...     foreign = os.path.join(directory, "domain.txt")
            ...     build_minimal_fixture(claimed)
            ...     build_minimal_fixture(foreign)
            ...     driver = TwoDmDriver()
            ...     (driver.can_open(claimed), driver.can_open(foreign))
            (True, False)

        - A file whose ``ND`` records carry only x and y gets no ``node_z`` at all,
          rather than a column of zeros a consumer could not tell from a real datum:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import TwoDmDriver
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "flat.2dm")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         print("MESH2D", file=handle)
            ...         print("ND 1 0.0 0.0", file=handle)
            ...         print("ND 2 1.0 0.0", file=handle)
            ...         print("ND 3 1.0 1.0", file=handle)
            ...         print("E3T 1 1 2 3 1", file=handle)
            ...     dataset = TwoDmDriver().read(path)
            >>> sorted(dataset.data_variable_names)
            ['material']
    """

    name = "2dm"
    long_name = "SMS/TUFLOW 2DM mesh"
    suffixes = (".2dm",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "multi_mesh"})

    def mesh_names(self, path: str) -> list[str]:
        """List the topologies the file holds, in surface-then-network order.

        A 2DM may carry a surface mesh (``E3T``/``E4Q``/``E6T``), a 1D network
        (``E2L``), or both. Naming them separately is what lets a caller pick one with
        ``mesh_name``, which is the only way to reach the 1D half of a mixed file.

        Args:
            path: The 2DM file to inspect.

        Returns:
            list[str]: ``["mesh2d"]``, ``["network1d"]``, or both. A file whose
                elements cannot be read at all reports the surface name, leaving the
                error to ``read``.

        Examples:
            - A file carrying both an ``E3T`` face and an ``E2L`` line element names
              two topologies, and either name may then be passed to ``read``:

                >>> import os
                >>> import tempfile
                >>> import warnings
                >>> from pyramids_mesh.drivers.sms import TwoDmDriver
                >>> with tempfile.TemporaryDirectory() as directory:
                ...     path = os.path.join(directory, "mixed.2dm")
                ...     with open(path, "w", encoding="utf-8") as handle:
                ...         print("MESH2D", file=handle)
                ...         print("ND 1 0.0 0.0 0.0", file=handle)
                ...         print("ND 2 1.0 0.0 0.0", file=handle)
                ...         print("ND 3 1.0 1.0 0.0", file=handle)
                ...         print("ND 4 0.0 1.0 0.0", file=handle)
                ...         print("E3T 1 1 2 3 1", file=handle)
                ...         print("E2L 1 3 4 1", file=handle)
                ...     driver = TwoDmDriver()
                ...     names = driver.mesh_names(path)
                ...     with warnings.catch_warnings():
                ...         warnings.simplefilter("ignore")
                ...         surface = driver.read(path)
                ...         channels = driver.read(path, mesh_name="network1d")
                >>> names
                ['mesh2d', 'network1d']
                >>> surface.mesh.face_node_connectivity.data.tolist()
                [[0, 1, 2]]
                >>> channels.n_edge
                1
                >>> channels.edge_node_connectivity.data.tolist()
                [[2, 3]]

            - A file with faces and no ``E2L`` names only the surface, which is the
              answer a caller needs before asking for a topology that is not there:

                >>> import os
                >>> import tempfile
                >>> from pyramids_mesh.drivers.sms import (
                ...     TwoDmDriver,
                ...     build_minimal_fixture,
                ... )
                >>> with tempfile.TemporaryDirectory() as directory:
                ...     path = os.path.join(directory, "domain.2dm")
                ...     build_minimal_fixture(path)
                ...     TwoDmDriver().mesh_names(path)
                ['mesh2d']
        """
        has_faces = False
        has_lines = False
        try:
            for keyword, _ in iter_records(path, comment=_COMMENT):
                upper = keyword.upper()
                if upper in _FACE_NODE_COUNTS:
                    has_faces = True
                elif upper == _LINE_ELEMENT:
                    has_lines = True
        except OSError:
            return [self._default_mesh_name]
        names = []
        if has_faces:
            names.append(self._default_mesh_name)
        if has_lines:
            names.append(self.network_mesh_name)
        return names or [self._default_mesh_name]

    def _sniff(self, path: str) -> bool:
        """Report whether the first significant line is the ``MESH2D`` header.

        Reads a bounded prefix and decodes it leniently, so a binary or empty file
        answers ``False`` instead of raising. A byte-order mark is dropped, and blocks
        are read until the first significant line or ``_SNIFF_LIMIT`` bytes, so a file
        whose comment header runs long is not declined for that alone.

        Blank and comment lines are skipped, exactly as
        :func:`~pyramids_mesh._text.iter_records` skips them while parsing. The two must
        agree: a sniff stricter than the parser would decline a file this driver can
        read, and dispatch would then hand it to nobody.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first significant line starts with ``MESH2D``.
        """
        with open(path, "rb") as handle:
            buffer = b""
            while len(buffer) < _SNIFF_LIMIT:
                block = handle.read(_SNIFF_BLOCK)
                if not block:
                    break
                buffer += block
                # `utf-8-sig` drops a byte-order mark, which Windows editors add and
                # which would otherwise make the first token '﻿MESH2D'.
                for line in buffer.decode("utf-8-sig", errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith(_COMMENT):
                        # Compare the first token for equality, exactly as the parser
                        # does. A `startswith` here accepted MESH2DX, which the parser
                        # then refused: the sniff must not be looser than the parser.
                        return stripped.split()[0].upper() == _HEADER
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a 2DM file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Node and element ids are collected as the file gives them and resolved to
        contiguous indices through :func:`~pyramids_mesh._text.native_id_map` in a
        second pass, exactly as MDAL does: 2DM ids may have gaps and may be out of
        order, so ``id == index + 1`` would mis-wire every face of a renumbered file.

        Args:
            path: The 2DM file to read.
            lazy: Ignored -- a 2DM file is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, faces or edges, and the material/elevation variables.

        Raises:
            FileFormatNotSupportedError: If the ``MESH2D`` header is missing, an
                unsupported element type appears, the file declares no elements, a
                record is short of the fields its keyword requires, a numeric field is
                not a number, or ``NUM_MATERIALS_PER_ELEM`` declares fewer than one
                column.
            ValueError: If an element references a node id the file never defined, or
                two nodes, two faces or two line elements share an id.
        """
        scan = _Scan()
        scan.run(path)
        return scan.build(path)


def build_minimal_fixture(path: str) -> None:
    """Write the smallest valid 2DM file, for the cross-driver invariant harness.

    This is test support on the installed surface, deliberately: the harness discovers
    it by name on each driver's module, which is what lets a new driver join the
    invariant suite by being registered rather than by anyone writing new tests. It
    writes the same seven lines as the committed ``tests/data/minimal.2dm``, and a test
    asserts the two match so neither can drift.

    Args:
        path: Where to write the file.

    Examples:
        - The fixture is the unit square split into two counter-clockwise
          triangles -- four nodes and two elements, the smallest thing that is
          still a mesh:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import build_minimal_fixture
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.2dm")
            ...     build_minimal_fixture(path)
            ...     with open(path, encoding="utf-8") as handle:
            ...         lines = handle.read().splitlines()
            >>> lines[0]
            'MESH2D'
            >>> len([line for line in lines if line.startswith("ND")])
            4
            >>> [line for line in lines if line.startswith("E3T")]
            ['E3T 1 1 2 3 1', 'E3T 2 1 3 4 1']
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_MINIMAL_FIXTURE)


class _Scan:
    """One 2DM file's records, collected in file order then resolved to indices."""

    def __init__(self) -> None:
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.face_ids: list[int] = []
        self.face_vertices: list[list[int]] = []
        self.face_extras: list[list[str]] = []
        self.edge_ids: list[int] = []
        self.edge_vertices: list[list[int]] = []
        self.materials_per_element = 1
        # Whether any ND record actually carried a z column. A file without one gets no
        # node_z at all rather than a column of zeros, which a consumer could not tell
        # from a real zero datum.
        self.saw_elevation = False

    def run(self, path: str) -> None:
        """Stream the file once, collecting records without resolving any ids.

        Args:
            path: The 2DM file to read.

        Raises:
            FileFormatNotSupportedError: On a missing header or an unsupported element.
        """
        header_seen = False
        for keyword, fields in iter_records(path, comment=_COMMENT):
            upper = keyword.upper()
            if not header_seen:
                if upper != _HEADER:
                    raise FileFormatNotSupportedError(
                        f"{path!r} is not a 2DM file: expected a {_HEADER} header, "
                        f"found {keyword!r}"
                    )
                header_seen = True
                continue
            self._consume(upper, fields, path)
        if not header_seen:
            raise FileFormatNotSupportedError(
                f"{path!r} is empty: a 2DM file starts with a {_HEADER} header"
            )

    def _consume(self, upper: str, fields: list[str], path: str) -> None:
        """Dispatch one record onto the matching collector."""
        if upper in _UNSUPPORTED_ELEMENTS:
            raise FileFormatNotSupportedError(
                f"{path!r} uses the {upper} element, which is not supported; MDAL "
                f"refuses such a file rather than guessing a node mapping"
            )
        if upper == _MATERIALS_TAG:
            _require_fields(upper, fields, 1, path)
            declared = _as_int(fields[0], upper, path)
            if declared < 1:
                raise FileFormatNotSupportedError(
                    f"{path!r} declares {upper} {declared}, which is not a usable "
                    f"column count"
                )
            self.materials_per_element = declared
        elif upper == _NODE:
            self._add_node(fields, path)
        elif upper in _FACE_NODE_COUNTS:
            self._add_face(upper, fields, path)
        elif upper == _LINE_ELEMENT:
            _require_fields(upper, fields, 3, path)
            self.edge_ids.append(_as_int(fields[0], upper, path))
            self.edge_vertices.append(
                [_as_int(value, upper, path) for value in fields[1:3]]
            )

    def _add_node(self, fields: list[str], path: str) -> None:
        """Collect one ``ND <id> <x> <y> <z>`` record.

        Args:
            fields: The record's fields, without the keyword.
            path: The source path, for the error message.

        Raises:
            FileFormatNotSupportedError: If the record is short of an id and both
                coordinates.
        """
        _require_fields(_NODE, fields, 3, path)
        self.node_ids.append(_as_int(fields[0], _NODE, path))
        self.node_x.append(_as_float(fields[1], _NODE, path))
        self.node_y.append(_as_float(fields[2], _NODE, path))
        if len(fields) > 3:
            self.saw_elevation = True
            self.node_z.append(_as_float(fields[3], _NODE, path))
        else:
            self.node_z.append(0.0)

    def _add_face(self, upper: str, fields: list[str], path: str) -> None:
        """Collect one face record, keeping its trailing columns unsliced.

        The material and aux columns are stored verbatim because
        ``NUM_MATERIALS_PER_ELEM`` may appear after the elements it applies to; the
        split happens once the whole file has been read. Fields are only ever indexed
        from the start, since the column count varies with the aux columns.

        Args:
            upper: The element keyword, upper-cased.
            fields: The record's fields, without the keyword.
            path: The source path, for the error message.

        Raises:
            FileFormatNotSupportedError: If the record carries fewer vertex ids than
                the element type declares.
        """
        count = _FACE_NODE_COUNTS[upper]
        _require_fields(upper, fields, 1 + count, path)
        self.face_ids.append(_as_int(fields[0], upper, path))
        self.face_vertices.append(
            [_as_int(value, upper, path) for value in fields[1 : 1 + count]]
        )
        self.face_extras.append(fields[1 + count :])

    def build(self, path: str) -> ParsedMesh:
        """Resolve collected ids to indices and assemble the parsed mesh.

        Args:
            path: The source path, for error messages.

        Returns:
            ParsedMesh: The assembled mesh or 1D network.

        Raises:
            FileFormatNotSupportedError: If the file declares no elements at all.
            ValueError: If an element references an undefined node id.
        """
        if not self.face_vertices and not self.edge_vertices:
            raise FileFormatNotSupportedError(
                f"{path!r} declares no elements: no E3T/E4Q/E6T face and no E2L line"
            )
        node_map = native_id_map(self.node_ids)
        # Built for their duplicate-id check only: nothing in a 2DM references an
        # element by id, but two elements sharing one means a malformed file, and a
        # face and a line element are held to the same standard.
        native_id_map(self.face_ids)
        native_id_map(self.edge_ids)
        faces = self._resolve_faces(node_map, path)
        edges = self._resolve_edges(node_map, path)
        return self._assemble(path, faces, edges)

    def _resolve_faces(self, node_map: dict[int, int], path: str) -> np.ndarray | None:
        """Map face vertex ids to 0-based indices, padding short rows with ``-1``."""
        if not self.face_vertices:
            return None
        max_nodes = max(len(vertices) for vertices in self.face_vertices)
        faces = np.full((len(self.face_vertices), max_nodes), -1, dtype=np.int64)
        for row, vertices in enumerate(self.face_vertices):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, path)
        return faces

    def _resolve_edges(self, node_map: dict[int, int], path: str) -> np.ndarray | None:
        """Map E2L vertex ids to 0-based indices."""
        if not self.edge_vertices:
            return None
        edges = np.empty((len(self.edge_vertices), 2), dtype=np.int64)
        for row, vertices in enumerate(self.edge_vertices):
            for column, native in enumerate(vertices):
                edges[row, column] = _index_of(native, node_map, path)
        return edges

    def _assemble(
        self, path: str, faces: np.ndarray | None, edges: np.ndarray | None
    ) -> ParsedMesh:
        """Build the :class:`ParsedMesh` from resolved connectivity."""
        variables: dict[str, ParsedVariable] = {}
        if faces is not None:
            variables.update(self._material_variables(len(faces), path))
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            # E2L elements are a separate 1D topology, not this surface's edges, so
            # they never reach `Mesh2d.edge_node_connectivity`.
            network_edges=edges,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map, so
            # there is nothing for the validator to subtract. The file's own 1-based
            # numbering is recorded separately so a writer can restore it.
            start_index=0,
            original_start_index=1,
            node_z=(
                np.array(self.node_z, dtype=np.float64) if self.saw_elevation else None
            ),
            crs=None,
            variables=variables,
            global_attributes={"Conventions": "UGRID-1.0", "source": "SMS 2DM"},
            topology_dimension=1 if faces is None else 2,
            mesh_name="mesh2d",
        )

    def _material_variables(self, n_face: int, path: str) -> dict[str, ParsedVariable]:
        """Split the trailing columns into one face variable per material column.

        An element short of the declared material columns contributes ``0.0`` for the
        missing ones rather than failing the read, since a writer omitting a trailing
        column is common and the alternative is refusing an otherwise sound file.

        The columns are float64 even though a material is an integer id: the id is
        exact in float64, writers are not consistent about emitting ``7`` rather than
        ``7.0``, and refusing a file over that spelling would be worse than carrying it.

        Args:
            n_face: The face count, for each variable's shape.
            path: The source path, for error messages.

        Returns:
            dict[str, ParsedVariable]: ``material`` plus ``material_2``.. as declared.
        """
        variables: dict[str, ParsedVariable] = {}
        for index in range(self.materials_per_element):
            column = [
                (_as_float(extras[index], "E**", path) if len(extras) > index else 0.0)
                for extras in self.face_extras
            ]
            name = "material" if index == 0 else f"material_{index + 1}"
            variables[name] = ParsedVariable(
                location="face",
                shape=(n_face,),
                values=np.array(column, dtype=np.float64),
            )
        return variables


def _as_int(value: str, keyword: str, path: str) -> int:
    """Parse one field as an integer, naming the record and file if it is not one.

    Args:
        value: The field text.
        keyword: The record's keyword, for the error message.
        path: The source path, for the error message.

    Returns:
        int: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not an integer.
    """
    try:
        return int(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected an integer, found "
            f"{value!r}"
        ) from error


def _as_float(value: str, keyword: str, path: str) -> float:
    """Parse one field as a float, naming the record and file if it is not one.

    Args:
        value: The field text.
        keyword: The record's keyword, for the error message.
        path: The source path, for the error message.

    Returns:
        float: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not a number.
    """
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected a number, found "
            f"{value!r}"
        ) from error


def _require_fields(keyword: str, fields: list[str], needed: int, path: str) -> None:
    """Check a record carries at least the fields its keyword requires.

    Without this a truncated line fails with a bare ``IndexError`` deep in the parser,
    which says nothing about which record of which file is malformed.

    Args:
        keyword: The record's keyword, for the error message.
        fields: The record's fields, without the keyword.
        needed: The minimum number of fields the keyword requires.
        path: The source path, for the error message.

    Raises:
        FileFormatNotSupportedError: If fewer than ``needed`` fields are present.
    """
    if len(fields) < needed:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected at least {needed} "
            f"fields after the keyword, found {len(fields)}"
        )


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Return the 0-based index of a native node id.

    Args:
        native: The node id as the file wrote it.
        node_map: The id-to-index mapping built from the ``ND`` records.
        path: The source path, for the error message.

    Returns:
        int: The node's 0-based index.

    Raises:
        ValueError: If the file never defined that node id.
    """
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} references node id {native}, which no ND record defines"
        ) from None
