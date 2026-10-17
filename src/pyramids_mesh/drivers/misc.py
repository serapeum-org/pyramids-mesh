"""One-off drivers with no shared parser: Mike21, with H2i and ADCIRC still to come.

Mike21 ``.mesh`` is plain text despite the DHI name, so it needs no extra and lives in
phase 2 next to 2DM rather than with the DFS drivers. H2i and ADCIRC have no
driver-table entry in the reference landscape, so their support there is unstated --
verify against a real file before implementing. Tasks D4 and D12; see
``planning/design.md`` section 20.1.

The Mike21 structure here was read from the format specification. Two details could
not be checked against a vendor file -- the header's projection field and the node-record
column order -- so they are read as the format specifies and called out in the driver's
documented limitations rather than presented as certain.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import osr

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import native_id_map

# osr exceptions are enabled per call with ``osr.ExceptionMgr`` rather than globally
# at import, so importing the driver does not mutate the interpreter's GDAL state.

# A Mike21 element pads a triangle in a quad mesh with a trailing 0. Node ids are
# 1-based, so 0 is never a real node and is read as the fill marker.
_ELEMENT_PADDING = 0

# A node record is `id x y z boundary_code`; an element header is
# `count max_nodes type_code`. Named rather than magic so the parser reads as the format
# does.
_NODE_FIELDS = 5
_ELEMENT_HEADER_FIELDS = 3

# WKT begins with one of these top-level keywords. A projection field that does not is a
# DHI-internal name (`UTM-32`, `LONG/LAT`, `NON-UTM`), which carries no datum this
# package can turn into a real CRS -- so it is left as None rather than guessed at.
_WKT_PREFIXES = ("PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS", "COMPD_CS", "BOUNDCRS")

# EUM item/unit codes and the per-node boundary code the reader does not keep; written as
# documented defaults on a round trip, since the mesh model carries none of them. The
# element type code is chosen from the face width (triangle vs quad) and is likewise not
# read back. The ``LONG/LAT`` placeholder stands in for a mesh with no resolvable CRS.
_DEFAULT_ITEM = 100079
_DEFAULT_UNIT = 1000
_DEFAULT_BOUNDARY = 1
_MIKE21_ELEMENT_TYPE = {3: 21, 4: 25}
_PLACEHOLDER_PROJECTION = "LONG/LAT"

# The smallest valid Mike21 mesh: the unit square split into two counter-clockwise
# triangles. The projection is a DHI-internal name, so the read leaves `crs` None.
_MINIMAL_MIKE21_FIXTURE = """100079 1000 4 LONG/LAT
1 0.0 0.0 0.0 1
2 1.0 0.0 0.0 1
3 1.0 1.0 0.0 1
4 0.0 1.0 0.0 1
2 3 21
1 1 2 3
2 1 3 4
"""


class Mike21Driver(BaseDriver):
    """Reader for the DHI Mike21 ``.mesh`` frame format.

    Plain text despite the DHI name, so it needs no dependency. A header line, then one
    record per node, then an element header, then one record per element. The format
    carries a mesh frame and no results, so the driver declares ``read_mesh`` only.

    Node elevation becomes a ``node_z`` variable, as it does for 2DM: the pyramids mesh
    model is planar and carries no per-node elevation coordinate (design section 6). The
    file's z column is read as written, never fabricated.

    Documented limitations:
        - **The projection field and the node-record column order are read as the
          format specifies, but could not be checked against a vendor file.** If a real Mike21
          mesh orders its node columns differently, the coordinates would be wrong in a
          plausible way; this is the part of the driver a vendor sample should confirm.
        - **The projection is mapped to a CRS only when it is real.** A WKT string or an
          EPSG code that the vendored PROJ validates is carried; a DHI-internal name
          such as ``UTM-32`` or ``LONG/LAT`` carries no datum and leaves ``crs`` as
          ``None`` rather than being guessed at.
        - The boundary code in each node record is **not carried**. It marks open and
          land boundaries, which this package has nowhere to put on a bare frame; it is
          dropped rather than turned into a variable the mesh model would not recognise.
        - A triangle in a quad mesh is padded with a trailing ``0`` node id. Since ids
          are 1-based, ``0`` is read as fill and becomes ``-1`` padding, never node
          index zero.
        - Node ids are resolved through an id map rather than assumed contiguous, so a
          renumbered file is wired correctly; the 1-based numbering is recorded for a
          writer to restore.

    Examples:
        - Write the smallest valid Mike21 mesh and read it back. The unit square's two
          triangles come through as a 2D mesh, and the 1-based ids arrive 0-based with
          the file's base recorded for a writer:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.misc import (
            ...     Mike21Driver,
            ...     build_minimal_fixture_mike21,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.mesh")
            ...     build_minimal_fixture_mike21(path)
            ...     dataset = Mike21Driver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1

        - Elevation comes back as a node variable, because the mesh model carries no
          per-node elevation coordinate:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.misc import (
            ...     Mike21Driver,
            ...     build_minimal_fixture_mike21,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.mesh")
            ...     build_minimal_fixture_mike21(path)
            ...     dataset = Mike21Driver().read(path)
            >>> dataset.data_variable_names
            ['node_z']
            >>> dataset.get("node_z").location
            'node'

        - A DHI-internal projection name carries no datum, so the CRS is left unset
          rather than guessed:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.misc import (
            ...     Mike21Driver,
            ...     build_minimal_fixture_mike21,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.mesh")
            ...     build_minimal_fixture_mike21(path)
            ...     dataset = Mike21Driver().read(path)
            >>> dataset.crs_wkt is None
            True
    """

    name = "mike21"
    long_name = "DHI Mike21 mesh"
    suffixes = (".mesh",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "write_mesh", "write_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first line parses as a Mike21 header.

        The header is ``<item> <unit> <node_count> <projection...>``, so a readable
        first line whose third field is a positive integer is the cheapest signature
        the format offers. A binary file decodes to replacement characters and fails
        that test rather than raising.

        ``.mesh`` is a common suffix, so the sniff is what keeps a foreign ``.mesh``
        file -- a 3D model export, say -- from being claimed. It is still only a header
        shape: a file that passes it but breaks later is refused by ``read``, which is
        why the two must not disagree about what a header looks like.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first line has the header's shape.
        """
        with open(path, "rb") as handle:
            first = handle.readline(4096)
        tokens = first.decode("utf-8-sig", errors="replace").split()
        if len(tokens) < _ELEMENT_HEADER_FIELDS + 1:
            return False
        try:
            return int(tokens[2]) > 0
        except ValueError:
            return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Mike21 ``.mesh`` file into a :class:`ParsedMesh`.

        Args:
            path: The ``.mesh`` file to read.
            lazy: Ignored -- the format is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, faces and the ``node_z`` elevation variable.

        Raises:
            FileFormatNotSupportedError: If the file is empty, a record is short of its
                columns, a count does not match the records that follow, or a numeric
                field is not a number.
            ValueError: If an element references a node id the file never defined, or
                two nodes share an id.
        """
        return _Mike21Scan(path).build()

    def write(self, dataset, path: str, **options) -> None:
        """Write a mesh frame back to a Mike21 ``.mesh`` file.

        Emits the header line, one ``id x y z boundary`` record per node and one
        ``id n1 n2 n3[ n4]`` record per element, restoring the 1-based numbering the
        reader recorded. Node elevation comes from a node-located ``node_z`` variable when
        present, else ``0``; a triangle in a quad-width table is padded with a trailing
        ``0`` node id, the format's own convention.

        The EUM item and unit codes, the per-node boundary code and the element type code
        are not carried on the mesh model, so they are written as documented defaults; the
        reader does not read them back. A real CRS (EPSG or WKT) is written as the
        projection field, else the DHI-internal ``LONG/LAT`` placeholder.

        Args:
            dataset: The :class:`~pyramids.netcdf.ugrid.UgridDataset` to write.
            path: The output ``.mesh`` path.
            **options: Unused; accepted for a uniform writer signature.

        Raises:
            FileFormatNotSupportedError: If the dataset carries no faces.
        """
        mesh = dataset.mesh
        connectivity = mesh.face_node_connectivity
        if connectivity is None or connectivity.data is None:
            raise FileFormatNotSupportedError(
                f"cannot write {path!r}: the dataset has no face connectivity to write "
                f"as a Mike21 mesh"
            )
        faces = np.asarray(connectivity.data)
        base = connectivity.original_start_index
        if base is None:
            base = 1
        node_x = np.asarray(mesh.node_x, dtype=np.float64)
        node_y = np.asarray(mesh.node_y, dtype=np.float64)
        node_z = _writer_node_z(dataset, len(node_x))
        max_nodes = faces.shape[1] if faces.ndim == 2 else 3
        type_code = _MIKE21_ELEMENT_TYPE.get(max_nodes, _MIKE21_ELEMENT_TYPE[3])
        projection = _crs_to_projection(dataset)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                f"{_DEFAULT_ITEM} {_DEFAULT_UNIT} {len(node_x)} {projection}\n"
            )
            for index in range(len(node_x)):
                z = 0.0 if node_z is None else float(node_z[index])
                handle.write(
                    f"{index + base} {float(node_x[index])!r} "
                    f"{float(node_y[index])!r} {z!r} {_DEFAULT_BOUNDARY}\n"
                )
            handle.write(f"{len(faces)} {max_nodes} {type_code}\n")
            for row_index, row in enumerate(faces):
                vertices = [int(v) + base for v in row if int(v) != -1]
                padded = vertices + [_ELEMENT_PADDING] * (max_nodes - len(vertices))
                ids = " ".join(str(v) for v in padded)
                handle.write(f"{row_index + base} {ids}\n")


def _writer_node_z(dataset, expected: int):
    """Return the ``node_z`` column for a writer, or ``None`` when absent or mis-sized."""
    if "node_z" not in dataset.data_variable_names:
        return None
    values = np.asarray(dataset.get("node_z").data, dtype=np.float64)
    return values if values.shape == (expected,) else None


def _crs_to_projection(dataset) -> str:
    """Return the header projection field: the dataset CRS as WKT, else a placeholder.

    A WKT is collapsed onto one line so it stays within the single-line header the reader
    expects; a dataset with no resolvable CRS writes the DHI-internal ``LONG/LAT`` name,
    which the reader maps back to ``None``.
    """
    wkt = getattr(dataset, "crs_wkt", None)
    if wkt:
        return " ".join(str(wkt).split())
    return _PLACEHOLDER_PROJECTION


def build_minimal_fixture_mike21(path: str) -> None:
    """Write the smallest valid Mike21 mesh, for the cross-driver invariant harness.

    Named per driver because the harness looks for
    ``build_minimal_fixture_<driver name>`` before the shared ``build_minimal_fixture``;
    the ``misc`` module may hold several drivers. It writes the unit square split into
    two counter-clockwise triangles.

    Args:
        path: Where to write the file.

    Examples:
        - The fixture is four nodes and two triangles, the smallest thing that is still
          a mesh:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.misc import build_minimal_fixture_mike21
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.mesh")
            ...     build_minimal_fixture_mike21(path)
            ...     with open(path, encoding="utf-8") as handle:
            ...         lines = handle.read().splitlines()
            >>> lines[0].split()[2]
            '4'
            >>> lines[5]
            '2 3 21'
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_MINIMAL_MIKE21_FIXTURE)


def _projection_to_crs(projection: str) -> str | int | None:
    """Map a Mike21 projection field to a CRS, or ``None`` when it is not a real one.

    A WKT string or an EPSG code the vendored PROJ validates is carried; a DHI-internal
    name carries no datum this package can resolve, so it is left as ``None`` rather than
    guessed at. Validation is what tells the two apart -- the field is not trusted for
    being present.

    Args:
        projection: The projection text from the header, already stripped.

    Returns:
        str | int | None: A validated WKT string, a validated EPSG int, or ``None``.
    """
    if not projection:
        return None
    with osr.ExceptionMgr(useExceptions=True):
        if projection.isdigit():
            reference = osr.SpatialReference()
            try:
                reference.ImportFromEPSG(int(projection))
            except RuntimeError:
                return None
            return int(projection)
        if projection.upper().startswith(_WKT_PREFIXES):
            reference = osr.SpatialReference()
            try:
                reference.ImportFromWkt(projection)
                # Validate() inside the guard: under the scoped ExceptionMgr it raises
                # for an importable-but-unsupported WKT, which must fall back to None.
                valid = reference.Validate() == 0
            except RuntimeError:
                return None
            return projection if valid else None
    # A DHI-internal name such as UTM-32 or LONG/LAT: a label, not a datum.
    return None


class _Mike21Scan:
    """One Mike21 file's records, read in the fixed order the format lays them out."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.lines, self.raw_lines = self._read_lines(path)
        self.cursor = 0

    @staticmethod
    def _read_lines(path: str) -> tuple[list[list[str]], list[str]]:
        """Return the file's non-blank lines, both split into fields and raw.

        The raw lines are kept so the projection field on the header can be preserved
        verbatim: splitting and re-joining on whitespace would normalise any tab or
        multi-space run inside a WKT string before it reaches PROJ.

        ``utf-8-sig`` drops a byte-order mark, as the text helpers do, so a header saved
        by a Windows editor is not refused for that alone.
        """
        split: list[list[str]] = []
        raw: list[str] = []
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    split.append(stripped.split())
                    raw.append(stripped)
        return split, raw

    def _next(self, what: str) -> list[str]:
        """Return the next record, or refuse a file that ends before one is due."""
        if self.cursor >= len(self.lines):
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends before its {what}: the counts in the header "
                f"promise more records than the file holds"
            )
        record = self.lines[self.cursor]
        self.cursor += 1
        return record

    def build(self) -> ParsedMesh:
        """Read header, nodes and elements in order and assemble the parsed mesh.

        Returns:
            ParsedMesh: The nodes, faces and ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: On an empty file, a short record, or a
                non-numeric field.
            ValueError: On an undefined node reference or a duplicate node id.
        """
        if not self.lines:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is empty: a Mike21 mesh starts with a header line"
            )
        node_count, crs = self._read_header()
        node_ids, node_x, node_y, node_z = self._read_nodes(node_count)
        faces = self._read_elements(native_id_map(node_ids))
        return ParsedMesh(
            node_x=np.array(node_x, dtype=np.float64),
            node_y=np.array(node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map, so
            # there is nothing for the validator to subtract. The 1-based file numbering
            # is recorded separately so a writer can restore it.
            start_index=0,
            original_start_index=1,
            node_z=np.array(node_z, dtype=np.float64),
            crs=crs,
            variables={},
            global_attributes={"source": "DHI Mike21 mesh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_header(self) -> tuple[int, str | int | None]:
        """Read ``<item> <unit> <node_count> <projection...>``.

        Returns:
            tuple[int, str | int | None]: The node count and the mapped CRS.
        """
        header = self._next("header")
        if len(header) < _ELEMENT_HEADER_FIELDS + 1:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short header: expected at least "
                f"{_ELEMENT_HEADER_FIELDS + 1} fields, found {len(header)}"
            )
        node_count = _as_int(header[2], "header", self.path)
        # The projection is everything after the third field, taken from the raw header
        # line so a WKT string keeps its internal spacing rather than being re-joined on
        # single spaces. The header is always the first non-blank line, so raw_lines[0].
        raw_header = self.raw_lines[0]
        projection = raw_header.split(maxsplit=_ELEMENT_HEADER_FIELDS)
        projection_text = (
            projection[_ELEMENT_HEADER_FIELDS]
            if len(projection) > _ELEMENT_HEADER_FIELDS
            else ""
        )
        return node_count, _projection_to_crs(projection_text.strip())

    def _read_nodes(
        self, node_count: int
    ) -> tuple[list[int], list[float], list[float], list[float]]:
        """Read ``node_count`` records of ``id x y z boundary_code``.

        The boundary code is read to validate the record width but is not kept -- see
        the driver's documented limitations.
        """
        ids: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for _ in range(node_count):
            record = self._next("node records")
            if len(record) < _NODE_FIELDS:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short node record: expected {_NODE_FIELDS} "
                    f"fields (id x y z boundary), found {len(record)}"
                )
            ids.append(_as_int(record[0], "node", self.path))
            xs.append(_as_float(record[1], "node", self.path))
            ys.append(_as_float(record[2], "node", self.path))
            zs.append(_as_float(record[3], "node", self.path))
        return ids, xs, ys, zs

    def _read_elements(self, node_map: dict[int, int]) -> np.ndarray:
        """Read the element header and ``count`` element records into connectivity.

        A trailing ``0`` node id is the format's padding for a triangle in a quad mesh;
        it becomes ``-1`` fill. Every other id is resolved through ``node_map``.
        """
        header = self._next("element header")
        if len(header) < _ELEMENT_HEADER_FIELDS:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a short element header: expected "
                f"{_ELEMENT_HEADER_FIELDS} fields (count max_nodes type), found "
                f"{len(header)}"
            )
        count = _as_int(header[0], "element header", self.path)
        max_nodes = _as_int(header[1], "element header", self.path)
        if count <= 0 or max_nodes < 3:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares {count} element(s) of up to {max_nodes} "
                f"node(s), which is not a usable element block"
            )
        faces = np.full((count, max_nodes), -1, dtype=np.int64)
        for row in range(count):
            record = self._next("element records")
            if len(record) < 1 + 3:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short element record: a face needs at least "
                    f"three node ids, found {len(record) - 1}"
                )
            for column, field in enumerate(record[1 : 1 + max_nodes]):
                native = _as_int(field, "element", self.path)
                if native == _ELEMENT_PADDING:
                    continue
                faces[row, column] = _index_of(native, node_map, self.path)
        return faces


def _as_int(value: str, what: str, path: str) -> int:
    """Parse one field as an integer, naming the record and file if it is not one."""
    try:
        return int(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {what} record: expected an integer, found "
            f"{value!r}"
        ) from error


def _as_float(value: str, what: str, path: str) -> float:
    """Parse one field as a float, naming the record and file if it is not one."""
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {what} record: expected a number, found "
            f"{value!r}"
        ) from error


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Return the 0-based index of a native node id, or refuse an undefined one."""
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} references node id {native}, which no node record defines"
        ) from None
