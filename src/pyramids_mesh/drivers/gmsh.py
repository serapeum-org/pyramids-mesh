"""Gmsh ``.msh`` reader (task C3).

Reads the Gmsh mesh format into ``pyramids.netcdf.ugrid`` objects. Both the legacy
**v2.2** record layout and the current **v4.1** entity-block layout are supported, in
both their **ASCII** and **binary** encodings (``file-type`` ``0`` and ``1`` in the
``$MeshFormat`` header). A binary file carries a one-``int`` endianness marker and packs
its node and element records; the reader honours the marker and decodes the records into
the same node and face tables the ASCII path produces.

Gmsh is a generic mesh generator, so its coordinates are engineering coordinates with
no datum: ``crs`` is always ``None`` and is never guessed from the magnitudes. Only
surface elements become faces -- triangles (type ``2``) and quadrilaterals (type ``3``).
Lower-dimensional lines (type ``1``) and points (type ``15``) are the boundary scaffolding
Gmsh writes alongside the surface, and are dropped: feeding them to the face table would
mislabel one-dimensional entities as mesh faces. Node tags may be gapped or out of order,
so connectivity is resolved through a tag->index map, never by assuming ``tag == index + 1``.

See ``planning/design.md``; this driver follows the 2DM reference shape (task D1).
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import struct
from collections.abc import Iterator

import numpy as np
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records, native_id_map

_HEADER = "$MeshFormat"
_NODES = "$Nodes"
_END_NODES = "$EndNodes"
_ELEMENTS = "$Elements"
_END_ELEMENTS = "$EndElements"

# ASCII is file-type 0; 1 is the packed binary stream. Both are decoded.
_ASCII_FILE_TYPE = 0
_BINARY_FILE_TYPE = 1

# Byte widths of the fixed-size fields in a binary file: an ``int`` is 4 bytes and a
# ``double`` 8 (``_DOUBLE_SIZE`` doubles as the default ``data-size`` when the header
# omits it). The v4.1 layout also uses a ``size_t`` whose width is the header's
# ``data-size`` field (8 in every file Gmsh writes).
_INT_SIZE = 4
_DOUBLE_SIZE = 8

# Major versions with a layout this reader understands: 2 is the v2.2 record form, 4 the
# v4.1 entity-block form. A 1.x or 3.x file is refused rather than read as one of these.
_SUPPORTED_MAJORS = frozenset({2, 4})

# Node count per Gmsh element type code, for the types this reader can account for. The
# count is what lets a v2.2 element line and a v4.1 element block be split correctly; an
# element of any other type cannot be sized, so the file is refused rather than guessed at.
_ELEMENT_NODE_COUNTS = {1: 2, 2: 3, 3: 4, 4: 4, 15: 1}

# The element types that become mesh faces: a 3-node triangle and a 4-node quadrilateral.
# Everything else in ``_ELEMENT_NODE_COUNTS`` is a non-face element (line, point, tetra)
# and is consumed but not kept.
_FACE_TYPES = frozenset({2, 3})


def _as_int(value: str, context: str, path: str) -> int:
    """Parse one integer field, naming the file and record on failure.

    Args:
        value: The raw token to convert.
        context: What the token was expected to be, for the error message.
        path: The source path, for the error message.

    Returns:
        int: The parsed integer.

    Raises:
        FileFormatNotSupportedError: If the token is not an integer.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {context}: expected an integer, found {value!r}"
        ) from None


def _as_float(value: str, context: str, path: str) -> float:
    """Parse one float field, naming the file and record on failure.

    Args:
        value: The raw token to convert.
        context: What the token was expected to be, for the error message.
        path: The source path, for the error message.

    Returns:
        float: The parsed float.

    Raises:
        FileFormatNotSupportedError: If the token is not a number.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {context}: expected a number, found {value!r}"
        ) from None


def _node_count(element_type: int, path: str) -> int:
    """Return the node count for an element type, refusing an unaccounted one.

    Args:
        element_type: The Gmsh element type code.
        path: The source path, for the error message.

    Returns:
        int: The number of node tags an element of this type carries.

    Raises:
        FileFormatNotSupportedError: If the type has no known node count, since the
            record then cannot be sized and everything after it would misalign.
    """
    if element_type not in _ELEMENT_NODE_COUNTS:
        raise FileFormatNotSupportedError(
            f"{path!r} uses element type {element_type}, which this reader does "
            f"not account for; such a file is refused rather than misread"
        )
    return _ELEMENT_NODE_COUNTS[element_type]


def _mesh_format(path: str) -> tuple[int, int, int]:
    """Read the ``$MeshFormat`` header and return ``(major, file_type, data_size)``.

    The header is ASCII in both encodings, so it is read with the text record walk even
    for a binary file. The values decide which reader -- ASCII or binary -- and which
    byte widths the rest of the file uses.

    Args:
        path: The ``.msh`` file to inspect.

    Returns:
        tuple[int, int, int]: The major version, the file-type (``0`` ASCII, ``1``
            binary) and the ``data-size`` (the ``size_t`` width the v4.1 binary layout
            uses).

    Raises:
        FileFormatNotSupportedError: If the file is empty, its first record is not
            ``$MeshFormat``, or the version line is missing or non-numeric.
    """
    records = iter_records(path)
    try:
        keyword, _fields = next(records)
    except StopIteration:
        raise FileFormatNotSupportedError(
            f"{path!r} is empty, so it is not a Gmsh file"
        ) from None
    if keyword != _HEADER:
        raise FileFormatNotSupportedError(
            f"{path!r} is not a Gmsh file: expected a {_HEADER} header, found {keyword!r}"
        )
    try:
        version, rest = next(records)
    except StopIteration:
        raise FileFormatNotSupportedError(
            f"{path!r} ends while reading the $MeshFormat version line"
        ) from None
    major = int(_as_float(version, "Gmsh version", path))
    file_type = _as_int(rest[0], "file-type", path) if rest else _ASCII_FILE_TYPE
    data_size = _as_int(rest[1], "data-size", path) if len(rest) > 1 else _DOUBLE_SIZE
    return major, file_type, data_size


def _build_parsed_mesh(
    path: str,
    node_tags: list[int],
    node_x: list[float],
    node_y: list[float],
    node_z: list[float],
    face_tags: list[list[int]],
) -> ParsedMesh:
    """Resolve collected node tags and faces into a parsed surface mesh.

    Shared by the ASCII and binary readers: both fill the same five lists, so the final
    tag resolution, winding-neutral assembly and ``node_z`` decision live here once.

    Args:
        path: The source path, for error messages.
        node_tags: Node tags in file order.
        node_x: Node x-coordinates, aligned with ``node_tags``.
        node_y: Node y-coordinates, aligned with ``node_tags``.
        node_z: Node z-coordinates, aligned with ``node_tags``.
        face_tags: One list of node tags per triangular or quadrilateral face.

    Returns:
        ParsedMesh: The assembled mesh.

    Raises:
        FileFormatNotSupportedError: If no triangular or quadrilateral face was read.
        ValueError: If two nodes share a tag, or a face references an undefined tag.
    """
    if not face_tags:
        raise FileFormatNotSupportedError(
            f"{path!r} declares no triangular or quadrilateral faces, so it has "
            f"no surface to build"
        )
    tag_to_index = native_id_map(node_tags)
    faces = _resolve_faces(path, face_tags, tag_to_index)
    node_z_array = np.array(node_z, dtype=np.float64)
    return ParsedMesh(
        node_x=np.array(node_x, dtype=np.float64),
        node_y=np.array(node_y, dtype=np.float64),
        faces=faces,
        fill_value=-1,
        # Connectivity is already 0-based: it was resolved through the tag map, so the
        # validator has nothing to subtract. Gmsh's conventional 1-based tags are
        # recorded for a writer.
        start_index=0,
        original_start_index=1,
        # ``.any()`` keeps the elevation only when some z is non-zero; it treats an
        # exact 0.0 as absent without a float-equality check on the coordinates.
        node_z=node_z_array if node_z_array.any() else None,
        crs=None,
        global_attributes={"source": "Gmsh"},
        topology_dimension=2,
        mesh_name="mesh2d",
    )


def _resolve_faces(
    path: str, face_tags: list[list[int]], tag_to_index: dict[int, int]
) -> np.ndarray:
    """Map face node tags to 0-based indices, padding short rows with ``-1``.

    Args:
        path: The source path, for the error message.
        face_tags: One list of node tags per face.
        tag_to_index: The node-tag to 0-based-index map.

    Returns:
        np.ndarray: The ``(n_face, max_nodes)`` connectivity, fill-padded.

    Raises:
        ValueError: If a face references a node tag no node record defined.
    """
    max_nodes = max(len(row) for row in face_tags)
    faces = np.full((len(face_tags), max_nodes), -1, dtype=np.int64)
    for row, tags in enumerate(face_tags):
        for column, tag in enumerate(tags):
            if tag not in tag_to_index:
                raise ValueError(
                    f"{path!r} has an element referencing node tag {tag}, which "
                    f"no node record defines"
                )
            faces[row, column] = tag_to_index[tag]
    return faces


class GmshDriver(BaseDriver):
    """Reader for the Gmsh ``.msh`` mesh format (v2.2 and v4.1, ASCII and binary).

    Reads node coordinates and the surface connectivity, mapping triangle (type ``2``)
    and quadrilateral (type ``3``) elements to ``face_node_connectivity``; a mixed file
    pads its triangles to the quad width with ``-1``. Node elevation is carried as a
    ``node_z`` variable only when some node has a non-zero z, since a planar mesh gains
    nothing from an all-zero column.

    Both encodings are read. An ASCII file (``file-type`` ``0``) is parsed as text; a
    binary file (``file-type`` ``1``) is decoded from its packed records, honouring the
    ``$MeshFormat`` endianness marker, so a mesh written binary reads to the same result
    as its ASCII equivalent.

    Documented limitations:
        - **Only v2.2 and v4.1 are read.** The major version is taken from the header and
          must be ``2`` or ``4``; a 1.x or 3.x file is refused by name.
        - **Lines and points are dropped.** Type ``1`` line and type ``15`` point elements
          are the boundary scaffolding Gmsh writes beside the surface; they are consumed
          but never folded into the face table, which would mislabel them as faces.
        - **An element of an unaccounted type refuses the file.** Its node count is
          unknown, so the record cannot be sized and reading on would misalign everything
          after it.
        - **A file with no triangular or quadrilateral element is refused.** There is no
          surface to build, and returning an empty mesh would be a silent wrong answer.
        - **Node tags may be gapped or out of order.** They are resolved through a
          tag->index map, so a renumbered file stays wired correctly; the Gmsh base is
          recorded for a writer.
        - Gmsh carries no CRS, so ``crs`` is always ``None``; it is never guessed from the
          coordinate magnitudes. Physical-name and entity blocks are skipped.

    Examples:
        - A tiny v2.2 file with a boundary line and two triangles reads to two faces and
          four nodes; the line is skipped and the 1-based node tags arrive 0-based:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.gmsh import GmshDriver
            >>> text = '''$MeshFormat
            ... 2.2 0 8
            ... $EndMeshFormat
            ... $Nodes
            ... 4
            ... 1 0.0 0.0 0.0
            ... 2 1.0 0.0 0.0
            ... 3 1.0 1.0 0.0
            ... 4 0.0 1.0 0.0
            ... $EndNodes
            ... $Elements
            ... 3
            ... 1 1 2 0 1 1 2
            ... 2 2 2 0 1 1 2 3
            ... 3 2 2 0 1 1 3 4
            ... $EndElements
            ... '''
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "mesh.msh")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write(text)
            ...     dataset = GmshDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]

        - Dispatch checks the suffix and sniffs the ``$MeshFormat`` header, so junk under
          a ``.msh`` suffix is declined:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.gmsh import GmshDriver
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "junk.msh")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write("not a gmsh file at all")
            ...     GmshDriver().can_open(path)
            False
    """

    name = "gmsh"
    long_name = "Gmsh mesh"
    suffixes = (".msh",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens with a ``$MeshFormat`` header this reader reads.

        The first significant line must be ``$MeshFormat`` and the next must declare a
        major version in :data:`_SUPPORTED_MAJORS`. Never raises -- an empty file, or an
        unparsable version, answers ``False`` -- so ``can_open`` stays within its
        never-raise contract and dispatch is never aborted by a broken candidate. The
        header is ASCII in both encodings, so a binary file is recognised here too.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the header names a readable Gmsh version.
        """
        records = iter_records(path)
        try:
            keyword, _fields = next(records)
        except StopIteration:
            return False
        if keyword != _HEADER:
            return False
        try:
            version, _rest = next(records)
        except StopIteration:
            return False
        try:
            major = int(float(version))
        except ValueError:
            return False
        return major in _SUPPORTED_MAJORS

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Gmsh file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        The ``$MeshFormat`` header decides the path: an ASCII file is parsed as text, a
        binary file is decoded from its packed records.

        Args:
            path: The ``.msh`` file to read.
            lazy: Ignored -- a Gmsh file is read eagerly.

        Returns:
            ParsedMesh: Nodes, the triangular/quadrilateral face connectivity and, when
                present, a ``node_z`` elevation variable.

        Raises:
            FileFormatNotSupportedError: If the header is missing, the version or
                file-type is unsupported, an element type cannot be sized, a record is
                short, truncated or non-numeric, or the file declares no faces.
            ValueError: If two nodes share a tag, which makes the tag->index map ambiguous.
        """
        major, file_type, data_size = _mesh_format(path)
        if major not in _SUPPORTED_MAJORS:
            raise FileFormatNotSupportedError(
                f"{path!r} declares Gmsh version {major}; only the 2.2 and 4.1 layouts "
                f"are read"
            )
        if file_type not in (_ASCII_FILE_TYPE, _BINARY_FILE_TYPE):
            raise FileFormatNotSupportedError(
                f"{path!r} declares file-type {file_type}; only ASCII (0) and binary (1) "
                f"are known"
            )
        if file_type == _BINARY_FILE_TYPE:
            return _GmshBinaryReader(path, major, data_size).build()
        return _GmshReader(path).build()


class _GmshReader:
    """One ASCII Gmsh file: stream its significant lines and assemble the mesh."""

    def __init__(self, path: str) -> None:
        """Prepare to read ``path``, collecting records before resolving any tag.

        Args:
            path: The ``.msh`` file to read.
        """
        self.path = path
        self._lines: Iterator[list[str]] = (
            [keyword, *fields] for keyword, fields in iter_records(path)
        )
        self.node_tags: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.face_tags: list[list[int]] = []

    def build(self) -> ParsedMesh:
        """Read the whole file and resolve it into a parsed mesh.

        Returns:
            ParsedMesh: The assembled surface mesh.

        Raises:
            FileFormatNotSupportedError: On a missing header, an unsupported version, an
                unaccounted element type, a malformed record, or a file with no
                triangular or quadrilateral face.
            ValueError: If two nodes share a tag.
        """
        major = self._read_header()
        for line in self._lines:
            if line[0] == _NODES:
                self._read_nodes(major)
            elif line[0] == _ELEMENTS:
                self._read_elements(major)
        return _build_parsed_mesh(
            self.path,
            self.node_tags,
            self.node_x,
            self.node_y,
            self.node_z,
            self.face_tags,
        )

    def _next(self, context: str) -> list[str]:
        """Return the next significant line's tokens, or refuse a truncated file.

        Args:
            context: What was being read, for the error message.

        Returns:
            list[str]: The whitespace-separated tokens of the next significant line.

        Raises:
            FileFormatNotSupportedError: If the file ends where more was expected.
        """
        try:
            return next(self._lines)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends while reading the {context}"
            ) from None

    def _read_header(self) -> int:
        """Consume ``$MeshFormat`` and return the major version.

        :func:`_mesh_format` has already vetted the header, version and file-type before
        this reader was chosen, so this only walks the two header lines -- to position the
        stream at the first section -- and returns the major that drives node/element
        layout.

        Returns:
            int: The major version, ``2`` or ``4``.
        """
        self._next("$MeshFormat header")
        fields = self._next("version line")
        return int(_as_float(fields[0], "Gmsh version", self.path))

    def _read_nodes(self, major: int) -> None:
        """Read the ``$Nodes`` section in the layout the version dictates."""
        if major == 2:
            self._read_nodes_v2()
        else:
            self._read_nodes_v4()

    def _read_nodes_v2(self) -> None:
        """Read v2.2 nodes: a count, then one ``tag x y z`` record per node."""
        count = _as_int(self._next("node count")[0], "node count", self.path)
        for _ in range(count):
            fields = self._next("node record")
            if len(fields) < 4:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short node record: expected 'tag x y z', "
                    f"found {fields!r}"
                )
            self._add_node(fields[0], fields[1], fields[2], fields[3])

    def _read_nodes_v4(self) -> None:
        """Read v4.1 nodes: entity blocks, each a tag list then a matching coord list.

        Each block header is ``entityDim entityTag parametric numNodesInBlock``; the tags
        come first, all of them, then the coordinates in the same order. The two halves
        are zipped back together here.
        """
        header = self._next("node section header")
        num_blocks = _as_int(header[0], "node block count", self.path)
        for _ in range(num_blocks):
            block = self._next("node block header")
            in_block = _as_int(block[3], "node block size", self.path)
            tags = [
                _as_int(self._next("node tag")[0], "node tag", self.path)
                for _ in range(in_block)
            ]
            for tag in tags:
                coords = self._next("node coordinates")
                if len(coords) < 3:
                    raise FileFormatNotSupportedError(
                        f"{self.path!r} has a short node coordinate line: expected "
                        f"'x y z', found {coords!r}"
                    )
                self._add_node(str(tag), coords[0], coords[1], coords[2])

    def _add_node(self, tag: str, x: str, y: str, z: str) -> None:
        """Collect one node's tag and coordinates.

        Args:
            tag: The node tag.
            x: The x-coordinate.
            y: The y-coordinate.
            z: The z-coordinate.
        """
        self.node_tags.append(_as_int(tag, "node tag", self.path))
        self.node_x.append(_as_float(x, "node x", self.path))
        self.node_y.append(_as_float(y, "node y", self.path))
        self.node_z.append(_as_float(z, "node z", self.path))

    def _read_elements(self, major: int) -> None:
        """Read the ``$Elements`` section in the layout the version dictates."""
        if major == 2:
            self._read_elements_v2()
        else:
            self._read_elements_v4()

    def _read_elements_v2(self) -> None:
        """Read v2.2 elements: a count, then ``tag type numTags tags... nodes...`` rows.

        A face element's node tags are kept; a line or point is consumed and dropped.
        """
        count = _as_int(self._next("element count")[0], "element count", self.path)
        for _ in range(count):
            fields = self._next("element record")
            element_type = _as_int(fields[1], "element type", self.path)
            node_count = _node_count(element_type, self.path)
            num_tags = _as_int(fields[2], "element tag count", self.path)
            start = 3 + num_tags
            nodes = fields[start : start + node_count]
            if len(nodes) < node_count:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short element record for type {element_type}: "
                    f"expected {node_count} node tag(s), found {len(nodes)}"
                )
            self._keep_face(element_type, nodes)

    def _read_elements_v4(self) -> None:
        """Read v4.1 elements: entity blocks of ``elementTag nodeTag...`` rows.

        The block header ``entityDim entityTag elementType numInBlock`` fixes the type
        and node count for every row in the block, so an unaccounted type is refused
        before its rows are read rather than silently misaligning the stream.
        """
        header = self._next("element section header")
        num_blocks = _as_int(header[0], "element block count", self.path)
        for _ in range(num_blocks):
            block = self._next("element block header")
            element_type = _as_int(block[2], "element type", self.path)
            node_count = _node_count(element_type, self.path)
            in_block = _as_int(block[3], "element block size", self.path)
            for _ in range(in_block):
                fields = self._next("element record")
                nodes = fields[1 : 1 + node_count]
                if len(nodes) < node_count:
                    raise FileFormatNotSupportedError(
                        f"{self.path!r} has a short element record for type "
                        f"{element_type}: expected {node_count} node tag(s), found "
                        f"{len(nodes)}"
                    )
                self._keep_face(element_type, nodes)

    def _keep_face(self, element_type: int, nodes: list[str]) -> None:
        """Keep a face element's node tags; a non-face element is dropped.

        Args:
            element_type: The Gmsh element type code.
            nodes: The element's node-tag tokens.
        """
        if element_type in _FACE_TYPES:
            self.face_tags.append(
                [_as_int(tag, "element node tag", self.path) for tag in nodes]
            )


class _GmshBinaryReader:
    """One binary Gmsh file (file-type 1): decode its packed records into a mesh.

    A binary ``.msh`` keeps the ASCII section markers (``$Nodes`` ... ``$EndNodes``) but
    packs the records between them. The ``$MeshFormat`` header carries a one-``int``
    endianness marker; the **v2.2** layout then writes each section's count as ASCII and
    its body as packed ``int``/``double`` records, while the **v4.1** layout writes the
    whole ``$Nodes``/``$Elements`` section -- entity-block headers and bodies alike -- in
    binary ``size_t``/``int``/``double`` fields. Both reduce to the node and face tables
    the ASCII reader produces, so the assembly step is shared with it.

    Only ``$Nodes`` and ``$Elements`` are decoded; any other section (``$Entities``,
    ``$PhysicalNames`` ...) is located by its ``$End`` marker and skipped, so its packed
    body is never mistaken for a record this reader cares about.
    """

    def __init__(self, path: str, major: int, data_size: int) -> None:
        """Read ``path`` into memory, ready to slice its sections.

        Args:
            path: The ``.msh`` file to read.
            major: The major version, ``2`` or ``4``.
            data_size: The ``size_t`` width the v4.1 layout uses (8 in practice).
        """
        self.path = path
        self.major = major
        # The v4.1 ``size_t`` width. 8 bytes -> 'q'; a 4-byte size_t would be 'i'.
        self._size_t = "q" if data_size == _DOUBLE_SIZE else "i"
        with open(path, "rb") as handle:
            self.raw = handle.read()
        self.endian = "<"
        self.node_tags: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.face_tags: list[list[int]] = []

    def build(self) -> ParsedMesh:
        """Decode the ``$Nodes`` and ``$Elements`` sections and assemble the mesh.

        Returns:
            ParsedMesh: The assembled surface mesh.

        Raises:
            FileFormatNotSupportedError: On a truncated section, an unaccounted element
                type, or a file with no triangular or quadrilateral face.
            ValueError: If two nodes share a tag.
        """
        self._read_endianness()
        nodes = self._section(_NODES)
        if nodes is not None:
            self._read_nodes(nodes)
        elements = self._section(_ELEMENTS)
        if elements is not None:
            self._read_elements(elements)
        return _build_parsed_mesh(
            self.path,
            self.node_tags,
            self.node_x,
            self.node_y,
            self.node_z,
            self.face_tags,
        )

    def _read_endianness(self) -> None:
        """Set byte order from the one-``int`` marker that follows the version line.

        Gmsh writes a binary integer equal to ``1`` right after the ``$MeshFormat``
        version line; reading it as little- then big-endian tells the two apart. A file
        whose marker cannot be read as ``1`` either way keeps the little-endian default,
        which is what every mainstream platform writes. The header's presence was already
        confirmed by :func:`_mesh_format`, so it is found here.
        """
        begin = self.raw.find(b"$MeshFormat")
        end_keyword = self.raw.find(b"\n", begin)
        end_version = self.raw.find(b"\n", end_keyword + 1)
        marker = self.raw[end_version + 1 : end_version + 1 + _INT_SIZE]
        if len(marker) == _INT_SIZE:
            if struct.unpack("<i", marker)[0] == 1:
                self.endian = "<"
            elif struct.unpack(">i", marker)[0] == 1:
                self.endian = ">"

    def _section(self, keyword: str) -> bytes | None:
        """Return the raw bytes of a section's body, or ``None`` if it is absent.

        The body is what lies between the ``keyword`` marker line and its matching
        ``$End`` marker. Searching for ``$End<name>`` is how an interleaved binary body
        is bounded without decoding the sections before it.

        Args:
            keyword: The section marker, e.g. ``"$Nodes"``.

        Returns:
            bytes | None: The section body, or ``None`` when the section is not present.

        Raises:
            FileFormatNotSupportedError: If the opening marker has no matching ``$End``.
        """
        start = (keyword + "\n").encode("latin-1")
        begin = self.raw.find(start)
        if begin == -1:
            return None
        begin += len(start)
        end_marker = ("\n$End" + keyword[1:]).encode("latin-1")
        stop = self.raw.find(end_marker, begin)
        if stop == -1:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a {keyword} section with no matching "
                f"{end_marker.decode('latin-1').strip()} marker; it is truncated"
            )
        return self.raw[begin:stop]

    def _unpack(self, fmt: str, body: bytes, offset: int) -> tuple[tuple, int]:
        """Unpack one struct from ``body`` at ``offset``, refusing a short read.

        Args:
            fmt: A struct format string, already prefixed with the byte order.
            body: The section bytes.
            offset: Where in ``body`` to read from.

        Returns:
            tuple[tuple, int]: The unpacked values and the offset just past them.

        Raises:
            FileFormatNotSupportedError: If the body ends inside the record.
        """
        size = struct.calcsize(fmt)
        if offset + size > len(body):
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends inside a binary record; the section is truncated"
            )
        return struct.unpack_from(fmt, body, offset), offset + size

    def _read_nodes(self, body: bytes) -> None:
        """Decode the ``$Nodes`` body in the layout the version dictates."""
        if self.major == 2:
            self._read_nodes_v2(body)
        else:
            self._read_nodes_v4(body)

    def _read_nodes_v2(self, body: bytes) -> None:
        """Decode v2.2 binary nodes: an ASCII count, then ``int`` + 3 ``double`` records."""
        newline = body.find(b"\n")
        count = _as_int(
            body[:newline].decode("latin-1").strip(), "node count", self.path
        )
        values, _ = self._unpack(self.endian + ("i3d" * count), body, newline + 1)
        for index in range(count):
            base = index * 4
            self._add_node(
                values[base], values[base + 1], values[base + 2], values[base + 3]
            )

    def _read_nodes_v4(self, body: bytes) -> None:
        """Decode v4.1 binary nodes: entity blocks of ``size_t`` tags then ``double`` xyz.

        The section header is four ``size_t`` fields; each block then declares three
        ``int`` fields and a ``size_t`` count, followed by that many tags and three times
        as many coordinates.
        """
        size_t = self._size_t
        order = self.endian
        (num_blocks, _total, _min_tag, _max_tag), offset = self._unpack(
            order + (size_t * 4), body, 0
        )
        for _ in range(num_blocks):
            (_dim, _tag, _parametric), offset = self._unpack(order + "3i", body, offset)
            (in_block,), offset = self._unpack(order + size_t, body, offset)
            tags, offset = self._unpack(order + (size_t * in_block), body, offset)
            coords, offset = self._unpack(order + ("d" * (3 * in_block)), body, offset)
            for index, tag in enumerate(tags):
                self._add_node(
                    tag, coords[3 * index], coords[3 * index + 1], coords[3 * index + 2]
                )

    def _add_node(self, tag: int, x: float, y: float, z: float) -> None:
        """Collect one node's tag and coordinates.

        Args:
            tag: The node tag.
            x: The x-coordinate.
            y: The y-coordinate.
            z: The z-coordinate.
        """
        self.node_tags.append(int(tag))
        self.node_x.append(float(x))
        self.node_y.append(float(y))
        self.node_z.append(float(z))

    def _read_elements(self, body: bytes) -> None:
        """Decode the ``$Elements`` body in the layout the version dictates."""
        if self.major == 2:
            self._read_elements_v2(body)
        else:
            self._read_elements_v4(body)

    def _read_elements_v2(self, body: bytes) -> None:
        """Decode v2.2 binary elements: an ASCII count, then type-grouped ``int`` blocks.

        Each block is a three-``int`` header ``(elementType, count, numTags)`` followed by
        ``count`` records of ``int`` tag, ``numTags`` ``int`` tags and the element's node
        ``int`` tags. A face element's node tags are kept; a line or point is dropped.
        """
        newline = body.find(b"\n")
        total = _as_int(
            body[:newline].decode("latin-1").strip(), "element count", self.path
        )
        order = self.endian
        offset = newline + 1
        read = 0
        while read < total:
            (element_type, in_block, num_tags), offset = self._unpack(
                order + "3i", body, offset
            )
            node_count = _node_count(element_type, self.path)
            stride = 1 + num_tags + node_count
            flat, offset = self._unpack(
                order + ("i" * (in_block * stride)), body, offset
            )
            if element_type in _FACE_TYPES:
                for index in range(in_block):
                    base = index * stride + 1 + num_tags
                    self.face_tags.append(list(flat[base : base + node_count]))
            read += in_block

    def _read_elements_v4(self, body: bytes) -> None:
        """Decode v4.1 binary elements: entity blocks of ``size_t`` tag then node tags.

        The section header is four ``size_t`` fields; each block declares ``entityDim``,
        ``entityTag`` and ``elementType`` as ``int`` then a ``size_t`` count, followed by
        that many records of one ``size_t`` element tag and the type's node ``size_t``
        tags. Faces are kept; lines, points and volume cells are dropped.
        """
        size_t = self._size_t
        order = self.endian
        (num_blocks, _total, _min_tag, _max_tag), offset = self._unpack(
            order + (size_t * 4), body, 0
        )
        for _ in range(num_blocks):
            (_dim, _tag, element_type), offset = self._unpack(
                order + "3i", body, offset
            )
            (in_block,), offset = self._unpack(order + size_t, body, offset)
            node_count = _node_count(element_type, self.path)
            stride = 1 + node_count
            flat, offset = self._unpack(
                order + (size_t * (in_block * stride)), body, offset
            )
            if element_type in _FACE_TYPES:
                for index in range(in_block):
                    base = index * stride + 1
                    self.face_tags.append(list(flat[base : base + node_count]))
