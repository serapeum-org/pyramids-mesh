"""Gmsh ``.msh`` reader (task C3).

Reads the Gmsh ASCII mesh format into ``pyramids.netcdf.ugrid`` objects. Both the
legacy **v2.2** layout and the current **v4.1** entity-block layout are supported; a
binary file (``file-type`` ``1`` in the ``$MeshFormat`` header) is refused rather than
mis-read, since nothing here decodes the packed records.

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

from typing import Iterator

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

# ASCII is file-type 0; 1 is the binary stream this reader does not decode.
_ASCII_FILE_TYPE = 0

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


class GmshDriver(BaseDriver):
    """Reader for the Gmsh ``.msh`` ASCII mesh format (v2.2 and v4.1).

    Reads node coordinates and the surface connectivity, mapping triangle (type ``2``)
    and quadrilateral (type ``3``) elements to ``face_node_connectivity``; a mixed file
    pads its triangles to the quad width with ``-1``. Node elevation is carried as a
    ``node_z`` variable only when some node has a non-zero z, since a planar mesh gains
    nothing from an all-zero column.

    Documented limitations:
        - **Binary ``.msh`` is refused.** The ``$MeshFormat`` header's ``file-type`` must
          be ``0`` (ASCII); a ``1`` (binary) file is declined rather than mis-read.
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
        major version in :data:`_SUPPORTED_MAJORS`. Never raises -- a binary or empty
        file, or an unparsable version, answers ``False`` -- so ``can_open`` stays within
        its never-raise contract and dispatch is never aborted by a broken candidate.

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

        Args:
            path: The ``.msh`` file to read.
            lazy: Ignored -- a Gmsh ASCII file is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, the triangular/quadrilateral face connectivity and, when
                present, a ``node_z`` elevation variable.

        Raises:
            FileFormatNotSupportedError: If the header is missing or binary, the version
                is unsupported, an element type cannot be sized, a record is short or
                non-numeric, or the file declares no faces.
            ValueError: If two nodes share a tag, which makes the tag->index map ambiguous.
        """
        return _GmshReader(path).build()


class _GmshReader:
    """One Gmsh file: stream its significant lines and assemble the mesh."""

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
            FileFormatNotSupportedError: On a missing or binary header, an unsupported
                version, an unaccounted element type, a malformed record, or a file with
                no triangular or quadrilateral face.
            ValueError: If two nodes share a tag.
        """
        major = self._read_header()
        for line in self._lines:
            if line[0] == _NODES:
                self._read_nodes(major)
            elif line[0] == _ELEMENTS:
                self._read_elements(major)
        return self._assemble()

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
        """Read ``$MeshFormat`` and return the major version, refusing a binary file.

        Returns:
            int: The major version, ``2`` or ``4``.

        Raises:
            FileFormatNotSupportedError: If the header is missing, the version is
                unsupported, or the file declares itself binary.
        """
        first = self._next("$MeshFormat header")
        if first[0] != _HEADER:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a Gmsh file: expected a {_HEADER} header, "
                f"found {first[0]!r}"
            )
        fields = self._next("version line")
        major = int(_as_float(fields[0], "Gmsh version", self.path))
        if major not in _SUPPORTED_MAJORS:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares Gmsh version {fields[0]!r}; only the 2.2 and "
                f"4.1 ASCII layouts are read"
            )
        file_type = _as_int(fields[1], "file-type", self.path) if len(fields) > 1 else 0
        if file_type != _ASCII_FILE_TYPE:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a binary Gmsh file (file-type {file_type}); only the "
                f"ASCII layout is read"
            )
        return major

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
            node_count = self._node_count(element_type)
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
            node_count = self._node_count(element_type)
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

    def _node_count(self, element_type: int) -> int:
        """Return the node count for an element type, refusing an unaccounted one.

        Args:
            element_type: The Gmsh element type code.

        Returns:
            int: The number of node tags an element of this type carries.

        Raises:
            FileFormatNotSupportedError: If the type has no known node count, since the
                record then cannot be sized and everything after it would misalign.
        """
        if element_type not in _ELEMENT_NODE_COUNTS:
            raise FileFormatNotSupportedError(
                f"{self.path!r} uses element type {element_type}, which this reader does "
                f"not account for; such a file is refused rather than misread"
            )
        return _ELEMENT_NODE_COUNTS[element_type]

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

    def _assemble(self) -> ParsedMesh:
        """Resolve node tags to indices and build the parsed surface mesh.

        Returns:
            ParsedMesh: The assembled mesh.

        Raises:
            FileFormatNotSupportedError: If no triangular or quadrilateral face was read.
            ValueError: If two nodes share a tag.
        """
        if not self.face_tags:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no triangular or quadrilateral faces, so it has "
                f"no surface to build"
            )
        tag_to_index = native_id_map(self.node_tags)
        faces = self._resolve_faces(tag_to_index)
        node_z = np.array(self.node_z, dtype=np.float64)
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the tag map, so the
            # validator has nothing to subtract. Gmsh's conventional 1-based tags are
            # recorded for a writer.
            start_index=0,
            original_start_index=1,
            node_z=node_z if np.any(node_z != 0.0) else None,
            crs=None,
            global_attributes={"source": "Gmsh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _resolve_faces(self, tag_to_index: dict[int, int]) -> np.ndarray:
        """Map face node tags to 0-based indices, padding short rows with ``-1``.

        Args:
            tag_to_index: The node-tag to 0-based-index map.

        Returns:
            np.ndarray: The ``(n_face, max_nodes)`` connectivity, fill-padded.

        Raises:
            ValueError: If a face references a node tag no node record defined.
        """
        max_nodes = max(len(row) for row in self.face_tags)
        faces = np.full((len(self.face_tags), max_nodes), -1, dtype=np.int64)
        for row, tags in enumerate(self.face_tags):
            for column, tag in enumerate(tags):
                if tag not in tag_to_index:
                    raise ValueError(
                        f"{self.path!r} has an element referencing node tag {tag}, which "
                        f"no node record defines"
                    )
                faces[row, column] = tag_to_index[tag]
        return faces
