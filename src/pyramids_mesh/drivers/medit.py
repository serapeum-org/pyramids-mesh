"""Medit (INRIA) ``.mesh`` ASCII reader.

Reads the Medit mesh format: a plain-text stream of keyword blocks, each a keyword
followed by a count and then that many rows. The file opens with
``MeshVersionFormatted <version>`` and a ``Dimension`` of ``2`` or ``3``, then carries
``Vertices`` and any mix of element sections -- ``Edges``, ``Triangles``,
``Quadrilaterals``, ``Tetrahedra``, ``Hexahedra`` -- and an optional ``End``. The whole
file is parsed with the standard library alone, so the driver declares no dependency and
imports no mesh library.

Only the 2D surface sections (``Triangles`` and ``Quadrilaterals``) become a mesh, the
way the VTK driver keeps only a grid's surface cells. The volume sections
(``Tetrahedra``, ``Hexahedra``) carry a vertical structure this package does not model
yet (ADR-0005, design section 21.1); they are dropped rather than boundary-extracted,
which is pyramids' job, not a driver's. A file with no surface section at all is refused
rather than returned as an empty mesh.

The ``.mesh`` suffix is shared with the DHI Mike21 frame format, so ``can_open`` requires
the ``MeshVersionFormatted`` signature on the first non-blank line and declines a Mike21
``.mesh`` (which opens with a numeric EUM header instead).
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The ASCII signature. It is the first keyword of every Medit file and is what the sniff
# keys on, so a Mike21 ``.mesh`` -- which opens with a numeric EUM header -- is declined.
_SIGNATURE = "MeshVersionFormatted"

# ``can_open`` must not read the whole file, so the sniff decodes only this many bytes --
# enough for the first line, which is all it judges.
_SNIFF_LIMIT = 4096

# Surface element sections, mapped to the corner-node count a row carries before its
# trailing reference tag. These become faces.
_SURFACE_NODES = {"Triangles": 3, "Quadrilaterals": 4}

# Volume element sections, mapped to the node count a row carries before its reference
# tag. These are parsed past to stay aligned, then dropped.
_VOLUME_NODES = {"Tetrahedra": 4, "Hexahedra": 8}

# An ``Edges`` row is ``n1 n2 ref``: two boundary-segment nodes. It is parsed past to stay
# aligned but not kept -- a Medit edge is a 1D boundary marker, not a surface-mesh edge.
_EDGE_NODES = 2

# Medit node ids are 1-based and contiguous, so ``0`` is never a real node and is used to
# pad a triangle row out to a quad's width. The validator masks it before re-basing, so
# the pad becomes ``-1`` rather than being shifted to a negative index.
_FILL = 0

_VERTEX_KEYWORD = "Vertices"
_DIMENSION_KEYWORD = "Dimension"
_END_KEYWORD = "End"
_EDGE_KEYWORD = "Edges"

# The two dimensions a vertex record is read for: ``x y ref`` or ``x y z ref``.
_SUPPORTED_DIMENSIONS = (2, 3)


class MeditDriver(BaseDriver):
    """Reader for the Medit (INRIA) ``.mesh`` ASCII mesh format.

    A keyword-block text file: ``MeshVersionFormatted``, a ``Dimension`` of ``2`` or
    ``3``, ``Vertices`` and the element sections. Node coordinates come from the
    ``Vertices`` block (its trailing reference tag dropped); the surface sections
    ``Triangles`` and ``Quadrilaterals`` become the fill-padded face-node connectivity,
    and the volume sections are dropped. The format carries a frame and no results, so
    the driver declares ``read_mesh`` only.

    For a 3D file the vertex z column becomes a ``node_z`` variable, as it does for 2DM
    and Mike21: the pyramids mesh model is planar and carries no per-node elevation
    coordinate (design section 6). A 2D file has no z column, so it gets no ``node_z``.

    Documented limitations:
        - **Only the 2D surface sections become a mesh** -- ``Triangles`` (3-node) and
          ``Quadrilaterals`` (4-node). ``Tetrahedra`` and ``Hexahedra`` are volume cells
          that carry a vertical structure this package does not model yet; they are
          dropped, not boundary-extracted (that is pyramids' job). A file with no surface
          section at all is refused rather than returned as an empty mesh.
        - **``Edges`` are read past but not kept.** A Medit edge is a 1D boundary marker,
          not an edge of the surface mesh, so feeding it to the mesh's edge connectivity
          would mislabel it.
        - **A section keyword this reader does not know is refused**, rather than skipped
          by guessing its row arity. ``MeshVersionFormatted``, ``Dimension``,
          ``Vertices``, ``Edges``, ``Triangles``, ``Quadrilaterals``, ``Tetrahedra``,
          ``Hexahedra`` and ``End`` are the keywords read; a ``Corners`` or ``Normals``
          section, say, is declined rather than mis-parsed.
        - **Only the ASCII format is read.** The binary ``.meshb`` variant is not
          handled and is not claimed by the suffix.
        - **Node ids are 1-based and read as the file numbers them**, contiguously from
          ``1``; the 1-based numbering is recorded for a writer to restore.
        - There is no CRS in the format, so ``crs`` is ``None``; it is never guessed from
          the coordinate magnitudes.

    Examples:
        - A small 2D Medit mesh of two triangles reads into a surface mesh. The file
          numbers its nodes from ``1``; the connectivity arrives 0-based with the base
          recorded for a writer:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.medit import MeditDriver
            >>> text = '''MeshVersionFormatted 1
            ... Dimension 2
            ... Vertices
            ... 4
            ... 0 0 1
            ... 1 0 1
            ... 1 1 1
            ... 0 1 1
            ... Triangles
            ... 2
            ... 1 2 3 0
            ... 1 3 4 0
            ... End
            ... '''
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "two.mesh")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write(text)
            ...     dataset = MeditDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1

        - The suffix is shared with Mike21, so the sniff requires the Medit signature. A
          Mike21 ``.mesh`` -- which opens with a numeric header -- is declined:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.medit import MeditDriver
            >>> mike = '''100079 1000 4 LONG/LAT
            ... 1 0 0 0 1
            ... '''
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "mike.mesh")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         _ = handle.write(mike)
            ...     MeditDriver().can_open(path)
            False
    """

    name = "medit"
    long_name = "Medit (INRIA) mesh"
    suffixes = (".mesh",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the first non-blank line carries the Medit signature.

        Reads a bounded prefix and decodes it leniently, so a binary or empty file
        answers ``False`` rather than raising. The ``.mesh`` suffix is shared with
        Mike21, whose header is a numeric EUM line, so keying on ``MeshVersionFormatted``
        is what tells the two apart and keeps this driver from claiming a Mike21 file.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first non-blank line starts with
                ``MeshVersionFormatted``.
        """
        with open(path, "rb") as handle:
            head = handle.read(_SNIFF_LIMIT)
        for line in head.decode("utf-8-sig", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            return stripped.startswith(_SIGNATURE)
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Medit ``.mesh`` file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Args:
            path: The ``.mesh`` file to read.
            lazy: Ignored -- the format is plain text and read eagerly.

        Returns:
            ParsedMesh: The nodes, the surface faces, and -- for a 3D file -- the
                ``node_z`` elevation variable.

        Raises:
            FileFormatNotSupportedError: If the file declares no ``Dimension`` or an
                unsupported one, declares no ``Vertices``, carries no surface section to
                build a mesh from, uses a section keyword this reader does not know, ends
                before a count's rows, or has a non-numeric field.
        """
        return _MeditReader(path).build()


class _MeditReader:
    """One Medit file's token stream, walked keyword by keyword into a parsed mesh.

    Medit lets a keyword and its count fall on separate lines, so the file is flattened
    to a single token list and walked with a cursor rather than line by line.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.tokens = _tokenise(path)
        self.index = 0
        self.dimension: int | None = None
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        # Surface faces as 1-based node-id lists, triangles and quads together, in the
        # order their sections appear; padded to a rectangle at the end.
        self.faces: list[list[int]] = []
        self.saw_vertices = False

    def build(self) -> ParsedMesh:
        """Walk the token stream and assemble the parsed mesh.

        Returns:
            ParsedMesh: The nodes, surface faces and -- for a 3D file -- ``node_z``.

        Raises:
            FileFormatNotSupportedError: On a missing or unsupported ``Dimension``, a
                missing ``Vertices`` block, a file with no surface section, an unknown
                section keyword, a truncated count, or a non-numeric field.
        """
        self._walk()
        if self.dimension is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no {_DIMENSION_KEYWORD}; a Medit file names its "
                f"dimension before its vertices"
            )
        if not self.saw_vertices:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares no {_VERTEX_KEYWORD} block"
            )
        if not self.faces:
            raise FileFormatNotSupportedError(
                f"{self.path!r} holds no 2D surface section "
                f"({'/'.join(_SURFACE_NODES)}); a volume-only Medit mesh is refused "
                f"rather than built as an empty mesh"
            )
        faces = self._build_faces()
        node_z = (
            np.array(self.node_z, dtype=np.float64) if self.dimension == 3 else None
        )
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            # 1-based ids padded with 0: the validator masks the pad before re-basing,
            # so a short triangle row in a quad-width array becomes -1, not -1 shifted.
            fill_value=_FILL,
            start_index=1,
            original_start_index=1,
            node_z=node_z,
            crs=None,
            variables={},
            global_attributes={"source": "Medit mesh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _walk(self) -> None:
        """Dispatch each top-level section keyword until the stream is exhausted."""
        while not self._done():
            keyword = self._take("a section keyword")
            self._dispatch(keyword)

    def _dispatch(self, keyword: str) -> None:
        """Route one section keyword to its reader.

        Args:
            keyword: The keyword token just consumed.

        Raises:
            FileFormatNotSupportedError: If the keyword is one this reader does not know;
                guessing its row arity to skip it could desynchronise the stream.
        """
        if keyword == _SIGNATURE:
            self._take("the format version")  # the version value, not needed
        elif keyword == _DIMENSION_KEYWORD:
            self._read_dimension()
        elif keyword == _VERTEX_KEYWORD:
            self._read_vertices()
        elif keyword in _SURFACE_NODES:
            self._read_surface(keyword, _SURFACE_NODES[keyword])
        elif keyword in _VOLUME_NODES:
            self._skip_cells(keyword, _VOLUME_NODES[keyword])
        elif keyword == _EDGE_KEYWORD:
            self._skip_cells(_EDGE_KEYWORD, _EDGE_NODES)
        elif keyword == _END_KEYWORD:
            # End terminates the file; anything after it is ignored.
            self.index = len(self.tokens)
        else:
            raise FileFormatNotSupportedError(
                f"{self.path!r} uses the {keyword!r} Medit section, which this reader "
                f"does not read; its row count cannot be guessed without mis-parsing the "
                f"rest of the file"
            )

    def _read_dimension(self) -> None:
        """Read the ``Dimension`` value, refusing anything but 2 or 3.

        Raises:
            FileFormatNotSupportedError: On a non-integer or unsupported dimension.
        """
        value = self._take_int("the Dimension value")
        if value not in _SUPPORTED_DIMENSIONS:
            raise FileFormatNotSupportedError(
                f"{self.path!r} declares Dimension {value}; only "
                f"{' and '.join(str(d) for d in _SUPPORTED_DIMENSIONS)} are read"
            )
        self.dimension = value

    def _read_vertices(self) -> None:
        """Read the ``Vertices`` count and each ``x y [z] ref`` row.

        The trailing reference tag is read for width but dropped -- it is a Medit label,
        not a coordinate. For a 3D file the z column is kept and becomes ``node_z``.

        Raises:
            FileFormatNotSupportedError: If ``Vertices`` appears before ``Dimension`` (so
                the row width is unknown), the count is truncated, or a field is not a
                number.
        """
        if self.dimension is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a {_VERTEX_KEYWORD} block before its "
                f"{_DIMENSION_KEYWORD}; the vertex row width is unknown without it"
            )
        self.saw_vertices = True
        count = self._take_int(f"the {_VERTEX_KEYWORD} count")
        for _ in range(count):
            self.node_x.append(self._take_float("a vertex x"))
            self.node_y.append(self._take_float("a vertex y"))
            if self.dimension == 3:
                self.node_z.append(self._take_float("a vertex z"))
            self._take("a vertex reference tag")  # the ref tag, dropped

    def _read_surface(self, keyword: str, corners: int) -> None:
        """Read a surface section's count and collect each row's corner node ids.

        Args:
            keyword: The section keyword, for error messages.
            corners: The corner-node count a row carries before its reference tag.

        Raises:
            FileFormatNotSupportedError: On a truncated count or a non-integer node id.
        """
        count = self._take_int(f"the {keyword} count")
        for _ in range(count):
            row = [self._take_int(f"a {keyword} node id") for _ in range(corners)]
            self._take(f"a {keyword} reference tag")  # the ref tag, dropped
            self.faces.append(row)

    def _skip_cells(self, keyword: str, nodes: int) -> None:
        """Read past a section whose rows are not kept, to stay aligned.

        The volume sections (``Tetrahedra``, ``Hexahedra``) and the ``Edges`` section are
        parsed but dropped; each row is ``nodes`` ids plus a reference tag.

        Args:
            keyword: The section keyword, for error messages.
            nodes: The node count a row carries before its reference tag.

        Raises:
            FileFormatNotSupportedError: On a truncated count.
        """
        count = self._take_int(f"the {keyword} count")
        for _ in range(count):
            for _ in range(nodes + 1):  # the node ids plus the reference tag
                self._take(f"a {keyword} field")

    def _build_faces(self) -> np.ndarray:
        """Pack the collected faces into a rectangular, fill-padded array.

        Triangle rows in a mesh that also holds quads are padded to the widest face with
        ``0``; the validator rewrites that to ``-1`` as it re-bases the 1-based ids.

        Returns:
            np.ndarray: The ``(n_face, max_corners)`` connectivity, ``0``-padded.
        """
        width = max(len(row) for row in self.faces)
        faces = np.full((len(self.faces), width), _FILL, dtype=np.int64)
        for row_index, row in enumerate(self.faces):
            faces[row_index, : len(row)] = row
        return faces

    def _done(self) -> bool:
        """Whether every token has been consumed."""
        return self.index >= len(self.tokens)

    def _take(self, what: str) -> str:
        """Consume and return the next token, refusing a file that ends too soon.

        Args:
            what: A short description of the expected token, for the error message.

        Returns:
            str: The next token.

        Raises:
            FileFormatNotSupportedError: If no token remains.
        """
        if self._done():
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends before {what}: a section count promises more rows "
                f"than the file holds"
            )
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _take_int(self, what: str) -> int:
        """Consume the next token as an integer, naming it if it is not one.

        Raises:
            FileFormatNotSupportedError: If the token is not an integer.
        """
        token = self._take(what)
        try:
            return int(token)
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} expected an integer for {what}, found {token!r}"
            ) from error

    def _take_float(self, what: str) -> float:
        """Consume the next token as a float, naming it if it is not one.

        Raises:
            FileFormatNotSupportedError: If the token is not a number.
        """
        token = self._take(what)
        try:
            return float(token)
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} expected a number for {what}, found {token!r}"
            ) from error


def _tokenise(path: str) -> list[str]:
    """Return the file's whitespace-separated tokens, with comments stripped.

    A Medit comment runs from ``#`` to the end of the line, so each line is cut at its
    first ``#`` before being split. The file is decoded as ``utf-8-sig`` so a leading
    byte-order mark -- which a Windows editor writes routinely -- is dropped rather than
    glued onto the first keyword, and undecodable bytes are replaced rather than raising.

    Args:
        path: The ``.mesh`` file to tokenise.

    Returns:
        list[str]: Every significant token, in file order.
    """
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            code = line.split("#", 1)[0]
            tokens.extend(code.split())
    return tokens
