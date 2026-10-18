"""Triangle / TetGen ``.node`` + ``.ele`` reader (task C9).

Triangle (2-D) and TetGen (3-D) share one ASCII mesh layout split across two sibling
files. The ``.node`` file lists the vertices and the ``.ele`` file lists the elements:

- ``.node`` header ``<#verts> <dim> <#attrs> <#bndmarkers>``, then one
  ``id x y [z] [attrs...] [marker]`` line per vertex.
- ``.ele`` header ``<#elts> <nodes-per-elt> <#attrs>``, then one ``id n1 n2 n3[ n4]``
  line per element.

The index base is 0 or 1 depending on the tool, and is whatever the first vertex id is,
so it is detected rather than assumed. Triangle writes ``dim=2`` with 3-node triangles;
TetGen writes ``dim=3`` with 4-node tetrahedra, whose element node lists are read as the
faces. The vertices map to ``node_x``/``node_y`` (and ``node_z`` when ``dim=3``) and the
element node lists to ``face_node_connectivity``. Pure ASCII, parsed with the stdlib and
numpy -- no declared dependency.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import warnings

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._text import iter_records

_NODE = ".node"
_ELE = ".ele"
# The spatial dimensions and element sizes the two tools write: Triangle is 2-D with
# 3-node triangles, TetGen 3-D with 4-node tetrahedra.
_DIMENSIONS = (2, 3)
_ELEMENT_SIZES = (3, 4)


class TriangleDriver(BaseDriver):
    """Reader for the Triangle / TetGen ``.node`` + ``.ele`` mesh pair.

    The ``.node`` file carries the vertices and the ``.ele`` file the elements; either
    one can be the path handed to ``read_file`` and the sibling is found beside it. The
    vertices become the node coordinates (and ``node_z`` for a 3-D mesh), and the element
    node lists become the face-node connectivity.

    Documented limitations:
        - **Both files are required.** ``.node`` and ``.ele`` are a pair sharing a stem;
          ``read_file`` is pointed at either and the other is read from beside it.
          ``can_open`` declines a lone ``.node`` or ``.ele`` with no sibling.
        - **The index base is detected, not assumed.** It is whatever the first vertex id
          is -- ``0`` or ``1`` -- resolved to 0-based and recorded for a writer. Vertices
          are read in file order, as both tools write them contiguously from the base.
        - **TetGen tetrahedra are read as 4-node faces.** The element node list is taken
          verbatim as the face connectivity; no 3-D cell topology is derived here. Faces
          are normalised to counter-clockwise winding on the ``xy`` projection.
        - **Per-vertex attributes and boundary markers are ignored**, and no CRS is
          declared -- the format carries none, so ``crs`` is ``None`` rather than guessed.
          For a 3-D mesh the third coordinate becomes a ``node_z`` variable.

    Example:
        Read a Triangle/TetGen pair by either of its files (shown, not run as a doctest --
        it needs a real ``.node``/``.ele`` pair)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("path/to/mesh.node")
            elevation = mesh.get("node_z")  # present only for a 3-D (TetGen) mesh
    """

    name = "triangle"
    long_name = "Triangle/TetGen"
    suffixes = (_NODE, _ELE)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is one half of a Triangle/TetGen pair.

        Accepts a ``.node`` or ``.ele`` file whose sibling exists beside it and whose own
        first line parses as the matching Triangle/TetGen header. Never raises -- a sniff
        error is warned and the file declined, as the base contract requires.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a ``.node``/``.ele`` file whose sibling is present and whose
                header sniffs as this format.
        """
        if self._pair(path) is None:
            return False
        try:
            return self._sniff(path)
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            warnings.warn(
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False

    @staticmethod
    def _pair(path: str) -> tuple[str, str] | None:
        """Resolve the ``(node_path, ele_path)`` pair from either file of it.

        Args:
            path: A ``.node`` or ``.ele`` file.

        Returns:
            tuple[str, str] | None: The node and element paths when both exist, else
                ``None`` for a wrong suffix, a missing file or a missing sibling.
        """
        if not os.path.isfile(path):
            return None
        lowered = path.lower()
        if lowered.endswith(_NODE):
            ele = path[: -len(_NODE)] + _ELE
            return (path, ele) if os.path.isfile(ele) else None
        if lowered.endswith(_ELE):
            node = path[: -len(_ELE)] + _NODE
            return (node, path) if os.path.isfile(node) else None
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether ``path``'s first line is a valid header for its suffix.

        A ``.node`` header is ``<#verts> <dim> ...`` with ``dim`` 2 or 3; an ``.ele``
        header is ``<#elts> <nodes-per-elt> ...`` with 3 or 4 nodes per element. Both are
        all-integer lines, which is discriminating enough to decline a foreign file.

        Args:
            path: The file whose header to inspect.

        Returns:
            bool: ``True`` when the header parses and its second field is in range.
        """
        header = self._header(path)
        if header is None or len(header) < 2:
            return False
        if path.lower().endswith(_NODE):
            return header[1] in _DIMENSIONS
        return header[1] in _ELEMENT_SIZES

    @staticmethod
    def _header(path: str) -> list[int] | None:
        """Return the first significant line as integers, or ``None`` if it is not.

        Args:
            path: The file to read the header from.

        Returns:
            list[int] | None: The header fields as integers, or ``None`` when the file
                has no significant line or its first one is not all integers.
        """
        for keyword, fields in iter_records(path):
            try:
                return [int(token) for token in (keyword, *fields)]
            except ValueError:
                return None
        return None

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Triangle/TetGen pair into a :class:`ParsedMesh`.

        Args:
            path: Either the ``.node`` or the ``.ele`` file of the pair.
            lazy: Unused -- the pair is small ASCII and read whole.

        Returns:
            ParsedMesh: The nodes, element connectivity and -- for a 3-D mesh -- the
                ``node_z`` elevation.

        Raises:
            FileFormatNotSupportedError: If the path is not a ``.node``/``.ele`` pair, or
                either file's header or body is malformed.
        """
        pair = self._pair(path)
        if pair is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not a Triangle/TetGen mesh: a .node and its sibling .ele "
                f"(or the reverse) must both be present"
            )
        node_path, ele_path = pair
        node_x, node_y, node_z, base = self._read_nodes(node_path)
        faces = self._read_elements(ele_path)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # Both files share one base; validate re-bases and records it for a writer.
            start_index=base,
            original_start_index=base,
            node_z=node_z,
            crs=None,
            global_attributes={"source": "Triangle/TetGen"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_nodes(
        self, node_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Parse the ``.node`` file into coordinates and the detected index base.

        Args:
            node_path: The ``.node`` file.

        Returns:
            tuple: ``(node_x, node_y, node_z, base)`` -- the x and y coordinates, the z
                coordinates when ``dim == 3`` else ``None``, and the vertex id base.

        Raises:
            FileFormatNotSupportedError: On a malformed header, an unsupported
                dimension, or fewer vertex lines than the header declares.
        """
        records = list(iter_records(node_path))
        count, dim = self._dimensions(records, node_path)
        rows = records[1 : 1 + count]
        if len(rows) < count:
            raise FileFormatNotSupportedError(
                f"{node_path!r} declares {count} vertices but holds {len(rows)}"
            )
        base = self._as_int(rows[0][0], node_path)
        coordinates = np.array(
            [[float(value) for value in fields[:dim]] for _, fields in rows],
            dtype=np.float64,
        )
        node_z = coordinates[:, 2] if dim == 3 else None
        return coordinates[:, 0], coordinates[:, 1], node_z, base

    def _dimensions(self, records: list, node_path: str) -> tuple[int, int]:
        """Return the vertex count and spatial dimension from the ``.node`` header.

        Raises:
            FileFormatNotSupportedError: On an empty or non-integer header, or a
                dimension that is neither 2 nor 3.
        """
        if not records:
            raise FileFormatNotSupportedError(f"{node_path!r} has no header line")
        header = self._header(node_path)
        if header is None or len(header) < 2:
            raise FileFormatNotSupportedError(
                f"{node_path!r} has no '<#verts> <dim>' header"
            )
        count, dim = header[0], header[1]
        if dim not in _DIMENSIONS:
            raise FileFormatNotSupportedError(
                f"{node_path!r} declares dimension {dim}; only 2 or 3 is supported"
            )
        return count, dim

    def _read_elements(self, ele_path: str) -> np.ndarray:
        """Parse the ``.ele`` file into the element node connectivity.

        Args:
            ele_path: The ``.ele`` file.

        Returns:
            np.ndarray: The element node lists, shape ``(n_elts, nodes_per_elt)``, with
                ids still in the file's own base.

        Raises:
            FileFormatNotSupportedError: On a malformed header or fewer element lines
                than the header declares.
        """
        records = list(iter_records(ele_path))
        header = self._header(ele_path)
        if header is None or len(header) < 2:
            raise FileFormatNotSupportedError(
                f"{ele_path!r} has no '<#elts> <nodes-per-elt>' header"
            )
        count, per_element = header[0], header[1]
        rows = records[1 : 1 + count]
        if len(rows) < count:
            raise FileFormatNotSupportedError(
                f"{ele_path!r} declares {count} elements but holds {len(rows)}"
            )
        return np.array(
            [[int(node) for node in fields[:per_element]] for _, fields in rows],
            dtype=np.int64,
        )

    @staticmethod
    def _as_int(token: str, path: str) -> int:
        """Return ``token`` as an int, or raise a format error naming ``path``."""
        try:
            return int(token)
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{path!r} has a non-integer vertex id {token!r}"
            ) from error
