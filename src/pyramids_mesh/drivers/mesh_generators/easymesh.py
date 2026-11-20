"""SWAN Easymesh ``.n`` + ``.e`` reader (task C32).

SWAN defines no native unstructured grid of its own: ``READGRID UNSTRUCTURED``
delegates to one of three external mesh generators -- ADCIRC (``fort.14``), Triangle
(``.node``/``.ele``) or Easymesh (``.n``/``.e``). The first two are read by
:mod:`pyramids_mesh.drivers.coastal_ocean.adcirc` and :mod:`pyramids_mesh.drivers.mesh_generators.triangle`; this
driver adds the third, Easymesh, so all three SWAN unstructured routes are covered.

Easymesh (Bojan Niceno) writes a triangulation across three sibling ASCII files that
share a stem -- ``.n`` (nodes), ``.e`` (elements) and ``.s`` (sides). Only ``.n`` and
``.e`` carry the mesh SWAN reads, and only those two are used here:

- ``.n``: a node count on the first line, then one ``id:  x y marker`` line per node.
  Ids are 0-based and colon-suffixed; coordinates are written in scientific notation.
- ``.e``: an element count on the first line, then one
  ``id:  i j k  ei ej ek  si sj sk  xV yV  sign`` line per triangle. Only the first
  three fields -- the triangle's node ids ``i j k`` -- are read; the neighbour-element
  ids, side ids, circumcentre (Voronoi node) and sign are the mesh topology Easymesh
  derived and are reconstructed upstream in pyramids, so they are ignored here.

Easymesh closes each file with two trailer lines -- a rule of dashes and a column
legend -- which are neither blank nor ``#``-comments. They are never read because both
files are sliced to the count their header declares, exactly as the Triangle reader
does. Pure ASCII, parsed with the standard library and numpy -- no declared dependency.
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

_NODE = ".n"
_ELEMENT = ".e"
# The triangle node ids are the first three fields of an element record; everything
# after them (neighbours, sides, circumcentre, sign) is derived data pyramids rebuilds.
_NODES_PER_FACE = 3


class EasymeshDriver(BaseDriver):
    """Reader for the SWAN Easymesh ``.n`` + ``.e`` triangular-mesh pair.

    The ``.n`` file carries the node coordinates and the ``.e`` file the triangles;
    either one can be the path handed to ``read_file`` and the sibling is found beside
    it. The nodes become the node coordinates and each element's first three ids become
    a triangular face in the ``face_node_connectivity``.

    Documented limitations:
        - **Both files are required.** ``.n`` and ``.e`` are a pair sharing a stem;
          ``read_file`` is pointed at either and the other is read from beside it.
          ``can_open`` declines a lone ``.n`` or ``.e`` with no sibling.
        - **Only ``.n`` and ``.e`` are read.** The ``.s`` sides file Easymesh also writes
          is redundant -- the sides, element neighbours and circumcentres are topology
          pyramids derives from the nodes and faces -- so it is neither required nor read.
        - **Triangles only.** Easymesh produces Delaunay triangulations; each element
          record's first three ids are read as a triangular face. Faces are normalised to
          counter-clockwise winding on the ``xy`` projection.
        - **The index base is read from the first node id** -- Easymesh is 0-based
          (C-style), but the base is taken from the first node rather than presumed (the
          remaining ids are assumed contiguous from it, see below), resolved to 0-based
          and recorded for a writer.
        - **Node markers, element neighbours, side ids, the circumcentre and the sign are
          ignored**, and no CRS is declared -- the format carries none, so ``crs`` is
          ``None`` rather than guessed.
        - **The ``.e`` suffix is shared with the Exodus driver.** Dispatch resolves by
          content, not registration order: an Exodus ``.e`` is binary NetCDF (whose sniff
          declines an ASCII file) and this driver requires a sibling ``.n``, which an
          Exodus ``.e`` never has -- so each real file resolves to exactly one driver.
        - **Node ids are assumed dense and contiguous from the base**, read in file order
          with the ``.e`` references sharing the ``.n`` base; a gapped or renumbered ``.n``
          is out of scope (Easymesh always writes a dense 0-based block). An ``.e`` that
          references a node id outside the ``.n`` range is left for pyramids' validation to
          reject, not range-checked here.

    Example:
        Read an Easymesh pair by either of its files (shown, not run as a doctest -- it
        needs a real ``.n``/``.e`` pair)::

            from pyramids.netcdf.ugrid import MeshDataset

            mesh = MeshDataset.read_file("path/to/mesh.n")
            mesh = MeshDataset.read_file("path/to/mesh.e")
    """

    name = "easymesh"
    long_name = "SWAN Easymesh"
    suffixes = (_NODE, _ELEMENT)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is one half of an Easymesh pair.

        Accepts a ``.n`` or ``.e`` file whose sibling exists beside it and whose own
        first lines sniff as the Easymesh layout -- a lone-integer count followed by a
        colon-suffixed id record. Never raises -- a sniff error is warned and the file
        declined, as the base contract requires.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a ``.n``/``.e`` file whose sibling is present and whose
                first records sniff as this format.
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
        """Resolve the ``(node_path, element_path)`` pair from either file of it.

        Args:
            path: A ``.n`` or ``.e`` file.

        Returns:
            tuple[str, str] | None: The node and element paths when both exist, else
                ``None`` for a wrong suffix, a missing file or a missing sibling.
        """
        if not os.path.isfile(path):
            return None
        lowered = path.lower()
        if lowered.endswith(_NODE):
            element = path[: -len(_NODE)] + _ELEMENT
            return (path, element) if os.path.isfile(element) else None
        if lowered.endswith(_ELEMENT):
            node = path[: -len(_ELEMENT)] + _NODE
            return (node, path) if os.path.isfile(node) else None
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether ``path``'s first records look like an Easymesh file.

        An Easymesh ``.n``/``.e`` file opens with a bare integer count on its own line
        -- no other fields -- followed by records whose id token is colon-suffixed
        (``0:``). That pair of traits discriminates it from the Triangle ``.node``
        header (which carries ``dim``/``attr`` fields beside the count) and from foreign
        files without colon-tagged ids.

        Args:
            path: The file whose opening records to inspect.

        Returns:
            bool: ``True`` when the first record is a lone integer and the next record's
                id is colon-suffixed.
        """
        records = self._head(path, limit=2)
        if len(records) < 2:
            return False
        (first_keyword, first_fields), (second_keyword, _) = records
        if first_fields:
            return False
        try:
            int(first_keyword)
        except ValueError:
            return False
        return second_keyword.endswith(":")

    @staticmethod
    def _head(path: str, *, limit: int) -> list[tuple[str, list[str]]]:
        """Return up to ``limit`` significant records from the start of ``path``.

        Args:
            path: The file to read.
            limit: The most records to return.

        Returns:
            list[tuple[str, list[str]]]: The first ``limit`` ``(keyword, fields)``
                records, fewer if the file is shorter.
        """
        head: list[tuple[str, list[str]]] = []
        for record in iter_records(path):
            head.append(record)
            if len(head) >= limit:
                break
        return head

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an Easymesh ``.n``/``.e`` pair into a :class:`ParsedMesh`.

        Args:
            path: Either the ``.n`` or the ``.e`` file of the pair.
            lazy: Unused -- the pair is small ASCII and read whole.

        Returns:
            ParsedMesh: The nodes and triangular face-node connectivity.

        Raises:
            FileFormatNotSupportedError: If the path is not a ``.n``/``.e`` pair, or
                either file's header or body is malformed.
        """
        pair = self._pair(path)
        if pair is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an Easymesh mesh: a .n and its sibling .e "
                f"(or the reverse) must both be present"
            )
        node_path, element_path = pair
        node_x, node_y, base = self._read_nodes(node_path)
        faces = self._read_elements(element_path)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # Both files share one base; validate re-bases and records it for a writer.
            start_index=base,
            original_start_index=base,
            crs=None,
            global_attributes={"source": "SWAN Easymesh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_nodes(self, node_path: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Parse the ``.n`` file into coordinates and the detected index base.

        Args:
            node_path: The ``.n`` file.

        Returns:
            tuple: ``(node_x, node_y, base)`` -- the x and y coordinates and the node id
                base (``0`` for a standard Easymesh file).

        Raises:
            FileFormatNotSupportedError: On a missing or non-integer count, fewer node
                lines than the count declares, or a non-numeric coordinate.
        """
        records = list(iter_records(node_path))
        count = self._count(records, node_path, noun="nodes")
        rows = records[1 : 1 + count]
        if len(rows) < count:
            raise FileFormatNotSupportedError(
                f"{node_path!r} declares {count} nodes but holds {len(rows)}"
            )
        base = self._node_id(rows[0][0], node_path)
        try:
            coordinates = np.array(
                [[float(fields[0]), float(fields[1])] for _, fields in rows],
                dtype=np.float64,
            )
        except (IndexError, ValueError) as error:
            raise FileFormatNotSupportedError(
                f"{node_path!r} has a node line without two numeric coordinates: {error}"
            ) from error
        return coordinates[:, 0], coordinates[:, 1], base

    def _read_elements(self, element_path: str) -> np.ndarray:
        """Parse the ``.e`` file into the triangular face-node connectivity.

        Args:
            element_path: The ``.e`` file.

        Returns:
            np.ndarray: The triangle node ids, shape ``(n_face, 3)``, still in the
                file's own base.

        Raises:
            FileFormatNotSupportedError: On a missing or non-integer count, fewer
                element lines than the count declares, or an element line without three
                integer node ids.
        """
        records = list(iter_records(element_path))
        count = self._count(records, element_path, noun="elements")
        rows = records[1 : 1 + count]
        if len(rows) < count:
            raise FileFormatNotSupportedError(
                f"{element_path!r} declares {count} elements but holds {len(rows)}"
            )
        # A short row would otherwise yield a ``(n, 2)`` face table rather than being
        # declined; every element record must carry at least three node ids.
        if any(len(fields) < _NODES_PER_FACE for _, fields in rows):
            raise FileFormatNotSupportedError(
                f"{element_path!r} has an element line with fewer than three node ids"
            )
        try:
            return np.array(
                [
                    [int(field) for field in fields[:_NODES_PER_FACE]]
                    for _, fields in rows
                ],
                dtype=np.int64,
            )
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{element_path!r} has an element line without three integer node ids: "
                f"{error}"
            ) from error

    @staticmethod
    def _count(records: list, path: str, *, noun: str) -> int:
        """Return the leading count an Easymesh file declares on its first line.

        Args:
            records: The file's significant records, in order.
            path: The file path, for error messages.
            noun: What the count counts (``"nodes"`` or ``"elements"``), for messages.

        Returns:
            int: The declared count, guaranteed ``>= 1``.

        Raises:
            FileFormatNotSupportedError: When the file is empty, its first line is not a
                bare integer count, or that count is not positive (a zero or negative
                count is degenerate -- Easymesh never writes one -- and left unchecked a
                zero count would dereference an empty row list).
        """
        if not records:
            raise FileFormatNotSupportedError(
                f"{path!r} has no header line declaring the number of {noun}"
            )
        keyword, fields = records[0]
        if fields:
            raise FileFormatNotSupportedError(
                f"{path!r} header {keyword!r} is not a bare {noun} count"
            )
        try:
            count = int(keyword)
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{path!r} has a non-integer {noun} count {keyword!r}"
            ) from error
        if count < 1:
            raise FileFormatNotSupportedError(
                f"{path!r} declares a non-positive {noun} count {count}"
            )
        return count

    @staticmethod
    def _node_id(token: str, path: str) -> int:
        """Return a node id written ``"0:"`` as the int ``0``, naming ``path`` on error.

        Args:
            token: The colon-suffixed id token from a node record (e.g. ``"0:"``).
            path: The file path, for the error message.

        Returns:
            int: The id with its trailing colon removed.

        Raises:
            FileFormatNotSupportedError: When the token is not an integer id.
        """
        try:
            return int(token.rstrip(":"))
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{path!r} has a non-integer node id {token!r}"
            ) from error
