"""Esri TIN reader (task D5).

An Esri TIN is not a file but a **directory** of big-endian binary ``.adf`` components:
``tnxy.adf`` (node XY), ``tnz.adf`` (node elevation), ``tnod.adf`` (1-based triangle
triples), ``tmsx.adf``/``tmsk.adf`` (the mask that marks the convex-hull frame triangles
to drop) and ``prj.adf`` (the projection WKT). ``tdenv9.adf`` (or the older ``tdenv.adf``)
holds the node and triangle counts.

A directory has no file suffix, so ``read_file`` cannot dispatch a bare directory path.
The driver instead claims ``.adf`` and is reached two ways: ``read_file`` pointed at any
of the TIN's ``.adf`` component files (``read_file("tin_dir/tdenv9.adf")``), or the
directory itself passed with ``driver="esri_tin"``. Either way ``can_open`` confirms the
TIN components are present and the reader works from the directory. Nothing here reads a
raster ``.adf`` (an Esri GRID), which this package does not own -- ``can_open`` declines an
``.adf`` whose directory lacks the TIN triangle table.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import warnings

import numpy as np
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh._crs import crs_from_esri_wkt

# The components that make a directory an Esri TIN; the mask and projection are optional.
_REQUIRED = ("tnxy.adf", "tnz.adf", "tnod.adf")
_TDENV = ("tdenv9.adf", "tdenv.adf")  # v9+ first, then the older name


def _drop_unreferenced_nodes(
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    faces: np.ndarray,
    fill_value: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the nodes referenced by a face and reindex the faces to match.

    Dropping the masked hull-frame triangles leaves the nodes they alone referenced in the
    node table, unreferenced, which inflates the mesh extent and produces empty-looking
    plots (issue #79). Removing those orphan nodes and remapping the connectivity makes the
    mesh exactly the surface the TIN represents.

    Args:
        node_x: Node x coordinates.
        node_y: Node y coordinates.
        node_z: Node elevations.
        faces: Face-node connectivity (0-based), ``fill_value`` marking absent corners.
        fill_value: The sentinel for an absent corner; preserved in the output.

    Returns:
        tuple: The compacted ``(node_x, node_y, node_z, faces)`` with no orphan nodes.
    """
    used = np.unique(faces[faces != fill_value])
    remap = np.full(node_x.shape[0], fill_value, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    reindexed = np.where(faces != fill_value, remap[faces.clip(min=0)], fill_value)
    return node_x[used], node_y[used], node_z[used], reindexed


class EsriTinDriver(BaseDriver):
    """Reader for the Esri TIN directory format.

    An Esri TIN is a directory of big-endian ``.adf`` files. The driver reads the node
    coordinates and elevations, the 1-based triangle table, and the mask that marks the
    convex-hull frame triangles the TIN carries but does not treat as surface -- it keeps
    only the unmasked triangles, exactly as the stored kept-count records.

    Documented limitations:
        - **The input is a directory of ``.adf`` files.** ``read_file`` cannot dispatch a
          bare directory (it has no suffix), so point it at any of the TIN's ``.adf``
          files, or pass the directory with ``driver="esri_tin"``. ``can_open`` accepts
          either and confirms the TIN components are present; a raster ``.adf`` (an Esri
          GRID), whose directory has no triangle table, is declined.
        - **Masked triangles are dropped.** The hull-frame triangles flagged in
          ``tmsk.adf`` are excluded, so the mesh is the surface the TIN represents. The
          frame nodes they alone referenced are then dropped from the node table and the
          connectivity reindexed, so no orphan nodes inflate the extent (issue #79).
        - **Triangle indices are 1-based.** They are resolved to 0-based and the base is
          recorded for a writer. Every face is a triangle.
        - **The projection is read from ``prj.adf`` only when it validates.** A missing or
          unparseable ``prj.adf`` leaves ``crs`` ``None`` rather than guessed.
        - Node elevation becomes a ``node_z`` variable.

    Example:
        Read an Esri TIN by its directory path (shown, not run as a doctest -- it needs a
        real TIN directory)::

            from pyramids.netcdf.ugrid import MeshDataset

            mesh = MeshDataset.read_file("path/to/tin_directory")
            elevation = mesh.get("node_z")
    """

    name = "esri_tin"
    long_name = "Esri TIN"
    # An Esri TIN is a directory; read_file reaches it through any .adf component file.
    suffixes = (".adf",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is a TIN directory or an ``.adf`` file inside one.

        Overrides the base check, which declines every directory and only ever sniffs a
        file: an Esri TIN is a directory, reached either directly or through one of its
        ``.adf`` component files. Never raises -- a sniff error is warned and declined.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a TIN directory, or an ``.adf`` file whose directory is one.
        """
        tin_dir = self._tin_dir(path)
        if tin_dir is None:
            return False
        try:
            return self._sniff(tin_dir)
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            # Defensive: os.listdir on a readable directory does not raise, so this guard
            # mirrors the base class's never-raise contract rather than a reachable path.
            warnings.warn(  # pragma: no cover
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False  # pragma: no cover

    @staticmethod
    def _tin_dir(path: str) -> str | None:
        """Resolve the TIN directory from a directory or an ``.adf`` file path."""
        if os.path.isdir(path):
            return path
        if os.path.isfile(path) and path.lower().endswith(".adf"):
            return os.path.dirname(path)
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether the directory holds the required TIN components."""
        present = {entry.lower() for entry in os.listdir(path)}
        has_required = all(component in present for component in _REQUIRED)
        has_header = any(header in present for header in _TDENV)
        return has_required and has_header

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a TIN into a :class:`ParsedMesh`, from its directory or an ``.adf`` file.

        Args:
            path: The TIN directory, or one of its ``.adf`` component files.
            lazy: Unused -- a TIN is small and read whole.

        Returns:
            ParsedMesh: Nodes, node elevation and the unmasked triangle connectivity.

        Raises:
            FileFormatNotSupportedError: If the path is not a TIN or a component is missing.
        """
        tin_dir = self._tin_dir(path)
        if tin_dir is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an Esri TIN directory or an .adf file inside one"
            )
        return _EsriTinReader(tin_dir).build()


class _EsriTinReader:
    """One Esri TIN directory: read the components and assemble the mesh."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Read every component and build the masked triangular mesh.

        Returns:
            ParsedMesh: The surface mesh with node elevation.

        Raises:
            FileFormatNotSupportedError: If a required component is missing.
        """
        self._require()
        coordinates = self._read_big_endian("tnxy.adf", np.dtype(">f8")).reshape(-1, 2)
        node_x = coordinates[:, 0].astype(np.float64)
        node_y = coordinates[:, 1].astype(np.float64)
        node_z = self._read_big_endian("tnz.adf", np.dtype(">f4")).astype(np.float64)
        triples = self._read_big_endian("tnod.adf", np.dtype(">i4")).reshape(-1, 3)
        keep = self._keep_mask(len(triples))
        faces = (triples[keep] - 1).astype(np.int64)  # 1-based -> 0-based
        # The masked hull-frame triangles are gone, but the frame nodes they referenced
        # still sit in the node table, unreferenced -- drop them and reindex so the extent
        # is the surface, not the dropped frame (issue #79).
        node_x, node_y, node_z, faces = _drop_unreferenced_nodes(
            node_x, node_y, node_z, faces
        )
        crs = self._read_projection()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=0,
            original_start_index=1,  # tnod.adf is 1-based; recorded for a writer
            node_z=node_z,
            crs=crs,
            variables={},
            global_attributes={"source": "Esri TIN"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self) -> None:
        """Refuse a directory missing a required component."""
        present = {entry.lower() for entry in os.listdir(self.path)}
        missing = [name for name in _REQUIRED if name not in present]
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an Esri TIN: missing {missing}"
            )

    def _read_big_endian(self, name: str, dtype: np.dtype) -> np.ndarray:
        """Read one ``.adf`` component as a big-endian array of ``dtype``."""
        with open(os.path.join(self.path, name), "rb") as handle:
            return np.frombuffer(handle.read(), dtype=dtype)

    def _keep_mask(self, n_triangles: int) -> np.ndarray:
        """Return a boolean mask of the triangles to keep.

        The hull-frame triangles are flagged in ``tmsk.adf``; a set bit drops a triangle.
        ``tmsx.adf`` records, in its last word, how far from the end of ``tmsk.adf`` the
        mask block sits. When the mask files are absent, every triangle is kept.
        """
        present = {entry.lower() for entry in os.listdir(self.path)}
        if "tmsk.adf" not in present or "tmsx.adf" not in present:
            return np.ones(n_triangles, dtype=bool)
        index = self._read_big_endian("tmsx.adf", np.dtype(">i4"))
        with open(os.path.join(self.path, "tmsk.adf"), "rb") as handle:
            mask_bytes = handle.read()
        mask_begin = int(index[-1])
        start = len(mask_bytes) - mask_begin * 2
        header = np.frombuffer(mask_bytes, dtype=">i4", count=3, offset=start)
        word_count = int(header[0])
        words = np.frombuffer(
            mask_bytes, dtype=">u4", count=word_count, offset=start + 12
        )
        return self._unmask(words, n_triangles)

    def _unmask(self, words: np.ndarray, n_triangles: int) -> np.ndarray:
        """Decode the mask bitstream into a keep mask (a set bit excludes a triangle).

        Each triangle's bit lives at ``words[i // 32]`` bit ``i % 32`` (LSB-first).

        Raises:
            FileFormatNotSupportedError: If the word block is too short to cover every
                triangle -- a corrupt ``tmsk.adf`` -- rather than a bare ``IndexError``.
        """
        if words.size * 32 < n_triangles:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a corrupt Esri TIN mask: {words.size} words cover "
                f"{words.size * 32} triangles, fewer than the {n_triangles} in the table"
            )
        triangle = np.arange(n_triangles)
        bits = (words[triangle // 32] >> (triangle % 32)) & 1
        return bits == 0

    def _read_projection(self):
        """Return a CRS from ``prj.adf`` when it parses, else ``None``.

        ``prj.adf`` holds WKT in Esri's dialect (``GCS_WGS_1984``, ``D_WGS_1984``), which a
        strict OGC validation rejects. Decoding it is this driver's job; the shared
        :func:`pyramids_mesh._crs.crs_from_esri_wkt` morph -> authority-code-or-WKT step is
        the same one HEC-RAS uses.
        """
        projection = os.path.join(self.path, "prj.adf")
        if not os.path.isfile(projection):
            return None
        with open(projection, "rb") as handle:
            # The whole file, not just the first line: a producer may wrap the WKT across
            # several lines, and OGR's parser tolerates the embedded whitespace.
            wkt = handle.read().decode("latin-1")
        return crs_from_esri_wkt(wkt)
