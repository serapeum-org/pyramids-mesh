"""DHI MIKE drivers: DFSU (and DFS2 to come). Requires the ``dhi`` extra.

DFSU and DFS2 are proprietary DHI binary containers that cannot be read without DHI's own
Python library, ``mikeio`` -- the heaviest extra in this package (it pulls ``mikecore``,
pandas, matplotlib, scipy and tqdm), which is why it is isolated in ``[dhi]``.

``mikeio`` is imported at this module's top, so the driver registry skips the DHI drivers
with a warning when the extra is absent rather than failing the whole package; they appear
only when ``[dhi]`` is installed. Matplotlib's backend is pinned to the non-interactive
``Agg`` before ``mikeio`` is imported, so importing a driver never selects a GUI backend in
a headless run. Task D8; see ``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import matplotlib  # isort:skip

matplotlib.use("Agg")  # non-interactive; must precede the mikeio import below

import mikeio  # noqa: E402  # isort:skip

import numpy as np  # noqa: E402

from pyramids.base._errors import FileFormatNotSupportedError  # noqa: E402
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION  # noqa: E402

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable  # noqa: E402

# DFS binary files open with this ASCII tag; the cheap half of the sniff, so a foreign
# ``.dfsu`` is declined without paying for a full ``mikeio`` open.
_DFS_MAGIC = b"DHI_DFS_"


class DfsuDriver(BaseDriver):
    """Reader for the DHI MIKE DFSU unstructured-mesh format.

    DFSU carries both a mesh and cell-centred results, so the driver declares
    ``read_mesh`` and ``read_data``. The mesh comes from ``mikeio``'s geometry -- node
    coordinates and the (0-based) element table -- node elevation becomes a ``node_z``
    variable, and each item becomes a face-located variable over time.

    Documented limitations:
        - **Requires the ``dhi`` extra.** Without ``mikeio`` installed the driver is not
          registered at all; nothing here is reachable.
        - **Elements are cell-centred, so every item is face-located.** Mixed triangles
          and quads are padded to the widest row with ``-1``.
        - **The projection is mapped to a CRS only when it is a real one.** DHI writes an
          internal projection name such as ``UTM-33`` or ``LONG/LAT``, which carries no
          datum this package can resolve, so ``crs`` is left ``None`` rather than guessed.
        - **Values are lazy.** The loader reopens the file and reads one item, closing
          over a path and an item name, never an open handle (design section 7).
        - The element table ``mikeio`` returns is already 0-based, so the recorded base is
          0.

    Example:
        Read a DFSU mesh and its results (shown, not run as a doctest -- it needs the
        ``dhi`` extra and a real DFSU file)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("results.dfsu")
            level = mesh.get("Surface elevation")   # face-located, over time
    """

    name = "dfsu"
    long_name = "DHI MIKE DFSU"
    suffixes = (".dfsu",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens with the DFS binary tag."""
        with open(path, "rb") as handle:
            return handle.read(len(_DFS_MAGIC)) == _DFS_MAGIC

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a DFSU file into a :class:`ParsedMesh`.

        Args:
            path: The DFSU file.
            lazy: When ``True``, leave each item behind a loader.

        Returns:
            ParsedMesh: The mesh, ``node_z`` and the per-element variables.

        Raises:
            FileFormatNotSupportedError: If ``mikeio`` cannot open the file as a DFSU
                mesh (for example a DFS0/DFS2 that is not an unstructured mesh).
        """
        return _DfsuReader(path, lazy=lazy).build()


class _DfsuLoader:
    """A lazy loader for one DFSU item, reopening the file per read (design section 7)."""

    def __init__(self, path: str, item_name: str) -> None:
        self.path = path
        self.item_name = item_name

    def __call__(self) -> np.ndarray:
        """Reopen the file and read this item's values as ``(n_time, n_element)``."""
        dataset = mikeio.open(self.path).read(items=[self.item_name])
        return np.asarray(dataset[0].values, dtype=np.float64)


class _DfsuReader:
    """One DFSU file: build the mesh and its per-element variables through mikeio."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Open the file, read geometry and items, and assemble the mesh.

        Raises:
            FileFormatNotSupportedError: If mikeio cannot open the file as a mesh.
        """
        try:
            opened = mikeio.open(self.path)
        except Exception as error:  # noqa: BLE001 - mikeio raises various types
            raise FileFormatNotSupportedError(
                f"{self.path!r} could not be opened as a DFSU mesh: {error}"
            ) from error
        geometry = opened.geometry
        coordinates = np.asarray(geometry.node_coordinates, dtype=np.float64)
        node_x = coordinates[:, 0]
        node_y = coordinates[:, 1]
        node_z = coordinates[:, 2] if coordinates.shape[1] > 2 else None
        faces = self._faces(geometry.element_table)
        face_x, face_y = self._face_centres(geometry)
        times = self._times(opened)
        variables = self._variables(opened, times)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # mikeio normalises the element table to 0-based; nothing to subtract.
            start_index=0,
            original_start_index=0,
            node_z=node_z,
            face_x=face_x,
            face_y=face_y,
            crs=None,
            variables=variables,
            global_attributes={"source": "DHI MIKE DFSU"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    @staticmethod
    def _faces(element_table) -> np.ndarray:
        """Pad the (0-based) element table to a rectangular ``-1``-filled array.

        DFSU mixes triangles and quads; each row is padded to the widest element with
        ``-1`` so the connectivity is rectangular.
        """
        rows = [np.asarray(element, dtype=np.int64) for element in element_table]
        width = max(len(row) for row in rows)
        faces = np.full((len(rows), width), -1, dtype=np.int64)
        for index, row in enumerate(rows):
            faces[index, : len(row)] = row
        return faces

    @staticmethod
    def _face_centres(geometry):
        """Return the element-centre coordinates as face_x/face_y, when exposed."""
        centres = getattr(geometry, "element_coordinates", None)
        if centres is None:
            return None, None
        centres = np.asarray(centres, dtype=np.float64)
        return centres[:, 0], centres[:, 1]

    @staticmethod
    def _times(opened) -> np.ndarray:
        """Return the time axis as seconds from the first step."""
        time = opened.time
        start = time[0]
        return np.array(
            [(step - start).total_seconds() for step in time], dtype=np.float64
        )

    def _variables(self, opened, times: np.ndarray) -> dict[str, ParsedVariable]:
        """Build a lazy face variable for each item, sized from the header.

        The item values are shaped ``(n_time, n_element)``; the shape is taken from the
        item count and the element count without reading the values.
        """
        n_time = len(times)
        n_element = opened.geometry.n_elements
        variables: dict[str, ParsedVariable] = {}
        for item in opened.items:
            name = item.name
            loader = _DfsuLoader(self.path, name)
            variables[name] = ParsedVariable(
                location="face",
                shape=(n_time, n_element),
                loader=loader if self.lazy else None,
                values=None if self.lazy else loader(),
                units=str(item.unit) if getattr(item, "unit", None) else None,
                time_values=times,
            )
        return variables
