"""HEC-RAS 2D driver (task D11).

HDF5 plan file carrying one or more 2D flow-area meshes and their results. The internal
schema is undocumented and has changed across HEC-RAS releases, so this reader keys on the
structures it understands and fails loudly on an unrecognised layout rather than reading
what merely looks familiar.

HDF5 is read through the GDAL multidimensional API, over the GDAL vendored in the
``pyramids-core`` wheel and activated by ``import pyramids`` -- group traversal, attributes
and typed array reads, with no declared dependency and no ``h5py``. See
``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import warnings

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# The 8-byte signature every HDF5 file opens with; the cheap half of the sniff.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# The geometry group tree and the per-area arrays that define a 2D flow-area mesh.
_GEOMETRY = "Geometry"
_AREAS = "2D Flow Areas"
_CELL_INFO = "Cell Info"  # (n_area, 2) = [start, count] into Cell Points, per area
_CELL_POINTS = "Cell Points"  # (n_cell_total, 2) cell-centre coords, areas concatenated
_FACEPOINTS = "FacePoints Coordinate"  # (n_node, 2) the mesh nodes
_FACE_FP = "Faces FacePoint Indexes"  # (n_face, 2) each face's two facepoints
_CELL_FACE_INFO = "Cells Face and Orientation Info"  # (n_cell, 2) = [offset, count]
_CELL_FACE_VALUES = "Cells Face and Orientation Values"  # (n, 2) = [face index, orient]

# The summary-results group tree, and the per-cell result arrays shaped (2, n_cell): the
# first row is the value, the second the time of occurrence -- only the value is carried.
_SUMMARY_PATH = (
    "Results",
    "Unsteady",
    "Output",
    "Output Blocks",
    "Base Output",
    "Summary Output",
    "2D Flow Areas",
)
_SUMMARY_VARIABLES = ("Maximum Water Surface", "Minimum Water Surface")


class HecRasDriver(BaseDriver):
    """Reader for the HEC-RAS 2D HDF5 plan format.

    A plan file holds one or more named 2D flow areas, each an unstructured mesh of
    polygonal cells, plus unsteady results. The mesh nodes are the ``FacePoints``; a cell's
    polygon is reconstructed from its ordered faces (``Cells Face and Orientation`` into
    ``Faces FacePoint Indexes``). The per-cell summary results (maximum/minimum water
    surface) in the plan file become face-located variables.

    Documented limitations:
        - **Only the first 2D flow area is read.** A plan file may hold several; selecting a
          second 2D mesh by name is not wired through the read template yet, so the first
          area is read and the others are named in a warning rather than dropped silently.
        - **Ghost/boundary cells are excluded.** HEC-RAS appends perimeter cells (a single
          face each) after the real cells; ``Cell Info`` gives the real-cell count, and only
          those closed polygons become faces. The per-cell result arrays, which are sized to
          include the ghost cells, are sliced to the real cells to match.
        - **Summary results only.** The plan file carries the maximum/minimum water-surface
          summary; the per-timestep series lives in a separate, geometry-stripped
          ``*.timeseries.hdf`` companion, which cannot be paired into a single read here.
        - **Face velocities are not exposed.** They are edge-located (one per face, not per
          cell); this reader builds the cell mesh only.
        - There is no CRS in the arrays read here, so ``crs`` is ``None``.

    Example:
        Read a HEC-RAS 2D plan file (shown, not run as a doctest -- it needs a real HDF5
        file)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("model.p01.hdf")
            wse = mesh.get("Maximum Water Surface")   # face-located, per cell
    """

    name = "hecras"
    long_name = "HEC-RAS 2D"
    suffixes = (".hdf", ".hdf5")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a HEC-RAS 2D plan: HDF5 plus a 2D flow-area mesh.

        The magic is checked cheaply first; then the file is opened to confirm a
        ``Geometry/2D Flow Areas`` group with at least one area carrying the defining
        ``Faces FacePoint Indexes`` geometry. ``.hdf`` is shared with other HDF5 formats, so
        the group layout -- not the suffix -- decides, and a results-only (geometry-stripped)
        companion is declined.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the magic matches and a 2D flow-area mesh is present.
        """
        with open(path, "rb") as handle:
            if handle.read(len(_HDF5_MAGIC)) != _HDF5_MAGIC:
                return False
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                root = dataset.GetRootGroup()
                if _GEOMETRY not in root.GetGroupNames():
                    return False
                geometry = root.OpenGroup(_GEOMETRY)
                if _AREAS not in geometry.GetGroupNames():
                    return False
                areas = geometry.OpenGroup(_AREAS)
                return any(
                    _FACE_FP in areas.OpenGroup(name).GetMDArrayNames()
                    for name in areas.GetGroupNames()
                )
            finally:
                dataset.Close()

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read the first 2D flow area of a HEC-RAS plan file into a :class:`ParsedMesh`.

        Args:
            path: The HEC-RAS ``.hdf`` plan file.
            lazy: When ``True``, leave each summary variable behind a loader.

        Returns:
            ParsedMesh: The flow-area mesh and its summary water-surface variables.

        Raises:
            FileFormatNotSupportedError: If no readable 2D flow area is present.
        """
        return _HecRasReader(path, lazy=lazy).build()


class _HecRasLoader:
    """A lazy loader for one HEC-RAS summary variable's per-cell values.

    Closes over the path and the group path, never an open handle (design section 7):
    reading it reopens the file, walks to the array, and returns the value row sliced to the
    real-cell count.
    """

    def __init__(
        self, path: str, group_path: tuple[str, ...], array_name: str, n_cells: int
    ) -> None:
        self.path = path
        self.group_path = group_path
        self.array_name = array_name
        self.n_cells = n_cells

    def __call__(self) -> np.ndarray:
        """Reopen the file and read this summary variable's value row."""
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                for name in self.group_path:
                    group = group.OpenGroup(name)
                values = np.asarray(group.OpenMDArray(self.array_name).ReadAsArray())
                return values[0, : self.n_cells].astype(np.float64)
            finally:
                dataset.Close()


class _HecRasReader:
    """One HEC-RAS plan file: read the first 2D flow area's mesh and summary results."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Open the file, build the first flow area's mesh, and gather summary variables.

        Returns:
            ParsedMesh: The 2D flow-area surface mesh with its summary variables.

        Raises:
            FileFormatNotSupportedError: If no readable 2D flow area is present.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                areas = self._areas_group(dataset)
                area = self._select_area(areas)
                start, count = self._cell_slice(areas, area)
                flow_area = areas.OpenGroup(area)
                node_x, node_y = self._nodes(flow_area)
                faces = self._faces(flow_area, count)
                face_x, face_y = self._centres(areas, start, count)
                variables = self._summary_variables(dataset, area, count)
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=0,
            original_start_index=0,  # facepoint indices are already 0-based
            face_x=face_x,
            face_y=face_y,
            crs=None,
            variables=variables,
            global_attributes={"source": "HEC-RAS 2D", "flow_area": area},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _areas_group(self, dataset):
        """Return the ``Geometry/2D Flow Areas`` group, or refuse the file."""
        root = dataset.GetRootGroup()
        if _GEOMETRY not in root.GetGroupNames():
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a HEC-RAS plan: no {_GEOMETRY!r} group"
            )
        geometry = root.OpenGroup(_GEOMETRY)
        if _AREAS not in geometry.GetGroupNames():
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a HEC-RAS 2D plan: no {_AREAS!r} group"
            )
        return geometry.OpenGroup(_AREAS)

    def _select_area(self, areas) -> str:
        """Return the first 2D flow area that carries mesh geometry, warning of the rest."""
        named = [
            name
            for name in areas.GetGroupNames()
            if _FACE_FP in areas.OpenGroup(name).GetMDArrayNames()
        ]
        if not named:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a {_AREAS!r} group but no area carries the "
                f"{_FACE_FP!r} geometry; it may be a results-only companion file"
            )
        if len(named) > 1:
            warnings.warn(
                f"{self.path!r} holds {len(named)} 2D flow areas {named!r}; only the "
                f"first ({named[0]!r}) is read -- selecting another is not wired yet",
                stacklevel=2,
            )
        return named[0]

    def _cell_slice(self, areas, area: str) -> tuple[int, int]:
        """Return the (start, count) of ``area``'s real cells within ``Cell Points``.

        ``Cell Info`` holds one ``[start, count]`` row per 2D flow area, in the order of
        the area subgroups (which HEC-RAS writes in the order of the ``Attributes`` table;
        that table's string ``Name`` field is not readable through the vendored GDAL SWIG
        bindings, so the subgroup order stands in for it). The row is therefore bound by the
        area's position among **all** subgroups, not among the geometry-carrying ones only:
        a geometry-less area must not shift a real area's row. The count is the number of
        real (non-ghost) cells.

        Raises:
            FileFormatNotSupportedError: If ``Cell Info`` is absent, its row count is not one
                per area, or the area reports no cells.
        """
        all_names = list(areas.GetGroupNames())
        if _CELL_INFO not in areas.GetMDArrayNames():
            raise FileFormatNotSupportedError(
                f"{self.path!r} 2D flow areas carry no {_CELL_INFO!r} array"
            )
        cell_info = np.asarray(areas.OpenMDArray(_CELL_INFO).ReadAsArray())
        if cell_info.shape[0] != len(all_names):
            raise FileFormatNotSupportedError(
                f"{self.path!r} has {cell_info.shape[0]} {_CELL_INFO!r} rows for "
                f"{len(all_names)} 2D flow areas; cannot bind {area!r} to its cells"
            )
        row = all_names.index(area)
        start, count = int(cell_info[row, 0]), int(cell_info[row, 1])
        if count <= 0:
            raise FileFormatNotSupportedError(
                f"{self.path!r} flow area {area!r} reports {count} cells"
            )
        # Cross-check the bound row against the area's own geometry: its real-cell count
        # cannot exceed the total cells it describes (one Cells Face and Orientation Info
        # row per cell). The subgroup order is only a stand-in for the unreadable
        # Attributes-table order, so this catches a mis-bound row -- e.g. were GDAL ever to
        # return subgroups sorted rather than in definition order -- and refuses rather than
        # silently reading another area's cells.
        flow_area = areas.OpenGroup(area)
        if _CELL_FACE_INFO in flow_area.GetMDArrayNames():
            total = int(
                np.asarray(flow_area.OpenMDArray(_CELL_FACE_INFO).ReadAsArray()).shape[
                    0
                ]
            )
            if count > total:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} bound flow area {area!r} to a {_CELL_INFO!r} row of "
                    f"{count} real cells, but the area describes only {total}; the "
                    f"subgroup order does not match the {_CELL_INFO!r} order"
                )
        return start, count

    def _nodes(self, flow_area) -> tuple[np.ndarray, np.ndarray]:
        """Return the mesh node coordinates from ``FacePoints Coordinate``."""
        coords = np.asarray(
            flow_area.OpenMDArray(_FACEPOINTS).ReadAsArray(), dtype=np.float64
        )
        return coords[:, 0], coords[:, 1]

    def _faces(self, flow_area, count: int) -> np.ndarray:
        """Reconstruct the real cells' node rings from the face/orientation tables.

        For each cell, its faces are listed in ring order as ``[face index, orientation]``;
        a face's two facepoints come from ``Faces FacePoint Indexes``, and the orientation
        picks which facepoint starts the edge. Taking the start facepoint of each face in
        order yields the cell's ordered polygon. Rows are right-padded with ``-1`` to the
        widest ring.

        Args:
            flow_area: The flow-area group.
            count: The number of real cells (the first ``count`` rows).

        Returns:
            np.ndarray: ``(count, max_nodes)`` 0-based connectivity, ``-1``-padded.
        """
        info = np.asarray(
            flow_area.OpenMDArray(_CELL_FACE_INFO).ReadAsArray(), dtype=np.int64
        )
        values = np.asarray(
            flow_area.OpenMDArray(_CELL_FACE_VALUES).ReadAsArray(), dtype=np.int64
        )
        face_fp = np.asarray(
            flow_area.OpenMDArray(_FACE_FP).ReadAsArray(), dtype=np.int64
        )
        rings: list[list[int]] = []
        width = 0
        for cell in range(count):
            offset, n_faces = int(info[cell, 0]), int(info[cell, 1])
            ring = [
                int(face_fp[face_idx, 0] if orient == 1 else face_fp[face_idx, 1])
                for face_idx, orient in values[offset : offset + n_faces]
            ]
            rings.append(ring)
            width = max(width, len(ring))
        faces = np.full((count, width), -1, dtype=np.int64)
        for row, ring in enumerate(rings):
            faces[row, : len(ring)] = ring
        return faces

    def _centres(self, areas, start: int, count: int):
        """Return this area's cell-centre coordinates from ``Cell Points``, if present."""
        if _CELL_POINTS not in areas.GetMDArrayNames():
            return None, None
        points = np.asarray(
            areas.OpenMDArray(_CELL_POINTS).ReadAsArray(), dtype=np.float64
        )
        area_points = points[start : start + count]
        return area_points[:, 0], area_points[:, 1]

    def _summary_variables(
        self, dataset, area: str, count: int
    ) -> dict[str, ParsedVariable]:
        """Build a face-located variable for each summary result the plan file carries.

        Each summary array is shaped ``(2, n_cell)`` -- the value row and the time-of-
        occurrence row; only the value row is carried, sliced to the real-cell count.
        """
        group = dataset.GetRootGroup()
        for part in _SUMMARY_PATH:
            if part not in group.GetGroupNames():
                return {}
            group = group.OpenGroup(part)
        if area not in group.GetGroupNames():
            return {}
        summary = group.OpenGroup(area)
        present = set(summary.GetMDArrayNames())
        group_path = _SUMMARY_PATH + (area,)
        variables: dict[str, ParsedVariable] = {}
        for name in _SUMMARY_VARIABLES:
            if name not in present:
                continue
            loader = _HecRasLoader(self.path, group_path, name, count)
            variables[name] = ParsedVariable(
                location="face",
                shape=(count,),
                loader=loader if self.lazy else None,
                values=None if self.lazy else loader(),
            )
        return variables
