"""MPAS mesh reader (task C6).

MPAS (Model for Prediction Across Scales) writes its unstructured grid as NetCDF with
``Conventions = "MPAS"``. The grid is a spherical centroidal Voronoi tessellation: a
hex-dominant **primal** mesh of cells and a **dual** mesh whose vertices are the Voronoi
corners and whose faces are the primal cells. This driver reads the dual: the Voronoi
corners become the mesh nodes and each primal cell becomes a face listing the corners
around it.

The corner list per cell is ``verticesOnCell`` -- a ``(nCells, maxEdges)`` table that is
**1-based** and padded with ``0``, with ``nEdgesOnCell`` giving each cell's valid count.
Subtracting one and dropping the padding yields pyramids' fill-padded, mixed-degree
face-node connectivity: a coarse quasi-uniform mesh is 12 pentagons among hexagons, so
the face table carries both 5- and 6-node rows.

Read through the NetCDF substrate the vendored GDAL already provides -- the same
multidimensional route the other NetCDF drivers use, with no declared ``netCDF4`` or
``h5py``. MPAS grids use ``.nc``; they are reached through ``read_file``'s NetCDF-suffix
fallback (``pyramids-core`` 0.69.0), since an MPAS grid is not UGRID.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The dual-vertex coordinates (the Voronoi corners), in radians, and the per-cell
# corner table that lists the dual vertices around each primal cell.
_LON_VERTEX = "lonVertex"
_LAT_VERTEX = "latVertex"
_N_EDGES_ON_CELL = "nEdgesOnCell"
_VERTICES_ON_CELL = "verticesOnCell"

# The attribute whose value marks a file as MPAS, and the token looked for in it.
_CONVENTIONS = "Conventions"
_MPAS_TOKEN = "MPAS"


def _group_attributes(group) -> dict[str, object]:
    """Return a group's attributes as a name -> value mapping.

    GDAL exposes attributes as objects rather than a dict; an attribute whose value
    cannot be read is skipped rather than allowed to abort the sniff.

    Args:
        group: An open GDAL multidimensional group.

    Returns:
        dict[str, object]: The readable attributes, keyed by name.
    """
    attributes: dict[str, object] = {}
    for attribute in group.GetAttributes():
        try:
            attributes[attribute.GetName()] = attribute.Read()
        except RuntimeError:
            continue
    return attributes


class MpasDriver(BaseDriver):
    """Reader for MPAS NetCDF grids -- the Voronoi dual as a mixed-degree mesh.

    The mesh nodes are the dual vertices (``lonVertex``/``latVertex``) and each face is a
    primal cell, listing the dual vertices around it from ``verticesOnCell``. That table
    is 1-based and ``0``-padded, with ``nEdgesOnCell`` giving each cell's valid corner
    count; subtracting one and dropping the padding gives fill-padded, mixed-degree
    face-node connectivity (5- and 6-node faces in a hex-dominant mesh).

    Documented limitations:
        - **Longitude/latitude are exposed, in the file's native radians, not the
          Cartesian ``xVertex``/``yVertex``.** MPAS stores both: lon/lat (radians) and the
          3D Cartesian coordinates on the sphere. Dropping ``z`` from the Cartesian triple
          collapses the two hemispheres onto one another and degenerates the cells at the
          limb, so the geographic coordinates are used instead. They are carried verbatim
          in radians: converting to degrees is a projection concern that belongs in the
          GIS layer upstream (pyramids), not in a file-format driver -- the same rule the
          ICON reader follows. A cell straddling the antimeridian unwraps to a wide planar
          polygon -- inherent to flattening a global mesh -- but its winding is still well
          defined.
        - **No CRS.** The grid lives on a unit sphere (``sphere_radius = 1``) with no EPSG
          authority, so ``crs`` is ``None`` rather than guessed at ``4326`` (a WGS84
          ellipsoid the mesh is not on).
        - **The primal (hex) cells are the faces; the dual vertices are the nodes.** Only
          the topology needed for that is read -- coordinates and ``verticesOnCell``. The
          many other MPAS fields (``cellsOnCell``, edge tables, areas, quality metrics)
          are not shipped as variables; this driver declares ``read_mesh`` only.
        - **``verticesOnCell`` is 1-based with ``0`` padding** -- resolved to 0-based and
          the base ``1`` recorded for a writer.

    Example:
        Read an MPAS grid (shown, not run as a doctest -- it needs a real NetCDF file),
        reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("mesh.QU.1920km.nc")
            mesh.mesh.n_node          # the Voronoi dual vertices
            mesh.mesh.n_face          # the primal (hex-dominant) cells
    """

    name = "mpas"
    long_name = "MPAS mesh"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is an MPAS grid.

        Keyed on ``Conventions`` containing ``MPAS`` *and* the presence of
        ``verticesOnCell``, so a plain NetCDF, a UGRID mesh, or another model's ``.nc``
        (3Di, TUFLOW FV) is not claimed. Opening a non-NetCDF file raises inside GDAL and
        is reported as "not MPAS" rather than propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the MPAS convention and corner table are both present.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                conventions = _group_attributes(group).get(_CONVENTIONS, "")
            finally:
                dataset.Close()
        return _MPAS_TOKEN in str(conventions) and _VERTICES_ON_CELL in names

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an MPAS grid into a :class:`ParsedMesh`.

        Args:
            path: The MPAS ``.nc`` file.
            lazy: Unused -- this driver reads only the topology, which is eager.

        Returns:
            ParsedMesh: The Voronoi dual as nodes plus mixed-degree face connectivity.

        Raises:
            FileFormatNotSupportedError: If a defining array is missing.
        """
        return _MpasReader(path).build()


class _MpasReader:
    """One MPAS file: read the dual vertices and the per-cell corner table."""

    #: The arrays the mesh cannot be built without.
    _REQUIRED = (_LON_VERTEX, _LAT_VERTEX, _N_EDGES_ON_CELL, _VERTICES_ON_CELL)

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file and assemble the dual mesh.

        Returns:
            ParsedMesh: Nodes from ``lonVertex``/``latVertex`` (radians) and face-node
                connectivity from ``verticesOnCell``.

        Raises:
            FileFormatNotSupportedError: If a defining array is missing.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                self._require(names)
                lon = self._read(group, _LON_VERTEX)
                lat = self._read(group, _LAT_VERTEX)
                n_edges = self._read(group, _N_EDGES_ON_CELL).astype(np.int64)
                vertices_on_cell = self._read(group, _VERTICES_ON_CELL).astype(np.int64)
            finally:
                dataset.Close()
        faces = self._faces(vertices_on_cell, n_edges)
        return ParsedMesh(
            # lon/lat are carried in the file's native radians; converting to degrees is a
            # projection concern that belongs upstream (pyramids), not in a format driver.
            # The Cartesian xVertex/yVertex are deliberately not used (see the class docs).
            node_x=lon,
            node_y=lat,
            faces=faces,
            fill_value=-1,
            # verticesOnCell is 1-based; resolved to 0-based here, base recorded for a
            # writer.
            start_index=0,
            original_start_index=1,
            # The grid is on a unit sphere with no EPSG authority -- see the class docs.
            crs=None,
            global_attributes={"source": "MPAS mesh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    @staticmethod
    def _read(group, name: str) -> np.ndarray:
        """Read one array from the group as a NumPy array."""
        return np.asarray(group.OpenMDArray(name).ReadAsArray())

    def _require(self, names: set[str]) -> None:
        """Refuse a file missing any defining array.

        Raises:
            FileFormatNotSupportedError: If a required array is absent.
        """
        missing = [name for name in self._REQUIRED if name not in names]
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an MPAS grid: missing {sorted(missing)}"
            )

    @staticmethod
    def _faces(vertices_on_cell: np.ndarray, n_edges: np.ndarray) -> np.ndarray:
        """Resolve ``verticesOnCell`` to fill-padded, 0-based face connectivity.

        Each cell keeps its first ``nEdgesOnCell`` corners; the rest of the row is the
        format's ``0`` padding and becomes ``-1`` fill. Valid corners drop from 1-based
        to 0-based. The result is a mixed-degree table -- 5-node pentagons among 6-node
        hexagons in a quasi-uniform mesh.

        Args:
            vertices_on_cell: The ``(nCells, maxEdges)`` 1-based, ``0``-padded corner
                table.
            n_edges: The per-cell valid corner count, ``(nCells,)``.

        Returns:
            np.ndarray: The ``(nCells, maxEdges)`` connectivity, 0-based and ``-1``-padded.
        """
        max_edges = vertices_on_cell.shape[1]
        columns = np.arange(max_edges)
        valid = columns[None, :] < n_edges[:, None]
        return np.where(valid, vertices_on_cell - 1, -1).astype(np.int64)
