"""ESMF unstructured-mesh reader.

ESMF (the Earth System Modeling Framework) stores its unstructured mesh as NetCDF with
an explicit node list and element connectivity: node coordinates in ``nodeCoords``
(shape ``nodeCount`` x ``coordDim``, with column 0 longitude and column 1 latitude), the
per-element node table in ``elementConn`` (``elementCount`` x ``maxNodePElement``), and
``numElementConn`` giving each element's real node count. The driver builds the surface
mesh -- the node list becomes the nodes and ``elementConn`` the fill-padded, mixed-degree
face connectivity.

Read through the NetCDF substrate the vendored GDAL provides, with no declared
``netCDF4`` or ``h5py``. ESMF meshes use ``.nc``; they are reached through
``read_file``'s NetCDF-suffix fallback (``pyramids-gis`` 0.69.0), since an ESMF mesh is
not UGRID.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The arrays that make a NetCDF file an ESMF mesh: the node list, the element table and
# the per-element node count. All three are unique to ESMF, so requiring them declines
# every sibling ``.nc`` format (SCRIP, GEOS, MPAS, ICON, 3Di, TUFLOW FV).
_NODE_COORDS = "nodeCoords"
_ELEMENT_CONN = "elementConn"
_NUM_ELEMENT_CONN = "numElementConn"

# The element table is 1-based by default; a ``start_index`` attribute overrides the base.
_START_INDEX = "start_index"
_DEFAULT_START_INDEX = 1


def _group_names(group) -> set[str]:
    """Return the set of root-group MD-array names.

    Args:
        group: An open GDAL multidimensional root group.

    Returns:
        set[str]: The array names in the group.
    """
    return set(group.GetMDArrayNames())


def _array_attribute(group, name: str, attribute_name: str):
    """Return one named attribute of an array, or ``None``.

    An attribute whose value cannot be read is reported as absent rather than allowed to
    abort the parse.

    Args:
        group: The open root group.
        name: The array whose attribute to read.
        attribute_name: The attribute to look for.

    Returns:
        object | None: The attribute's value, or ``None`` when missing or unreadable.
    """
    for attribute in group.OpenMDArray(name).GetAttributes():
        if attribute.GetName() == attribute_name:
            try:
                return attribute.Read()
            except RuntimeError:
                return None
    return None


def _to_degrees(values: np.ndarray, units: str) -> np.ndarray:
    """Return ``values`` in degrees, converting from radians when ``units`` says so.

    Args:
        values: A coordinate array in the file's native unit.
        units: The array's declared units.

    Returns:
        np.ndarray: The array in degrees -- converted when the units name radians, else
            returned unchanged.
    """
    if "rad" in units.lower():
        return np.degrees(values)
    return values


class EsmfDriver(BaseDriver):
    """Reader for ESMF unstructured-mesh NetCDF.

    The nodes come from ``nodeCoords`` (column 0 longitude, column 1 latitude) and the
    1-based ``elementConn`` table is re-based and fill-padded into mixed-degree face
    connectivity. ``numElementConn`` gives each element's real node count, so a triangle
    in a wider table keeps its first three nodes and the rest of the row becomes fill.

    Documented limitations:
        - **The real node count comes from ``numElementConn``, not from a fill value.**
          Each element keeps its first ``numElementConn`` nodes; the rest of the row is
          padding and becomes ``-1`` fill, so a mesh of triangles and quads yields a
          fill-padded, mixed-degree table without depending on the file's own pad value.
        - **``elementConn`` is 1-based by default.** A ``start_index`` attribute overrides
          the base; the connectivity is resolved to 0-based and the base recorded for a
          writer.
        - **Coordinates are carried in the file's unit, degrees by default.** A ``units``
          attribute on ``nodeCoords`` naming radians is honoured and the coordinates
          converted to degrees; otherwise they are left as the file gave them.
        - **No CRS.** The mesh is lon/lat on a sphere with no EPSG authority, so ``crs``
          is ``None`` rather than guessed.
        - **Only the topology is read.** The element areas and other ESMF fields are not
          shipped as variables; this driver declares ``read_mesh`` only.

    Example:
        Read an ESMF mesh (shown, not run as a doctest -- it needs a real NetCDF file),
        reached through ``read_file``'s NetCDF fallback::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("ne4np4-esmf.nc")
            mesh.mesh.n_node          # the ESMF node list
            mesh.mesh.n_face          # the ESMF elements
    """

    name = "esmf"
    long_name = "ESMF unstructured mesh"
    suffixes = (".nc",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is an ESMF mesh, by its defining arrays.

        Keyed on ``nodeCoords``, ``elementConn`` and ``numElementConn``, which together
        are unique to ESMF, so a plain NetCDF or another model's ``.nc`` -- SCRIP, GEOS,
        MPAS, ICON, 3Di or TUFLOW FV -- is not claimed. Opening a non-NetCDF file raises
        inside GDAL and is reported as "not ESMF" rather than propagated.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when all three defining arrays are present.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                names = _group_names(dataset.GetRootGroup())
            finally:
                dataset.Close()
        return {_NODE_COORDS, _ELEMENT_CONN, _NUM_ELEMENT_CONN} <= names

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an ESMF mesh into a :class:`ParsedMesh`.

        Args:
            path: The ESMF ``.nc`` file.
            lazy: Unused -- this driver reads only the topology, which is eager.

        Returns:
            ParsedMesh: The node list and the fill-padded, mixed-degree face
                connectivity.

        Raises:
            FileFormatNotSupportedError: If a defining array is missing.
        """
        return _EsmfReader(path).build()


class _EsmfReader:
    """One ESMF file: read the node list and the per-element node table."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file and assemble the surface mesh.

        Returns:
            ParsedMesh: Nodes from ``nodeCoords`` and faces from ``elementConn``.

        Raises:
            FileFormatNotSupportedError: If a defining array is missing.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = _group_names(group)
                self._require(names)
                node_coords = np.asarray(
                    group.OpenMDArray(_NODE_COORDS).ReadAsArray(), np.float64
                )
                units = str(_array_attribute(group, _NODE_COORDS, "units") or "")
                element_conn = np.asarray(
                    group.OpenMDArray(_ELEMENT_CONN).ReadAsArray(), np.int64
                )
                counts = np.asarray(
                    group.OpenMDArray(_NUM_ELEMENT_CONN).ReadAsArray(), np.int64
                )
                start_index = self._start_index(group)
            finally:
                dataset.Close()
        # Column 0 is longitude, column 1 latitude; both carried in the file's unit.
        node_x = _to_degrees(node_coords[:, 0], units)
        node_y = _to_degrees(node_coords[:, 1], units)
        faces = self._faces(element_conn, counts, start_index)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # elementConn resolved to 0-based here; the file's base recorded for a writer.
            start_index=0,
            original_start_index=start_index,
            # Lon/lat on a sphere with no EPSG authority -- see the class docs.
            crs=None,
            global_attributes={"source": "ESMF unstructured mesh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self, names: set[str]) -> None:
        """Refuse a file missing a defining array.

        Args:
            names: The array names present in the file.

        Raises:
            FileFormatNotSupportedError: If any defining array is absent.
        """
        missing = {_NODE_COORDS, _ELEMENT_CONN, _NUM_ELEMENT_CONN} - names
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an ESMF mesh: missing {sorted(missing)}"
            )

    @staticmethod
    def _start_index(group) -> int:
        """Return the connectivity base, from ``start_index`` or the ESMF default of 1.

        Args:
            group: The open root group.

        Returns:
            int: The element table's base; ``1`` when the attribute is absent.
        """
        value = _array_attribute(group, _ELEMENT_CONN, _START_INDEX)
        if value is None:
            return _DEFAULT_START_INDEX
        return int(np.asarray(value).ravel()[0])

    @staticmethod
    def _faces(
        element_conn: np.ndarray, counts: np.ndarray, start_index: int
    ) -> np.ndarray:
        """Resolve ``elementConn`` to fill-padded, 0-based face connectivity.

        Each element keeps its first ``numElementConn`` nodes, re-based to 0; every
        further slot is padding and becomes ``-1``, so the real node count comes from
        ``numElementConn`` rather than from the file's pad value.

        Args:
            element_conn: The ``(elementCount, maxNodePElement)`` node table.
            counts: The per-element real node count, ``(elementCount,)``.
            start_index: The base subtracted from the real nodes.

        Returns:
            np.ndarray: The ``(elementCount, maxNodePElement)`` connectivity, 0-based and
                ``-1``-padded.
        """
        max_width = element_conn.shape[1]
        columns = np.arange(max_width)
        valid = columns[None, :] < counts[:, None]
        return np.where(valid, element_conn - start_index, -1).astype(np.int64)
