"""Exodus II reader (task C10).

Exodus II is a finite-element mesh-and-results container that is NetCDF (CDF-2)
underneath: node coordinates in ``coordx``/``coordy``/``coordz``, and one or more element
blocks ``connect1``/``connect2``/... whose rows are 1-based element-to-node lists carrying
an ``elem_type`` attribute (``QUAD4``, ``TRI3``, ``TETRA``, ...). The block ids live in
``eb_prop1``.

The NetCDF substrate is read through the GDAL multidimensional API, over the GDAL vendored
in the ``pyramids-gis`` wheel and activated by ``import pyramids`` -- no declared
``netCDF4`` dependency and no ``exodus`` library. See ``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

_COORD_X = "coordx"
_COORD_Y = "coordy"
_COORD_Z = "coordz"
# Element blocks are numbered from 1: connect1, connect2, ...; each carries an elem_type.
_CONNECT_PREFIX = "connect"
_ELEM_TYPE = "elem_type"
_BLOCK_IDS = "eb_prop1"
# Dimensions every Exodus file declares; required so a plain NetCDF is not claimed. Only
# these two are universal -- a file with no elements omits num_elem, so it is not required.
_REQUIRED_DIMS = frozenset({"num_dim", "num_nodes"})

# Exodus connectivity is 1-based, so 0 is never a real node id and serves as the pad for
# the short rows of a mixed-width (e.g. triangles among quads) connectivity table. The
# shared normaliser rewrites it to the ``-1`` pyramids expects.
_PAD = 0


class ExodusDriver(BaseDriver):
    """Reader for the Exodus II (``.exo``) finite-element mesh format.

    The mesh comes from the node coordinates ``coordx``/``coordy`` (and ``coordz`` for a
    3D file, carried as ``node_z``) and the element blocks. Each block ``connectN`` is a
    1-based element-to-node table tagged with an ``elem_type``; every block's rows are
    concatenated into one face-node connectivity table, padded to the widest block so a
    mesh mixing triangles and quads is rectangular. The ids drop from 1-based to 0-based
    and the base is recorded for a writer; ``eb_prop1`` and the per-element ``elem_type``
    are carried as metadata.

    Documented limitations:
        - **Every element block's node list becomes a face row**, including a volume
          element type such as ``TETRA``. A genuine volumetric (3D) mesh is not modelled
          yet (design section 21.1), so a tetrahedron's four node ids are read verbatim as
          a face rather than expanded into its bounding triangles; winding is then judged
          in the ``xy`` plane like any other face.
        - **Connectivity is 1-based with mixed widths padded by ``0``.** A short row in a
          mixed-element table is padded with ``0`` -- never a valid 1-based id -- and the
          normaliser rewrites it to ``-1`` fill; the ``1`` base is recorded for a writer.
        - **Results are not read.** Only the mesh is built; the per-timestep nodal and
          element variables an Exodus file may carry are out of scope for this reader.
        - There is no CRS in the format, so ``crs`` is ``None``.

    Example:
        Read an Exodus mesh (shown, not run as a doctest -- it needs a real ``.exo``
        file), reached through ``read_file``::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("model.exo")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "exodus"
    long_name = "Exodus II"
    suffixes = (".exo", ".e")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is Exodus II, by its coordinates, blocks and dims.

        Keyed on ``coordx`` plus a first element block ``connect1`` and the ``num_dim`` /
        ``num_nodes`` dimensions, so a plain NetCDF or another model's ``.nc``-style file
        is declined rather than misread.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the defining arrays and dimensions are present.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            except RuntimeError:
                return False
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                dimensions = _dimension_sizes(group).keys()
            finally:
                dataset.Close()
        if not {_COORD_X, f"{_CONNECT_PREFIX}1"} <= names:
            return False
        return _REQUIRED_DIMS <= set(dimensions)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an Exodus II file into a :class:`ParsedMesh`.

        Args:
            path: The ``.exo``/``.e`` file.
            lazy: Unused -- the mesh is read eagerly and the format carries no lazy
                results here; accepted to match the driver protocol.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` for a 3D file, and the block-id and
                element-type metadata.

        Raises:
            FileFormatNotSupportedError: If the file carries no element block.
        """
        return _ExodusReader(path).build()


def _dimension_sizes(group) -> dict[str, int]:
    """Return every dimension's size, gathered from the root group's arrays."""
    sizes: dict[str, int] = {}
    for name in group.GetMDArrayNames():
        for dimension in group.OpenMDArray(name).GetDimensions():
            sizes[dimension.GetName()] = dimension.GetSize()
    return sizes


def _read_attribute(array, name: str) -> str | None:
    """Return a string attribute's value from a GDAL MD-array, or ``None`` when absent."""
    for attribute in array.GetAttributes():
        if attribute.GetName() == name:
            try:
                return str(attribute.Read())
            except RuntimeError:
                return None
    return None


def _pad_blocks(
    blocks: list[tuple[np.ndarray, str | None]],
) -> tuple[np.ndarray, list[str | None]]:
    """Concatenate element blocks into one table, padding short rows to the widest block.

    Each block is an ``(n_element, n_node_per_element)`` connectivity array; a block
    narrower than the widest (triangles among quads) is right-padded with ``_PAD`` so the
    stacked table is rectangular. The per-element ``elem_type`` is expanded to one entry
    per row, in block order.

    Args:
        blocks: One ``(connectivity, elem_type)`` pair per element block, in block order.

    Returns:
        tuple[np.ndarray, list[str | None]]: The padded ``(n_element, max_width)``
            connectivity and the per-element element types.

    Examples:
        - A triangle block beside a quad block pads the triangle rows to width 4 with the
          ``0`` that is never a valid 1-based Exodus id:

            >>> import numpy as np
            >>> tris = np.array([[1, 2, 3]])
            >>> quads = np.array([[2, 3, 4, 5]])
            >>> faces, types = _pad_blocks([(tris, "TRI3"), (quads, "QUAD4")])
            >>> faces.tolist()
            [[1, 2, 3, 0], [2, 3, 4, 5]]
            >>> types
            ['TRI3', 'QUAD4']
    """
    max_width = max(block.shape[1] for block, _ in blocks)
    padded: list[np.ndarray] = []
    element_types: list[str | None] = []
    for block, elem_type in blocks:
        rows = block.astype(np.int64)
        if rows.shape[1] < max_width:
            fill = np.full((rows.shape[0], max_width - rows.shape[1]), _PAD, np.int64)
            rows = np.concatenate([rows, fill], axis=1)
        padded.append(rows)
        element_types.extend([elem_type] * block.shape[0])
    return np.concatenate(padded, axis=0), element_types


class _ExodusReader:
    """One Exodus II file: read the node coordinates and the element blocks."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file and assemble the mesh from its coordinates and element blocks.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` for a 3D file, and metadata.

        Raises:
            FileFormatNotSupportedError: If the file carries no element block.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                names = set(group.GetMDArrayNames())
                node_x, node_y, node_z = self._coordinates(group, names)
                blocks = self._blocks(group, names)
                block_ids = self._block_ids(group, names)
            finally:
                dataset.Close()
        faces, element_types = _pad_blocks(blocks)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=_PAD,
            # connectN is 1-based; the normaliser re-bases and records the base.
            start_index=1,
            original_start_index=1,
            node_z=node_z,
            crs=None,
            variables={},
            global_attributes={
                "source": "Exodus II",
                "block_ids": block_ids,
                "element_types": element_types,
            },
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _coordinates(self, group, names: set[str]):
        """Read node coordinates, with ``coordz`` becoming ``node_z`` only when non-zero.

        A planar Exodus mesh may still carry an all-zero ``coordz`` column; it is dropped
        rather than kept as a meaningless ``node_z``, matching the gmsh and vtk readers
        which gate ``node_z`` on ``np.any(z != 0)``.
        """
        node_x = np.asarray(group.OpenMDArray(_COORD_X).ReadAsArray(), np.float64)
        node_y = np.asarray(group.OpenMDArray(_COORD_Y).ReadAsArray(), np.float64)
        node_z = None
        if _COORD_Z in names:
            coordz = np.asarray(group.OpenMDArray(_COORD_Z).ReadAsArray(), np.float64)
            if np.any(coordz != 0.0):
                node_z = coordz
        return node_x, node_y, node_z

    def _blocks(self, group, names: set[str]) -> list[tuple[np.ndarray, str | None]]:
        """Collect the element blocks ``connect1``, ``connect2``, ... in order.

        Raises:
            FileFormatNotSupportedError: If no ``connectN`` block is present, so there is
                no connectivity to build a mesh from.
        """
        blocks: list[tuple[np.ndarray, str | None]] = []
        index = 1
        while f"{_CONNECT_PREFIX}{index}" in names:
            array = group.OpenMDArray(f"{_CONNECT_PREFIX}{index}")
            connectivity = np.asarray(array.ReadAsArray(), np.int64)
            blocks.append((connectivity, _read_attribute(array, _ELEM_TYPE)))
            index += 1
        if not blocks:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable Exodus II mesh: no "
                f"{_CONNECT_PREFIX}1 element block to build connectivity from"
            )
        return blocks

    def _block_ids(self, group, names: set[str]) -> list[int]:
        """Return the block ids from ``eb_prop1``, or ``[]`` when the file omits them."""
        if _BLOCK_IDS not in names:
            return []
        values = np.asarray(group.OpenMDArray(_BLOCK_IDS).ReadAsArray())
        return [int(value) for value in values]
