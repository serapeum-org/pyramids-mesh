"""CGNS/HDF5 driver (CFD General Notation System).

A CGNS file is a plain HDF5 container holding a SIDS tree: a ``CGNSBase_t`` base, one or
more ``Zone_t`` zones, each with ``GridCoordinates`` (``CoordinateX``/``Y``/``Z``) and one
or more ``Elements_t`` sections. A section names an element type code and carries a flat,
1-based ``ElementConnectivity`` list plus an ``ElementRange``.

``.cgns`` is not auto-sniffed by GDAL, so the file is opened by forcing the HDF5
multidimensional driver, over the GDAL vendored in the ``pyramids-core`` wheel and activated
by ``import pyramids`` -- group traversal, attributes and typed array reads, with no
declared dependency and no ``h5py`` or ``CGNS`` library. See ``planning/design.md`` section
20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The 8-byte signature every HDF5 file opens with; the cheap half of the sniff.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# CGNS/HDF5 stores a node's value in a child dataset named literally " data" (a leading
# space): the node's own array, the section type code, the ranges and the coordinates all
# live under this name.
_NODE_DATA = " data"

# The SIDS node labels this reader keys on, read from a group's "label" attribute.
_LABEL = "label"
_BASE_LABEL = "CGNSBase_t"
_ZONE_LABEL = "Zone_t"
_ELEMENTS_LABEL = "Elements_t"

_GRID = "GridCoordinates"
_COORD_X = "CoordinateX"
_COORD_Y = "CoordinateY"
_COORD_Z = "CoordinateZ"
_CONNECTIVITY = "ElementConnectivity"
_RANGE = "ElementRange"

# SIDS element-type codes. Surface types map straight to a face; volumetric types are only
# used as faces when a zone carries no surface section (design section 21.1 -- 3D meshes are
# not modelled, so a cell's node list is read verbatim rather than expanded into its
# bounding faces, exactly as the Exodus reader does). MIXED and every 0D/1D code are skipped.
_SURFACE_CODES = {5: "TRI_3", 6: "TRI_6", 7: "QUAD_4", 8: "QUAD_8", 9: "QUAD_9"}
_VOLUME_CODES = {
    10: "TETRA_4",
    11: "TETRA_10",
    12: "PYRA_5",
    13: "PYRA_14",
    14: "PENTA_6",
    15: "PENTA_15",
    16: "PENTA_18",
    17: "HEXA_8",
    18: "HEXA_20",
    19: "HEXA_27",
}
_MIXED_CODE = 20


def _cgns_corner_count(name: str) -> int:
    """Return a CGNS surface element's corner-node count (3 for TRI_*, else 4).

    Higher-order elements (TRI_6, QUAD_8, QUAD_9) carry mid-side/mid-face nodes after
    their corners; only the corners form the face polygon.

    Args:
        name: The element-type name, e.g. ``"TRI_6"`` or ``"QUAD_8"``.

    Returns:
        int: 3 for a triangle family, 4 for a quad family.
    """
    return 3 if name.startswith("TRI") else 4


# Exodus and CGNS share the convention that connectivity is 1-based, so 0 is never a real
# node id and serves as the pad for the short rows of a mixed-width table; the shared
# normaliser rewrites it to the ``-1`` pyramids expects.
_PAD = 0


class CgnsDriver(BaseDriver):
    """Reader for the CGNS/HDF5 (CFD General Notation System) mesh format.

    The mesh comes from the first zone's node coordinates ``CoordinateX``/``CoordinateY``
    (and ``CoordinateZ`` for a 3D file, carried as ``node_z``) and its ``Elements_t``
    sections. Each section's flat, 1-based ``ElementConnectivity`` is reshaped to one row
    per element; the chosen sections' rows are concatenated into one face-node connectivity
    table, padded to the widest so a mesh mixing triangles and quads is rectangular. The ids
    drop from 1-based to 0-based and the ``1`` base is recorded for a writer.

    Surface sections (``TRI_*``/``QUAD_*``) are preferred: when a zone carries any, only
    those become faces, so a 3D volume mesh with boundary patches returns its surface skin.
    A zone carrying only volumetric sections maps those as faces verbatim (design section
    21.1) rather than returning an empty mesh.

    Documented limitations:
        - **Only the first base and first zone are read.** A multi-base or multi-zone file
          is read for its first ``CGNSBase_t`` / ``Zone_t`` only; selecting another is not
          wired through the read template yet.
        - **Surface sections are preferred; volumetric sections are a fallback.** A
          volumetric element (e.g. ``HEXA_8``) is read as a face verbatim, its node list
          taken as a polygon rather than expanded into bounding faces -- a genuine 3D mesh
          is not modelled yet (design section 21.1), and winding is judged in the ``xy``
          plane like any other face.
        - **``MIXED`` sections and 0D/1D codes are skipped.** A ``MIXED`` section interleaves
          a per-element type code with the node lists and has no constant width; it is
          dropped rather than misread. ``NODE``/``BAR`` sections are not surface faces and
          are skipped too.
        - **Connectivity is 1-based with mixed widths padded by ``0``** -- the normaliser
          rewrites the pad to ``-1`` fill and records the ``1`` base for a writer.
        - **Results and flow solutions are not read.** Only the mesh is built; a
          ``FlowSolution_t`` an analysis file may carry is out of scope for this reader.
        - There is no CRS in the arrays read here, so ``crs`` is ``None``.

    Example:
        Read a CGNS mesh (shown, not run as a doctest -- it needs a real ``.cgns`` file),
        reached through ``read_file``::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("grid.cgns")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "cgns"
    long_name = "CGNS/HDF5"
    suffixes = (".cgns",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is CGNS: HDF5 magic plus a ``CGNSBase_t`` group.

        The magic is checked first and cheaply; only then is the file opened -- forcing the
        HDF5 driver, because ``.cgns`` is not auto-sniffed by GDAL -- to confirm a root group
        labelled ``CGNSBase_t``. A MED file (whose root is ``ENS_MAA``) or another HDF5
        container carries no such group and is declined.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the magic matches and a ``CGNSBase_t`` base is present.
        """
        with open(path, "rb") as handle:
            if handle.read(len(_HDF5_MAGIC)) != _HDF5_MAGIC:
                return False
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                dataset = gdal.OpenEx(
                    path,
                    gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY,
                    allowed_drivers=["HDF5"],
                )
            except RuntimeError:
                return False
            try:
                root = dataset.GetRootGroup()
                return any(
                    _group_label(root.OpenGroup(name)) == _BASE_LABEL
                    for name in root.GetGroupNames()
                )
            finally:
                dataset.Close()

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read the first zone of a CGNS file into a :class:`ParsedMesh`.

        Args:
            path: The ``.cgns`` file.
            lazy: Unused -- the mesh is read eagerly and no results are read here; accepted
                to match the driver protocol.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` for a 3D file, and the element-type
                metadata.

        Raises:
            FileFormatNotSupportedError: If the file carries no base, no zone, or no usable
                element section.
        """
        return _CgnsReader(path).build()


def _group_label(group) -> str:
    """Return a group's SIDS ``label`` attribute, or ``""`` when it carries none."""
    for attribute in group.GetAttributes():
        if attribute.GetName() == _LABEL:
            try:
                return str(attribute.Read())
            except RuntimeError:
                return ""
    return ""


def _node_value(group) -> np.ndarray:
    """Return a SIDS node's own value array, held in its ``" data"`` child dataset."""
    return np.asarray(group.OpenMDArray(_NODE_DATA).ReadAsArray())


class _CgnsReader:
    """One CGNS file: read the first zone's coordinates and element sections."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file and assemble the mesh from the first zone.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` for a 3D file, and metadata.

        Raises:
            FileFormatNotSupportedError: If there is no base, no zone, or no usable section.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(
                self.path,
                gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY,
                allowed_drivers=["HDF5"],
            )
            try:
                zone = self._first_zone(dataset.GetRootGroup())
                node_x, node_y, node_z = self._coordinates(zone)
                faces, element_types = self._faces(zone)
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=_PAD,
            # ElementConnectivity is 1-based; the normaliser re-bases and records the base.
            start_index=1,
            original_start_index=1,
            node_z=node_z,
            crs=None,
            variables={},
            global_attributes={
                "source": "CGNS/HDF5",
                "element_types": element_types,
            },
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _first_zone(self, root):
        """Return the first zone of the first base, or refuse the file.

        Raises:
            FileFormatNotSupportedError: If no ``CGNSBase_t`` base or no ``Zone_t`` zone is
                present.
        """
        base = self._first_labelled(root, _BASE_LABEL)
        if base is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable CGNS file: no {_BASE_LABEL!r} base"
            )
        zone = self._first_labelled(base, _ZONE_LABEL)
        if zone is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} CGNS base carries no {_ZONE_LABEL!r} zone"
            )
        return zone

    @staticmethod
    def _first_labelled(group, label: str):
        """Return the first child group carrying ``label``, or ``None``."""
        for name in group.GetGroupNames():
            child = group.OpenGroup(name)
            if _group_label(child) == label:
                return child
        return None

    def _coordinates(self, zone):
        """Read node coordinates, with ``CoordinateZ`` becoming ``node_z`` when non-zero.

        A planar mesh may still carry an all-zero ``CoordinateZ``; it is dropped rather than
        kept as a meaningless ``node_z``, matching the Exodus reader.

        Raises:
            FileFormatNotSupportedError: If the zone carries no ``GridCoordinates``.
        """
        if _GRID not in zone.GetGroupNames():
            raise FileFormatNotSupportedError(
                f"{self.path!r} CGNS zone carries no {_GRID!r} group"
            )
        grid = zone.OpenGroup(_GRID)
        present = set(grid.GetGroupNames())
        node_x = _node_value(grid.OpenGroup(_COORD_X)).astype(np.float64)
        node_y = _node_value(grid.OpenGroup(_COORD_Y)).astype(np.float64)
        node_z = None
        if _COORD_Z in present:
            coordz = _node_value(grid.OpenGroup(_COORD_Z)).astype(np.float64)
            if np.any(coordz != 0.0):
                node_z = coordz
        return node_x, node_y, node_z

    def _faces(self, zone) -> tuple[np.ndarray, list[str]]:
        """Build the face table from the zone's element sections, preferring surface types.

        Returns:
            tuple[np.ndarray, list[str]]: The padded ``(n_element, max_width)`` 1-based
                connectivity and the per-element element-type names.

        Raises:
            FileFormatNotSupportedError: If the zone carries no usable surface or volumetric
                element section.
        """
        surface: list[tuple[np.ndarray, str]] = []
        volume: list[tuple[np.ndarray, str]] = []
        for name in zone.GetGroupNames():
            section = zone.OpenGroup(name)
            if _group_label(section) != _ELEMENTS_LABEL:
                continue
            self._sort_section(section, surface, volume)
        chosen = surface or volume
        if not chosen:
            raise FileFormatNotSupportedError(
                f"{self.path!r} CGNS zone carries no surface or volumetric element "
                f"section to build connectivity from (only MIXED/0D/1D sections, if any)"
            )
        return _pad_sections(chosen)

    def _sort_section(
        self,
        section,
        surface: list[tuple[np.ndarray, str]],
        volume: list[tuple[np.ndarray, str]],
    ) -> None:
        """Classify one ``Elements_t`` section into the surface or volume bucket.

        A ``MIXED`` section (interleaved per-element type codes, no constant width) and any
        0D/1D code are skipped. The element count comes from ``ElementRange`` so the per-
        element node width is derived rather than hardcoded.
        """
        type_code = int(_node_value(section)[0])
        if type_code == _MIXED_CODE:
            return
        name = _SURFACE_CODES.get(type_code) or _VOLUME_CODES.get(type_code)
        if name is None:
            return
        element_range = _node_value(section.OpenGroup(_RANGE))
        n_element = int(element_range[1] - element_range[0] + 1)
        if n_element <= 0:
            return
        connectivity = _node_value(section.OpenGroup(_CONNECTIVITY)).astype(np.int64)
        block = connectivity.reshape(n_element, connectivity.size // n_element)
        if type_code in _SURFACE_CODES:
            # Higher-order codes (TRI_6, QUAD_8/9) list corners first then mid-side/
            # mid-face nodes; keep only the corners so the face ring is not self-crossing.
            surface.append((block[:, : _cgns_corner_count(name)], name))
        else:
            volume.append((block, name))


def _pad_sections(
    sections: list[tuple[np.ndarray, str]],
) -> tuple[np.ndarray, list[str]]:
    """Concatenate element blocks into one table, padding short rows to the widest block.

    A block narrower than the widest (triangles among quads) is right-padded with ``_PAD``,
    the ``0`` that is never a valid 1-based CGNS id, so the stacked table is rectangular.
    The per-element type name is expanded to one entry per row, in section order.

    Args:
        sections: One ``(connectivity, type_name)`` pair per element section.

    Returns:
        tuple[np.ndarray, list[str]]: The padded ``(n_element, max_width)`` connectivity and
            the per-element element-type names.
    """
    max_width = max(block.shape[1] for block, _ in sections)
    padded: list[np.ndarray] = []
    element_types: list[str] = []
    for block, name in sections:
        rows = block.astype(np.int64)
        if rows.shape[1] < max_width:
            fill = np.full((rows.shape[0], max_width - rows.shape[1]), _PAD, np.int64)
            rows = np.concatenate([rows, fill], axis=1)
        padded.append(rows)
        element_types.extend([name] * block.shape[0])
    return np.concatenate(padded, axis=0), element_types
