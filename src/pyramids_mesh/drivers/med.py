"""MED-file driver (Salome / code_aster, HDF5-based).

A MED file is a plain HDF5 container. A mesh lives under ``ENS_MAA/<mesh>/<step>`` with its
node coordinates in ``NOE/COO`` -- a flat, coordinate-major block (all x, then all y, then
all z) -- and its elements in ``MAI/<geomcode>`` groups, one per element type (``TR3``,
``QU4``, ``TE4``, ``HE8``, ...), each with a flat, 1-based, coordinate-major ``NOD``
connectivity list.

``.med`` is not auto-sniffed by GDAL, so the file is opened by forcing the HDF5
multidimensional driver, over the GDAL vendored in the ``pyramids-core`` wheel and activated
by ``import pyramids`` -- group traversal, attributes and typed array reads, with no
declared dependency and no ``h5py`` or ``med`` library. See ``planning/design.md`` section
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

# The MED group tree: meshes under ENS_MAA, nodes under NOE/COO, elements under MAI.
_MESHES = "ENS_MAA"
_NODES = "NOE"
_COORDS = "COO"
_ELEMENTS = "MAI"
_CONNECTIVITY = "NOD"
# The element count a NOD/COO array carries, read from its "NBR" attribute: COO stores
# NBR nodes, a MAI/<code> NOD stores NBR elements. The spatial dimension / nodes-per-element
# is then the flat length divided by NBR.
_COUNT = "NBR"

# MED geometry codes this reader maps to faces. Surface types become faces directly;
# volumetric types are a fallback used only when a mesh carries no surface element (design
# section 21.1 -- 3D meshes are not modelled, so a cell's node list is read verbatim as a
# face rather than expanded, exactly as the Exodus reader does). Segment types (SE2/SE3) are
# 1D and skipped.
_SURFACE_CODES = frozenset({"TR3", "TR6", "TR7", "QU4", "QU8", "QU9"})
_VOLUME_CODES = frozenset(
    {"TE4", "T10", "TE10", "HE8", "H20", "HE20", "PY5", "P13", "PE6", "P15"}
)

# MED connectivity is 1-based, so 0 is never a real node id and serves as the pad for the
# short rows of a mixed-width (triangles among quads) table; the shared normaliser rewrites
# it to the ``-1`` pyramids expects.
_PAD = 0


class MedDriver(BaseDriver):
    """Reader for the MED-file (Salome / code_aster) finite-element mesh format.

    The mesh comes from the first mesh's node coordinates ``NOE/COO`` and its ``MAI``
    element groups. ``COO`` is coordinate-major (all x, then all y, then all z), so the
    node coordinates are read as its leading columns; ``CoordinateZ`` becomes ``node_z``
    only when non-zero. Each ``MAI/<geomcode>`` group's flat, 1-based ``NOD`` is reshaped
    (coordinate-major) to one row per element; the chosen groups' rows are concatenated into
    one face-node connectivity table, padded to the widest so a mesh mixing triangles and
    quads is rectangular. The ids drop from 1-based to 0-based and the ``1`` base is recorded
    for a writer.

    Surface groups (``TR*``/``QU*``) are preferred: when a mesh carries any, only those
    become faces, so a 3D volume mesh with a surface skin returns the surface. A mesh
    carrying only volumetric groups maps those as faces verbatim (design section 21.1)
    rather than returning an empty mesh.

    Documented limitations:
        - **Only the first mesh (and its first step) is read.** A multi-mesh file is read
          for its first ``ENS_MAA`` entry only; selecting another is not wired through the
          read template yet.
        - **Surface groups are preferred; volumetric groups are a fallback.** A volumetric
          element (e.g. ``TE4``/``HE8``) is read as a face verbatim, its node list taken as a
          polygon rather than expanded into bounding faces -- a genuine 3D mesh is not
          modelled yet (design section 21.1), and winding is judged in the ``xy`` plane like
          any other face.
        - **Segment (``SE2``/``SE3``) groups are skipped.** They are 1D element types, not
          surface faces; boundary extraction is pyramids' job (ADR-0005), not this reader's.
        - **Connectivity is 1-based with mixed widths padded by ``0``** -- the normaliser
          rewrites the pad to ``-1`` fill and records the ``1`` base for a writer.
        - **Families, fields and results are not read.** The ``FAM``/``NUM`` arrays and any
          ``CHA`` field groups a MED file carries are out of scope for this reader.
        - There is no CRS in the arrays read here, so ``crs`` is ``None``.

    Example:
        Read a MED mesh (shown, not run as a doctest -- it needs a real ``.med`` file),
        reached through ``read_file``::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("cylinder.med")
            mesh.mesh.n_node, mesh.mesh.n_face
    """

    name = "med"
    long_name = "MED-file (Salome / code_aster)"
    suffixes = (".med",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is MED: HDF5 magic plus an ``ENS_MAA`` group.

        The magic is checked first and cheaply; only then is the file opened -- forcing the
        HDF5 driver, because ``.med`` is not auto-sniffed by GDAL -- to confirm the root
        ``ENS_MAA`` meshes group. A CGNS file (whose root carries ``CGNSBase_t``) or another
        HDF5 container carries no such group and is declined.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the magic matches and an ``ENS_MAA`` group is present.
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
                return _MESHES in root.GetGroupNames()
            finally:
                dataset.Close()

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read the first mesh of a MED file into a :class:`ParsedMesh`.

        Args:
            path: The ``.med`` file.
            lazy: Unused -- the mesh is read eagerly and no results are read here; accepted
                to match the driver protocol.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` for a 3D file, and the element-type
                metadata.

        Raises:
            FileFormatNotSupportedError: If the file carries no mesh or no usable element
                group.
        """
        return _MedReader(path).build()


def _med_corner_count(code: str) -> int:
    """Return a MED surface element's corner-node count (3 for TR*, else 4).

    Higher-order elements (TR6/TR7, QU8/QU9) carry mid-side nodes after their corners;
    only the corners form the face polygon.

    Args:
        code: The MED geometry code, e.g. ``"TR6"`` or ``"QU8"``.

    Returns:
        int: 3 for a triangle family, 4 for a quad family.
    """
    return 3 if code.startswith("TR") else 4


def _array_count(array) -> int:
    """Return an array's ``NBR`` attribute -- its node or element count.

    Raises:
        FileFormatNotSupportedError: If the array carries no ``NBR`` attribute, so its row
            count cannot be known.
    """
    for attribute in array.GetAttributes():
        if attribute.GetName() == _COUNT:
            return int(attribute.Read())
    raise FileFormatNotSupportedError(
        f"a MED array carries no {_COUNT!r} attribute; its element count is unknown"
    )


class _MedReader:
    """One MED file: read the first mesh's coordinates and element groups."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Open the file and assemble the mesh from the first ``ENS_MAA`` entry.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` for a 3D file, and metadata.

        Raises:
            FileFormatNotSupportedError: If there is no mesh or no usable element group.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(
                self.path,
                gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY,
                allowed_drivers=["HDF5"],
            )
            try:
                step = self._first_step(dataset.GetRootGroup())
                node_x, node_y, node_z = self._coordinates(step)
                faces, element_types = self._faces(step)
            finally:
                dataset.Close()
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=_PAD,
            # NOD is 1-based; the normaliser re-bases and records the base.
            start_index=1,
            original_start_index=1,
            node_z=node_z,
            crs=None,
            variables={},
            global_attributes={
                "source": "MED-file",
                "element_types": element_types,
            },
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _first_step(self, root):
        """Return the first mesh's step group (the one carrying ``NOE``), or refuse.

        A MED mesh nests a computation step between the mesh name and its data; the step is
        the subgroup carrying the ``NOE`` node group.

        Raises:
            FileFormatNotSupportedError: If there is no ``ENS_MAA`` meshes group, no mesh
                under it, or no step carrying node coordinates.
        """
        if _MESHES not in root.GetGroupNames():
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable MED file: no {_MESHES!r} group"
            )
        meshes = root.OpenGroup(_MESHES)
        mesh_names = meshes.GetGroupNames()
        if not mesh_names:
            raise FileFormatNotSupportedError(
                f"{self.path!r} {_MESHES!r} group carries no mesh"
            )
        mesh = meshes.OpenGroup(mesh_names[0])
        for step_name in mesh.GetGroupNames():
            step = mesh.OpenGroup(step_name)
            if _NODES in step.GetGroupNames():
                return step
        raise FileFormatNotSupportedError(
            f"{self.path!r} MED mesh {mesh_names[0]!r} carries no step with a {_NODES!r} "
            f"node group"
        )

    def _coordinates(self, step):
        """Read node coordinates from the coordinate-major ``NOE/COO`` block.

        ``COO`` holds all x, then all y, then all z, so reshaping ``(n_node, dim)`` in
        column-major order recovers the per-node coordinates. ``z`` becomes ``node_z`` only
        for a 3D file whose z is not all zero, matching the Exodus reader.
        """
        coo = step.OpenGroup(_NODES).OpenMDArray(_COORDS)
        n_node = _array_count(coo)
        flat = np.asarray(coo.ReadAsArray(), np.float64)
        dim = flat.size // n_node
        coords = flat.reshape(n_node, dim, order="F")
        node_x = coords[:, 0]
        node_y = coords[:, 1] if dim >= 2 else np.zeros(n_node)
        node_z = None
        if dim >= 3 and np.any(coords[:, 2] != 0.0):
            node_z = coords[:, 2]
        return node_x, node_y, node_z

    def _faces(self, step) -> tuple[np.ndarray, list[str]]:
        """Build the face table from the ``MAI`` element groups, preferring surface types.

        Returns:
            tuple[np.ndarray, list[str]]: The padded ``(n_element, max_width)`` 1-based
                connectivity and the per-element geometry-code names.

        Raises:
            FileFormatNotSupportedError: If there is no ``MAI`` group, or no surface or
                volumetric element group under it.
        """
        if _ELEMENTS not in step.GetGroupNames():
            raise FileFormatNotSupportedError(
                f"{self.path!r} MED mesh carries no {_ELEMENTS!r} element group"
            )
        mai = step.OpenGroup(_ELEMENTS)
        surface: list[tuple[np.ndarray, str]] = []
        volume: list[tuple[np.ndarray, str]] = []
        for code in mai.GetGroupNames():
            self._sort_group(mai, code, surface, volume)
        chosen = surface or volume
        if not chosen:
            raise FileFormatNotSupportedError(
                f"{self.path!r} MED mesh carries no surface or volumetric element group "
                f"(only segment groups, if any)"
            )
        return _pad_groups(chosen)

    def _sort_group(
        self,
        mai,
        code: str,
        surface: list[tuple[np.ndarray, str]],
        volume: list[tuple[np.ndarray, str]],
    ) -> None:
        """Classify one ``MAI/<code>`` group into the surface or volume bucket.

        Segment (``SE*``) and any unrecognised code are skipped. The element count comes
        from the ``NOD`` array's ``NBR`` attribute, so the per-element node width is derived
        and the flat, coordinate-major list is reshaped to one row per element.
        """
        if code not in _SURFACE_CODES and code not in _VOLUME_CODES:
            return
        nod = mai.OpenGroup(code).OpenMDArray(_CONNECTIVITY)
        n_element = _array_count(nod)
        if n_element <= 0:
            return
        flat = np.asarray(nod.ReadAsArray(), np.int64)
        block = flat.reshape(n_element, flat.size // n_element, order="F")
        if code in _SURFACE_CODES:
            # Higher-order codes (TR6/TR7, QU8/QU9) list corners first then mid-side nodes;
            # keep only the corners so the face ring is not self-crossing.
            surface.append((block[:, : _med_corner_count(code)], code))
        else:
            volume.append((block, code))


def _pad_groups(
    groups: list[tuple[np.ndarray, str]],
) -> tuple[np.ndarray, list[str]]:
    """Concatenate element blocks into one table, padding short rows to the widest block.

    A block narrower than the widest (triangles among quads) is right-padded with ``_PAD``,
    the ``0`` that is never a valid 1-based MED id, so the stacked table is rectangular. The
    per-element geometry-code name is expanded to one entry per row, in group order.

    Args:
        groups: One ``(connectivity, code)`` pair per element group.

    Returns:
        tuple[np.ndarray, list[str]]: The padded ``(n_element, max_width)`` connectivity and
            the per-element geometry-code names.
    """
    max_width = max(block.shape[1] for block, _ in groups)
    padded: list[np.ndarray] = []
    element_types: list[str] = []
    for block, code in groups:
        rows = block.astype(np.int64)
        if rows.shape[1] < max_width:
            fill = np.full((rows.shape[0], max_width - rows.shape[1]), _PAD, np.int64)
            rows = np.concatenate([rows, fill], axis=1)
        padded.append(rows)
        element_types.extend([code] * block.shape[0])
    return np.concatenate(padded, axis=0), element_types
