"""XDMF reader (task D7).

XDMF is an XML index over an HDF5 heavy-data payload: the ``.xdmf``/``.xmf`` file names the
mesh topology and geometry and, for a time series, the per-step result slabs, each pointing
at an HDF5 dataset (``file.h5:/path``). So two parsers cooperate -- stdlib
``xml.etree.ElementTree`` for the index, and the GDAL multidimensional API (over the GDAL
vendored in the ``pyramids-gis`` wheel, activated by ``import pyramids``) for the HDF5 arrays,
with no declared dependency and no ``h5py``. See ``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import xml.etree.ElementTree as ET

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable

# The cheap half of the sniff: an XDMF index is an XML document whose root is ``Xdmf``.
_ROOT_TAG = "Xdmf"

# 2D surface topologies this reader builds. Volumetric (3D) topologies carry ragged
# vertical structure this package does not model yet (design section 21.1), and the mixed /
# higher-order / polyline types need bespoke handling, so all of those are refused rather
# than read as something they are not.
_SURFACE_TOPOLOGIES = frozenset({"Triangle", "Quadrilateral"})

# XDMF ``Center`` values map onto pyramids mesh locations.
_CENTER_TO_LOCATION = {"Node": "node", "Cell": "face", "Edge": "edge"}


class XdmfDriver(BaseDriver):
    """Reader for the XDMF (XML + HDF5) mesh-and-results format.

    The ``.xdmf`` index names a shared ``Grid`` holding the mesh ``Geometry`` (node
    coordinates) and ``Topology`` (connectivity), plus -- for a time series -- a temporal
    ``Grid`` collection whose per-step child grids list ``Attribute`` results. Every heavy
    array lives in HDF5 and is referenced as ``file.h5:/dataset``; the driver reads the mesh
    eagerly and leaves each result variable behind a reopen-per-read loader.

    Documented limitations:
        - **Only 2D surface topologies are read** (``Triangle``, ``Quadrilateral``). A
          volumetric topology (``Tetrahedron``/``Hexahedron``/...) is a 3D mesh whose
          vertical structure is not modelled yet (design section 21.1); it, and the
          ``Mixed``/higher-order/``Polyline`` types, are refused rather than flattened.
        - **Vector attributes are skipped.** The 2D scalar case ships first; a ``Vector``
          attribute (e.g. velocity components) is left out rather than mislabelled as a
          scalar, exactly as the HDF5-results sibling driver does.
        - **Values are lazy.** Each result variable's loader reopens the HDF5 file per read
          and stacks the per-step slabs, closing over paths, never an open handle (design
          section 7); shapes come from the XML without forcing a read.
        - ``GeometryType="XYZ"`` contributes a ``node_z`` variable; ``XY`` leaves it unset.
        - There is no CRS in the format, so ``crs`` is ``None``.

    Example:
        Read an XDMF time series (shown, not run as a doctest -- it needs a real ``.xdmf``
        plus its ``.h5`` payload)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("out.xdmf")
            phi = mesh.get("phi")          # node-located, over time
            a = mesh.get("a")              # cell (face) located, over time
    """

    name = "xdmf"
    long_name = "XDMF"
    suffixes = (".xdmf", ".xmf")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is an XDMF index: an XML document rooted at ``Xdmf``.

        Reads only the head of the file and looks for the ``<Xdmf`` opening, so a non-XML or
        foreign file is declined cheaply without parsing the whole document.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when an ``<Xdmf`` root tag appears near the top of the file.
        """
        with open(path, "rb") as handle:
            head = handle.read(4096)
        return b"<Xdmf" in head

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an XDMF index (and its HDF5 payload) into a :class:`ParsedMesh`.

        Args:
            path: The ``.xdmf``/``.xmf`` index file.
            lazy: When ``True``, leave each result variable behind a loader.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` (for ``XYZ`` geometry) and the per-step
                result variables.

        Raises:
            FileFormatNotSupportedError: If the index holds no mesh grid, or the topology is
                not a 2D surface type.
        """
        return _XdmfReader(path, lazy=lazy).build()


class _XdmfLoader:
    """A lazy loader for one XDMF variable's values.

    Closes over the HDF5 file/dataset references, never an open handle (design section 7):
    reading it reopens each referenced file, reads the per-step slabs and stacks them over
    time. A non-temporal variable has a single reference and is returned unstacked.
    """

    def __init__(self, references: tuple[tuple[str, str], ...], *, stack: bool) -> None:
        self.references = references
        self.stack = stack

    def __call__(self) -> np.ndarray:
        """Reopen the referenced HDF5 file(s), read each slab, and stack over time."""
        opened: dict[str, gdal.Dataset] = {}
        with gdal.ExceptionMgr(useExceptions=True):
            try:
                arrays = [
                    _read_md_array(self._root(opened, h5_path), name)
                    for h5_path, name in self.references
                ]
            finally:
                for dataset in opened.values():
                    dataset.Close()
        return np.stack(arrays) if self.stack else arrays[0]

    @staticmethod
    def _root(opened: dict, h5_path: str):
        """Open ``h5_path`` once per call and return its root group."""
        if h5_path not in opened:
            opened[h5_path] = gdal.OpenEx(
                h5_path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY
            )
        return opened[h5_path].GetRootGroup()


def _read_md_array(root, name: str) -> np.ndarray:
    """Read one HDF5 MD-array by its (possibly group-nested) name from a root group."""
    group = root
    parts = [part for part in name.split("/") if part]
    for part in parts[:-1]:
        group = group.OpenGroup(part)
    return np.asarray(group.OpenMDArray(parts[-1]).ReadAsArray())


def _resolve_reference(text: str, base_dir: str) -> tuple[str, str]:
    """Resolve a ``DataItem`` reference ``file.h5:/dataset`` to an absolute path and name.

    Args:
        text: The ``DataItem`` text, e.g. ``"out.h5:/data0"``.
        base_dir: The directory the ``.xdmf`` file sits in, that relative paths resolve
            against.

    Returns:
        tuple[str, str]: The absolute HDF5 path and the dataset name within it.
    """
    reference = text.strip()
    file_part, separator, dataset_part = reference.partition(":")
    if not separator:
        # No "file:" prefix -- the whole reference is a dataset path in a sibling .h5 is not
        # knowable, so treat the text as a path with an empty dataset, which the caller's
        # read then reports as a clear GDAL error rather than a silent wrong array.
        return os.path.join(base_dir, reference), ""
    return os.path.join(base_dir, file_part), dataset_part.lstrip("/")


class _XdmfReader:
    """One XDMF index: parse the XML, read the mesh from HDF5, map the result slabs."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.base_dir = os.path.dirname(os.path.abspath(path))
        self.lazy = lazy

    def build(self) -> ParsedMesh:
        """Parse the index and assemble the mesh and its per-step variables.

        Returns:
            ParsedMesh: The 2D surface mesh with its result variables.

        Raises:
            FileFormatNotSupportedError: On a missing mesh grid or an unsupported topology.
        """
        root = ET.parse(self.path).getroot()
        domain = root.find("Domain")
        grids = domain.findall("Grid") if domain is not None else []
        mesh_grid = self._mesh_grid(grids)
        # Validate the topology type from the XML first, so an unsupported (e.g. 3D) mesh is
        # refused before any HDF5 payload is opened.
        self._check_topology_type(mesh_grid)
        node_x, node_y, node_z = self._geometry(mesh_grid)
        faces = self._topology(mesh_grid)
        variables = self._variables(grids, n_node=len(node_x), n_face=len(faces))
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=0,
            original_start_index=0,  # XDMF connectivity is 0-based
            node_z=node_z,
            crs=None,
            variables=variables,
            global_attributes={"source": "XDMF"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _mesh_grid(self, grids: list) -> ET.Element:
        """Return the grid carrying the mesh (a ``Geometry`` and a ``Topology``)."""
        for grid in grids:
            if grid.find("Geometry") is not None and grid.find("Topology") is not None:
                return grid
        raise FileFormatNotSupportedError(
            f"{self.path!r} is not a readable XDMF mesh: no grid carries both a Geometry "
            f"and a Topology"
        )

    def _geometry(self, mesh_grid: ET.Element):
        """Read node coordinates (and ``node_z`` for ``XYZ``) from the mesh grid."""
        geometry = mesh_grid.find("Geometry")
        geometry_type = geometry.get("GeometryType", "XY")
        coords = self._read(geometry.find("DataItem"))
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, coords.shape[-1])
        node_x = coords[:, 0].copy()
        node_y = coords[:, 1].copy()
        node_z = None
        if geometry_type.upper().replace("_", "") == "XYZ" and coords.shape[1] >= 3:
            node_z = coords[:, 2].copy()
        return node_x, node_y, node_z

    def _check_topology_type(self, mesh_grid: ET.Element) -> None:
        """Refuse a topology that is not a 2D surface type, before any HDF5 is read.

        Raises:
            FileFormatNotSupportedError: On a volumetric, mixed or higher-order topology.
        """
        topology = mesh_grid.find("Topology")
        topology_type = topology.get("TopologyType") or topology.get("Type") or ""
        if topology_type not in _SURFACE_TOPOLOGIES:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has TopologyType {topology_type!r}; only 2D surface "
                f"topologies {sorted(_SURFACE_TOPOLOGIES)} are read -- a volumetric, mixed "
                f"or higher-order topology is refused rather than flattened"
            )

    def _topology(self, mesh_grid: ET.Element) -> np.ndarray:
        """Read the connectivity array (its type was already validated)."""
        faces = self._read(mesh_grid.find("Topology").find("DataItem"))
        return np.asarray(faces, dtype=np.int64).reshape(len(faces), -1)

    def _variables(
        self, grids: list, *, n_node: int, n_face: int
    ) -> dict[str, ParsedVariable]:
        """Map the temporal collection's ``Attribute``s to lazy result variables.

        Each scalar attribute becomes a variable over the ``Time`` axis; its per-step HDF5
        slabs are stacked lazily. Vector attributes are skipped (2D scalar ships first).
        """
        collection = next((g for g in grids if g.get("GridType") == "Collection"), None)
        if collection is None:
            return {}
        steps = collection.findall("Grid")
        times = np.array(
            [float(step.find("Time").get("Value")) for step in steps], dtype=np.float64
        )
        collected: dict[str, dict] = {}
        for step in steps:
            for attribute in step.findall("Attribute"):
                name = attribute.get("Name")
                entry = collected.setdefault(
                    name,
                    {
                        "type": attribute.get("AttributeType", "Scalar"),
                        "center": attribute.get("Center", "Node"),
                        "items": [],
                    },
                )
                entry["items"].append(attribute.find("DataItem"))
        variables: dict[str, ParsedVariable] = {}
        for name, entry in collected.items():
            variable = self._build_variable(name, entry, times, n_node, n_face)
            if variable is not None:
                variables[name] = variable
        return variables

    def _build_variable(
        self, name: str, entry: dict, times: np.ndarray, n_node: int, n_face: int
    ) -> ParsedVariable | None:
        """Build one lazy result variable, or ``None`` to skip a vector attribute."""
        if entry["type"].upper() == "VECTOR":
            return None
        location = _CENTER_TO_LOCATION.get(entry["center"], "node")
        items = entry["items"]
        element_count = self._element_count(items[0])
        references = tuple(
            _resolve_reference(item.text, self.base_dir) for item in items
        )
        loader = _XdmfLoader(references, stack=True)
        return ParsedVariable(
            location=location,
            shape=(len(times), element_count),
            loader=loader if self.lazy else None,
            values=None if self.lazy else loader(),
            time_values=times,
        )

    @staticmethod
    def _element_count(data_item: ET.Element) -> int:
        """Return the per-step element count from the ``DataItem`` ``Dimensions`` first axis."""
        dimensions = data_item.get("Dimensions", "").split()
        return int(dimensions[0]) if dimensions else 0

    def _read(self, data_item: ET.Element) -> np.ndarray:
        """Read a single HDF5 array a ``DataItem`` references, eagerly (mesh geometry)."""
        h5_path, dataset = _resolve_reference(data_item.text, self.base_dir)
        with gdal.ExceptionMgr(useExceptions=True):
            handle = gdal.OpenEx(h5_path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                return _read_md_array(handle.GetRootGroup(), dataset)
            finally:
                handle.Close()
