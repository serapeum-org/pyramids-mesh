"""XMDF driver (task D6).

An HDF5 results container: results-only, lazy by nature, and carrying the per-step
wet/dry ``Active`` mask. A driver declares ``needs_mesh`` and attaches to a mesh the
caller already holds.

HDF5 is read through the GDAL multidimensional API, over the GDAL vendored in the
``pyramids-gis`` wheel and activated by ``import pyramids`` -- group traversal, attributes
and typed array reads, with no declared dependency and no ``h5py``. See
``planning/design.md`` section 20.3.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import numpy as np

from osgeo import gdal

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION, MeshVariable, UgridDataset

from pyramids_mesh._base import BaseDriver, ParsedVariable

# The 8-byte signature every HDF5 file opens with; the cheap half of the sniff.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# The root array naming the format; its value is "Xmdf". Read with Read() -- a string
# MD-array cannot go through ReadAsArray in the numpy bindings.
_FILE_TYPE = "File Type"
_FILE_TYPE_VALUE = "Xmdf"

# A dataset group is marked by this attribute; its value starts with "DATASET".
_GROUPTYPE = "Grouptype"
_DATASET_PREFIX = "DATASET"
_VECTOR_MARKER = "VECTOR"

# The per-group arrays.
_VALUES = "Values"
_TIMES = "Times"
_ACTIVE = "Active"
_MINS = "Mins"
_MAXS = "Maxs"

# Group names that hold derived summaries rather than a time series: a dataset nested
# under one of these is a single-step maximum/final snapshot, not a timestep, and is
# skipped so it cannot masquerade as extra time levels.
_SUMMARY_CONTAINERS = frozenset({"Final", "Maximums", "Maximum", "Minimums"})


class XmdfDriver(BaseDriver):
    """Reader for the XMDF HDF5 results format.

    XMDF carries values and no topology, so the driver declares ``needs_mesh`` and
    implements ``read_data`` rather than ``read``; the values attach to a mesh the caller
    already holds, through ``UgridDataset.read_file("domain.2dm").attach_data("r.xmdf")``.

    The file is an HDF5 tree. Each dataset is a group carrying a ``Grouptype`` attribute
    beginning ``DATASET``, with ``Values`` shaped ``(n_time, n_element)``, a ``Times``
    coordinate, an ``Active`` wet/dry mask and precomputed ``Mins``/``Maxs``. The tree is
    walked recursively, so the driver is not tied to one container layout.

    Documented limitations:
        - **The element-count check is the only thing tying the results to the mesh.** A
          dataset's ``Values`` second axis is matched against the mesh's node or face
          count to place it; a count matching neither is refused, because nothing in the
          file identifies the mesh it belongs to.
        - **Summary datasets are skipped, not read as timesteps.** A dataset nested under
          a ``Final``/``Maximums`` container is a single-step snapshot; reading it as a
          time level would fabricate data, so it is dropped.
        - **The active mask is a standalone face-located variable** ``<name>_active``,
          never folded into the values as NaN (design section 21.2). It is not carried
          through ``ParsedVariable.active``: the mask counts faces while a node-located
          dataset counts nodes, so the sibling cannot take the parent's location or shape.
          It is attached only when its width matches the mesh's face count.
        - **Values are lazy.** The loader reopens the file per read and seeks to the
          dataset's array, so it closes over a path, never an open handle (design section
          7); the shape is read from the header without loading the values.
        - ``Mins``/``Maxs`` are carried as per-timestep ``minimum``/``maximum`` attributes.
        - **A layered 3D dataset is refused.** A ``Values`` array with more than two axes
          is a 3D result whose ragged vertical levels this package does not model yet
          (design section 21.1); it is refused rather than flattened.
        - A vector dataset (``Grouptype`` containing ``VECTOR``) is refused for now --
          the 2D scalar case ships first.

    Example:
        Attach an XMDF results file to the mesh it belongs to. The example is shown
        rather than executed as a doctest because XMDF needs a real HDF5 file, which
        cannot be built inline::

            from pyramids.netcdf.ugrid import UgridDataset

            dataset = UgridDataset.read_file("domain.2dm")
            attached = dataset.attach_data("results.xmdf")
            depth = attached.get("Minimum dt")       # node-located, (n_time, n_node)
            depth.attributes["time_values"]           # the Times coordinate
            attached.get("Minimum dt_active")         # per-face wet/dry mask sibling
    """

    name = "xmdf"
    long_name = "XMDF HDF5 results"
    suffixes = (".xmdf", ".h5")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"needs_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is XMDF: HDF5 magic plus a ``File Type`` of ``Xmdf``.

        The magic is checked first and cheaply; only then is the file opened to read the
        root ``File Type`` array, so a non-HDF5 file never reaches GDAL. ``.h5`` is a
        generic HDF5 suffix shared with other formats, which is why the ``File Type``
        value -- not just the suffix -- decides.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the magic matches and ``File Type`` reads ``Xmdf``.
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
                if _FILE_TYPE not in root.GetMDArrayNames():
                    return False
                value = root.OpenMDArray(_FILE_TYPE).Read()
            finally:
                dataset.Close()
        return bool(value) and str(value[0]) == _FILE_TYPE_VALUE

    def read(
        self, path: str, *, mesh_name: str | None = None, lazy: bool = True
    ) -> UgridDataset:
        """Refuse the mesh-reading entry point, naming the one to use instead.

        Args:
            path: The results file, named in the message.
            mesh_name: Unused.
            lazy: Unused.

        Returns:
            UgridDataset: Never -- the signature matches the driver protocol so the
                refusal is a clear message rather than a ``TypeError``.

        Raises:
            ValueError: Always. XMDF carries values and no topology.
        """
        raise ValueError(
            f"{self.name} carries values but no mesh, so it has no read(); attach it to "
            f"a mesh you already have with "
            f"UgridDataset.read_file(mesh_path).attach_data({path!r})"
        )

    def read_data(
        self, path: str, dataset: UgridDataset, *, lazy: bool = True
    ) -> dict[str, MeshVariable]:
        """Read an XMDF file's datasets as variables for a mesh the caller supplies.

        Args:
            path: The XMDF file to read.
            dataset: The mesh the values attach to, for the element-count check.
            lazy: When ``True``, leave each dataset's values behind a loader.

        Returns:
            dict[str, MeshVariable]: One variable per dataset, plus a face-located
                ``<name>_active`` sibling for each dataset carrying an active mask.

        Raises:
            FileFormatNotSupportedError: If a dataset is 3D or vector, or no dataset's
                element count matches the mesh.
            ValueError: If a dataset's element count matches neither the node nor the
                face count of ``dataset``.
        """
        n_node = int(dataset.mesh.n_node)
        n_face = int(dataset.mesh.n_face)
        reader = _XmdfReader(path, n_node=n_node, n_face=n_face, lazy=lazy)
        parsed = reader.read()
        return self._build_data_variables(parsed, dataset.mesh_name)


class _XmdfLoader:
    """A lazy loader for one XMDF dataset's values.

    Closes over the path and the group path, never an open handle (design section 7):
    reading it reopens the file, walks to the dataset's ``Values`` array, and closes.
    """

    def __init__(self, path: str, group_path: tuple[str, ...]) -> None:
        self.path = path
        self.group_path = group_path

    def __call__(self) -> np.ndarray:
        """Reopen the file and read this dataset's ``Values`` array."""
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                group = dataset.GetRootGroup()
                for name in self.group_path:
                    group = group.OpenGroup(name)
                return np.asarray(group.OpenMDArray(_VALUES).ReadAsArray())
            finally:
                dataset.Close()


class _XmdfReader:
    """One XMDF file: walk the HDF5 tree and build parsed variables for its datasets."""

    def __init__(self, path: str, *, n_node: int, n_face: int, lazy: bool) -> None:
        self.path = path
        self.n_node = n_node
        self.n_face = n_face
        self.lazy = lazy

    def read(self) -> dict[str, ParsedVariable]:
        """Open the file, collect its dataset groups, and parse each into a variable.

        Returns:
            dict[str, ParsedVariable]: Keyed by dataset name, with an ``<name>_active``
                sibling for each dataset whose active mask matches the face count.

        Raises:
            FileFormatNotSupportedError: If the file holds no readable dataset, or a
                dataset is 3D or vector.
            ValueError: On an element-count mismatch.
        """
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.OpenEx(self.path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
            try:
                found: list[tuple[str, tuple[str, ...]]] = []
                self._collect(dataset.GetRootGroup(), (), found)
                if not found:
                    raise FileFormatNotSupportedError(
                        f"{self.path!r} holds no XMDF dataset group (a group with a "
                        f"{_GROUPTYPE!r} of {_DATASET_PREFIX!r} and a {_VALUES!r} array)"
                    )
                variables: dict[str, ParsedVariable] = {}
                for name, group_path in found:
                    self._add_dataset(dataset, name, group_path, variables)
            finally:
                dataset.Close()
        return variables

    def _collect(self, group, path: tuple[str, ...], found: list) -> None:
        """Recursively gather dataset groups, skipping summary containers.

        A group is a dataset when it carries a ``Grouptype`` attribute beginning
        ``DATASET`` and a ``Values`` array. A group whose own name is a summary container
        (``Final``/``Maximums``) is not descended into, so its single-step snapshots never
        reach the variable list.
        """
        if self._is_dataset(group):
            found.append((path[-1] if path else self.name, path))
            return
        for child_name in group.GetGroupNames():
            if child_name in _SUMMARY_CONTAINERS:
                continue
            self._collect(group.OpenGroup(child_name), path + (child_name,), found)

    @staticmethod
    def _is_dataset(group) -> bool:
        """Whether a group is an XMDF dataset (DATASET Grouptype plus a Values array)."""
        attributes = {a.GetName() for a in group.GetAttributes()}
        if _GROUPTYPE not in attributes or _VALUES not in group.GetMDArrayNames():
            return False
        # An attribute's Read() returns the scalar string directly (an MD-array's Read()
        # returns a one-element list -- the two are not the same, and treating this as a
        # list took its first character instead of the word).
        grouptype = str(group.GetAttribute(_GROUPTYPE).Read())
        return grouptype.startswith(_DATASET_PREFIX)

    def _add_dataset(
        self, dataset, name: str, group_path: tuple[str, ...], variables: dict
    ) -> None:
        """Parse one dataset group into a variable (and its active sibling)."""
        group = dataset.GetRootGroup()
        for part in group_path:
            group = group.OpenGroup(part)
        self._guard_scalar_2d(group, name)
        values_array = group.OpenMDArray(_VALUES)
        shape = tuple(d.GetSize() for d in values_array.GetDimensions())
        location = self._location(shape, name)
        times = np.asarray(group.OpenMDArray(_TIMES).ReadAsArray(), dtype=np.float64)
        attributes = self._range_attributes(group)
        loader = _XmdfLoader(self.path, group_path)
        variables[name] = ParsedVariable(
            location=location,
            shape=shape,
            loader=loader if self.lazy else None,
            values=None if self.lazy else loader(),
            time_values=times,
            attributes=attributes,
        )
        self._maybe_active(group, name, times, variables)

    def _guard_scalar_2d(self, group, name: str) -> None:
        """Refuse a vector or layered-3D dataset -- only 2D scalar ships for now.

        Raises:
            FileFormatNotSupportedError: On a ``VECTOR`` Grouptype, or a ``Values`` array
                with more than two axes (a 3D result with ragged vertical levels).
        """
        grouptype = str(group.GetAttribute(_GROUPTYPE).Read())
        if _VECTOR_MARKER in grouptype.upper():
            raise FileFormatNotSupportedError(
                f"{self.path!r} dataset {name!r} is a vector dataset ({grouptype!r}); "
                f"only scalar XMDF datasets are read so far"
            )
        rank = len(group.OpenMDArray(_VALUES).GetDimensions())
        if rank > 2:
            raise FileFormatNotSupportedError(
                f"{self.path!r} dataset {name!r} has rank-{rank} values, a layered 3D "
                f"result; its vertical levels are not modelled yet, so it is refused "
                f"rather than flattened"
            )

    def _location(self, shape: tuple[int, ...], name: str) -> str:
        """Resolve a dataset's location from its element count against the mesh.

        Raises:
            ValueError: If the element count matches neither the node nor the face count.
        """
        n_element = shape[1] if len(shape) > 1 else shape[0]
        if n_element == self.n_node:
            return "node"
        if n_element == self.n_face:
            return "face"
        raise ValueError(
            f"{self.path!r} dataset {name!r} has {n_element} values per step, which is "
            f"neither the mesh's {self.n_node} nodes nor its {self.n_face} faces; the "
            f"results do not belong to this mesh"
        )

    def _range_attributes(self, group) -> dict:
        """Carry the precomputed ``Mins``/``Maxs`` as per-timestep range attributes."""
        attributes: dict[str, object] = {}
        names = group.GetMDArrayNames()
        if _MINS in names:
            attributes["minimum"] = np.asarray(group.OpenMDArray(_MINS).ReadAsArray())
        if _MAXS in names:
            attributes["maximum"] = np.asarray(group.OpenMDArray(_MAXS).ReadAsArray())
        return attributes

    def _maybe_active(
        self, group, name: str, times: np.ndarray, variables: dict
    ) -> None:
        """Attach a face-located ``<name>_active`` sibling when the mask fits the faces.

        The mask counts faces while the parent may be node-located, so it is a standalone
        variable, not ``ParsedVariable.active``. It is only attached when its width is the
        mesh's face count; a mask of another width is left off rather than mislabelled.
        """
        if _ACTIVE not in group.GetMDArrayNames():
            return
        active = np.asarray(group.OpenMDArray(_ACTIVE).ReadAsArray())
        if active.ndim != 2 or active.shape[1] != self.n_face:
            return
        variables[f"{name}_active"] = ParsedVariable(
            location="face",
            shape=active.shape,
            values=active,
            time_values=times,
        )
