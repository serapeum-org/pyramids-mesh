"""TELEMAC Selafin / Serafin driver (task D3).

The first binary driver, and the one that proves the lazy-read path: variable slabs are
left behind a loader that seeks straight to a record, while the topology and the small
time records are read eagerly. Fortran unformatted records with undeclared endianness and
an optional double-precision variant, read through the helpers in ``_binary.py``.

The record layout here was read from the Selafin format specification and has since been
checked against real files from two independent writers -- including the row-versus-column
order of the ``IKLE`` connectivity block, which is read row-major (one element's nodes
contiguous). ``tests/drivers/coastal_ocean/test_selafin_vendor.py`` holds that check. See
``planning/design.md`` section 20.1.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import warnings
from datetime import datetime, timedelta

import numpy as np
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION, MeshDataset

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._binary import (
    _read_record,
    detect_endianness,
    read_payload_floats,
    read_record_floats,
    read_record_ints,
    record_offsets,
)

# The title record is 80 bytes: 72 of free text then an 8-byte format marker. SERAFIND
# means double precision, SERAFIN single; SERAFIND contains SERAFIN, so it is tested
# first. Decoded as latin-1 because every byte is a valid code point there and the field
# is padding, not real text.
_TITLE_LENGTH = 80
_FORMAT_DOUBLE = "SERAFIND"
_FORMAT_SINGLE = "SERAFIN"

# A variable name record is 32 bytes: 16 of name then 16 of unit, each space-padded.
# Split by position, never by collapsing whitespace, or a multi-word name loses its
# tail to the unit.
_NAME_LENGTH = 32
_NAME_HALF = 16

# IPARAM is ten ints. These are the entries this driver acts on, 0-based into the array
# as read (the TELEMAC docs number them 1-10).
_IPARAM_COUNT = 10
_IPARAM_X_ORIGIN = 2
_IPARAM_Y_ORIGIN = 3
_IPARAM_PLANES = 6
# Interface points: nodes shared between partitions, so nonzero only in a parallel
# sub-domain file and zero in any serial one. This is the single reliable parallel
# marker -- IPARAM[7] (boundary points) is a mesh property, not a parallel flag, and
# neither gates the read (IPOBO is read regardless).
_IPARAM_INTERFACE_POINTS = 8
_IPARAM_DATE_FLAG = 9
_DATE_FIELDS = 6
# Boundary points: how many of IPOBO's entries are non-zero. A mesh property, written
# back from the preserved numbering so a written file is usable by TELEMAC.
_IPARAM_BOUNDARY_POINTS = 7

# The element widths of a 2D surface mesh: 3-node triangles or 4-node quadrilaterals. A
# layered 3D file carries 6-node prisms instead, and NDP -- not the plane count -- is what
# separates the two. A file holding one layer extracted from a 3D run keeps its parent's
# plane count in IPARAM[6] while its elements are plain triangles.
_SURFACE_ELEMENT_WIDTHS = (3, 4)
# The width of a layered 3D file's prisms, kept apart from "any other width" so the two are
# reported as what they are: a 3D file this driver will not read, versus a width Selafin
# has no element for at all.
_PRISM_ELEMENT_WIDTH = 6

# Every integer record in the file is int32, so a value outside this range cannot be
# stored and must not be narrowed silently.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# The node variable the boundary numbering (IPOBO) is carried on. One integer per node,
# non-zero on a boundary node: an ordering, not a mask, and not derivable from the
# topology. Excluded from the data slabs a writer emits -- it is mesh metadata that goes
# back into its own record.
_BOUNDARY_VARIABLE = "boundary_node_number"


def _peek_header(path: str, endian: str) -> tuple[np.ndarray, int]:
    """Read just enough of the header to decide whether the file can be read.

    Walks the fixed prefix -- title, the variable counts, one record per variable name,
    ``IPARAM``, the optional date record, then the dimensions record -- and stops there.
    Shared with :meth:`_SelafinReader._read_header` so :meth:`SelafinDriver._sniff` and the
    read cannot drift apart about which files are acceptable.

    Args:
        path: The Selafin file to inspect.
        endian: The byte order :func:`~pyramids_mesh._binary.detect_endianness` found.

    Returns:
        tuple: ``(iparam, ndp)`` -- the ten-entry ``IPARAM`` array and the element width.

    Raises:
        ValueError: If a record is missing or the file ends inside the header.

    Examples:
        - The minimal fixture is a serial 2D mesh of triangles:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh._binary import detect_endianness
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     _peek_header,
            ...     build_minimal_fixture_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     build_minimal_fixture_selafin(path)
            ...     iparam, ndp = _peek_header(
            ...         path, detect_endianness(path, expect_first_record=80)
            ...     )
            >>> ndp, int(iparam[_IPARAM_INTERFACE_POINTS])
            (3, 0)
    """
    with open(path, "rb") as handle:
        if _read_record(handle, endian) is None:
            raise ValueError("the file ends before its title record")
        counts = read_record_ints(handle, endian, 2)
        for _ in range(int(counts[0] + counts[1])):
            if _read_record(handle, endian) is None:
                raise ValueError("the file ends inside its variable-name records")
        iparam = read_record_ints(handle, endian, _IPARAM_COUNT).astype(np.int64)
        if iparam[_IPARAM_DATE_FLAG] == 1:
            read_record_ints(handle, endian, _DATE_FIELDS)
        dims = read_record_ints(handle, endian, 4)
    return iparam, int(dims[2])


class SelafinDriver(BaseDriver):
    """Reader for the TELEMAC Selafin / Serafin format.

    Fortran unformatted records whose byte order is not declared and is detected from the
    first record length (see :func:`~pyramids_mesh._binary.detect_endianness`). The file
    carries a mesh and node-located results over time, so the driver declares
    ``read_mesh``, ``read_data`` and ``lazy_data``: the topology and the one-float time
    records are read eagerly, and each variable's per-timestep slab is left behind a
    loader that seeks to it.

    Documented limitations:
        - **The coordinate origin is applied, and getting it wrong is undetectable.**
          ``IPARAM[2]`` and ``IPARAM[3]`` are the x and y origins the stored coordinates
          are offsets from; they are added to the node coordinates. A file with a
          non-zero origin read without adding it gives a mesh sitting near zero instead
          of in its real location -- plausible and completely wrong, with nothing in the
          data to catch it.
        - **Single versus double precision applies to every float record.** The title's
          format marker (``SERAFIND`` double, ``SERAFIN`` single) sets the width for the
          coordinates and every data slab alike; reading the wrong width desynchronises
          the file, which the trailing-length-word check then reports.
        - **A layered 3D file is refused, keyed on ``NDP``.** The element width says what
          the elements in this file are: 3-node triangles or 4-node quadrilaterals are a
          2D surface mesh, 6-node prisms a layered 3D one whose ``NPOIN`` is the full 3D
          node count. Reading prisms as flat 2D returns every plane's nodes under 6-node
          faces -- plausible and wrong -- so it is refused; a full 3D read needs the
          layered mesh model pyramids does not have yet (design section 21.1). The plane
          count ``IPARAM[6]`` is **not** the test and appears only in the message: a file
          holding one layer extracted from a 3D run inherits its parent's plane count
          while its elements are plain triangles, and keying on the count refused a mesh
          that reads perfectly well.
        - **A parallel sub-domain file is refused.** A non-zero interface-point count
          (``IPARAM[8]``) means local numbering within one partition, so the connectivity
          would index a sub-domain rather than the whole mesh. It is keyed on interface
          points alone -- the boundary-point field is a mesh property, nonzero for
          ordinary serial meshes, and keying on it would refuse them.
        - **Both refusals are declined by ``can_open`` too.** Auto-detection trusts
          ``can_open``, so the sniff reads the same header fields the guards use rather
          than claiming a file the read then throws on. An explicit
          ``driver="selafin"`` still gets the guard's explanation.
        - **The title, start date and boundary numbering are kept.** The title record and
          the ``IPARAM[9]`` date record arrive as the ``title`` and ``start_date`` global
          attributes -- a series of seconds with no anchor is of little use -- and
          ``IPOBO`` arrives on the ``boundary_node_number`` node variable. The numbering is
          an ordering, not a mask, and is not derivable from the topology; TELEMAC needs it
          to attach boundary conditions, so it is carried through a read and written back
          into its own record. Each is omitted rather than faked when the file has none,
          and ``IPOBO`` is read from its record rather than trusted to ``IPARAM[7]``:
          BlueKenue writes a populated record while leaving that count at zero.
        - **Selafin variables are node-located, always.** The location is not inferred
          from an array length.
        - **The IKLE row/column order is row-major**, one element's ``NDP`` node ids
          contiguous, as the format specifies -- confirmed against real files from two
          independent writers. On a TELEMAC 2D result (648 nodes, 1030 triangles) the
          row-major reading gives 1030 counter-clockwise triangles whose areas sum to the
          domain footprint exactly, while the transposed reading inverts 537 of them and
          covers 89 times the domain. The same holds for a BlueKenue-written mesh. A
          fixture cannot settle this -- a transposed writer and a transposed reader agree
          with each other -- so the check runs against real files committed under
          ``tests/data/selafin/``, in the ordinary suite.
        - There is no CRS in the format, so ``crs`` is ``None``; it is never inferred
          from the coordinate magnitudes.

    Examples:
        - Write the smallest valid Selafin file and read it back. The 1-based ``IKLE``
          arrives 0-based, with the file's base recorded for a writer:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     SelafinDriver,
            ...     build_minimal_fixture_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     build_minimal_fixture_selafin(path)
            ...     dataset = SelafinDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1

        - The one variable comes back node-located, with its time axis and the name and
          unit split from the 32-character record by position:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     SelafinDriver,
            ...     build_minimal_fixture_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     build_minimal_fixture_selafin(path)
            ...     dataset = SelafinDriver().read(path)
            ...     variable = dataset.get("WATER DEPTH")
            ...     location, shape, units = (
            ...         variable.location,
            ...         variable.shape,
            ...         variable.units,
            ...     )
            ...     second_step = variable.data[1].tolist()
            >>> location, shape, units
            ('node', (2, 4), 'M')
            >>> second_step
            [1.5, 1.25, 1.75, 1.125]

        - The header fields the file carries about itself arrive with it. The title and the
          start date are global attributes, and the boundary numbering is a node variable,
          so a write can put all three back:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     SelafinDriver,
            ...     write_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "run.slf")
            ...     write_selafin(
            ...         path,
            ...         node_x=[0.0, 1.0, 1.0, 0.0],
            ...         node_y=[0.0, 0.0, 1.0, 1.0],
            ...         ikle=[[1, 2, 3], [1, 3, 4]],
            ...         variable_names=["WATER DEPTH     M               "],
            ...         times=[0.0],
            ...         values=[[[0.5, 0.25, 0.75, 0.125]]],
            ...         title="Rouse profile test",
            ...         ipobo=[1, 2, 0, 3],
            ...         date=[1993, 7, 14, 6, 30, 15],
            ...     )
            ...     dataset = SelafinDriver().read(path)
            >>> dataset.global_attributes["title"]
            'Rouse profile test'
            >>> dataset.global_attributes["start_date"]
            '1993-07-14T06:30:15'
            >>> dataset.get("boundary_node_number").data.tolist()
            [1, 2, 0, 3]
    """

    name = "selafin"
    long_name = "TELEMAC Selafin / Serafin"
    suffixes = (".slf", ".srf", ".ser", ".sel")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset(
        {"read_mesh", "read_data", "lazy_data", "write_mesh", "write_data"}
    )

    def _sniff(self, path: str) -> bool:
        """Report whether this driver can read the file, not merely recognise the format.

        The endianness probe settles whether the file is Selafin at all: the first record
        must declare an 80-byte title in one byte order or the other. It is not the whole
        sniff, because a file can be unmistakably Selafin and still be one :meth:`read`
        refuses -- a layered 3D mesh or a parallel sub-domain. Answering ``True`` for
        those would have auto-detection hand the file to a driver that then throws, so the
        header fields the read-time guards use are read here too and the file is declined.

        The sniff reads the header prefix and stops at the dimensions record, so it rules
        out the *variants* this driver will not read -- not every way a file can fail. A
        file truncated after that point, or corrupt in its connectivity or its data slabs,
        is still claimed here and reported by :meth:`read`; a sniff that proved readability
        outright would have to read the whole file, which is not what dispatch can afford.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the header declares a serial 2D Selafin mesh.
        """
        try:
            endian = detect_endianness(path, expect_first_record=_TITLE_LENGTH)
        except ValueError:
            return False
        try:
            iparam, ndp = _peek_header(path, endian)
        except (ValueError, OSError):
            # A header too short or too corrupt to peek at is not a file this driver can
            # read, whatever its first record said.
            return False
        if iparam[_IPARAM_INTERFACE_POINTS]:
            return False
        return ndp in _SURFACE_ELEMENT_WIDTHS

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a Selafin file into a :class:`ParsedMesh`.

        Args:
            path: The Selafin file to read.
            lazy: When ``True``, leave each variable's slab behind a loader; when
                ``False``, read every value now.

        Returns:
            ParsedMesh: Nodes, faces and the node-located variables.

        Raises:
            FileFormatNotSupportedError: If the file is not Selafin, declares a parallel
                sub-domain, or is truncated against the counts its header promises.
        """
        return _SelafinReader(path, lazy=lazy).build()

    def write(self, dataset: MeshDataset, path: str, **options: object) -> None:
        """Write a mesh and its node-located results back to a Selafin file.

        Emits the same Fortran-record layout the reader parses, through
        :func:`write_selafin`: the title, the variable names, ``IPARAM``, the
        ``NELEM``/``NPOIN``/``NDP`` record, the ``IKLE`` connectivity (the base recorded
        on the mesh is re-applied, so a file read 1-based is written back 1-based),
        ``IPOBO``, the ``x``/``y`` node arrays, then one time record per step carrying each
        node-located variable's slab.

        Selafin stores one element width, so every face must share a vertex count; a
        mixed-width mesh is refused. Only node-located variables are written -- the format
        has no face-located storage. Coordinates and values are written at ``float64``
        (double precision) by default, so the round trip is lossless; pass
        ``precision="single"`` for the smaller, lossy ``float32`` layout.

        Four header fields are restored rather than stamped with this package's defaults:
        the ``title`` and ``start_date`` attributes become the title and date records, the
        ``boundary_node_number`` variable becomes ``IPOBO``, and the ``x_origin``/
        ``y_origin`` attributes become ``IPARAM[2]``/``IPARAM[3]`` with the node
        coordinates written as offsets from them. ``IPARAM[7]`` and the ``IPARAM[9]`` flag
        are derived from what is actually written, so the header cannot claim what the
        records do not carry. A file read and written again is therefore still the model's
        own file.

        Restoring the origin is what keeps ``precision="single"`` usable on a projected
        mesh: absolute coordinates spend the float32 mantissa on the large constant, so the
        same mesh lost 0.125 m in y before this. Three consequences worth knowing:
        ``boundary_node_number`` is mesh metadata and is never emitted as a result slab; a
        dataset built in memory has no boundary numbering to restore -- a boundary
        *ordering* cannot be derived from the node set -- so zeros are written and a
        warning says the output is geometry only; and the plane count ``IPARAM[6]`` is
        **not** carried forward, because what is written is a single 2D layer whatever it
        was extracted from.

        Args:
            dataset: The :class:`~pyramids.netcdf.ugrid.MeshDataset` to write.
            path: The output ``.slf`` path.
            **options: ``precision`` is ``"double"`` (default) or ``"single"``.

        Raises:
            FileFormatNotSupportedError: If the dataset has no face connectivity, if its
                faces do not all share one vertex count (Selafin needs a uniform ``NDP``),
                or if ``precision`` is neither ``"single"`` nor ``"double"``.

        Warns:
            UserWarning: If the dataset carries no boundary numbering, so ``IPOBO`` is
                written as zeros, or if a node-located ``node_z`` is dropped.

        Examples:
            - A written file keeps the dataset's own title, start date and boundary
              numbering, so read -> write -> read is still the model's file and not this
              package's:

                >>> import os
                >>> import tempfile
                >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
                ...     SelafinDriver,
                ...     write_selafin,
                ... )
                >>> with tempfile.TemporaryDirectory() as directory:
                ...     source = os.path.join(directory, "source.slf")
                ...     write_selafin(
                ...         source,
                ...         node_x=[0.0, 1.0, 1.0, 0.0],
                ...         node_y=[0.0, 0.0, 1.0, 1.0],
                ...         ikle=[[1, 2, 3], [1, 3, 4]],
                ...         variable_names=["WATER DEPTH     M               "],
                ...         times=[0.0],
                ...         values=[[[0.5, 0.25, 0.75, 0.125]]],
                ...         title="Rouse profile test",
                ...         ipobo=[1, 2, 0, 3],
                ...         date=[1993, 7, 14, 6, 30, 15],
                ...     )
                ...     copy = os.path.join(directory, "copy.slf")
                ...     SelafinDriver().write(SelafinDriver().read(source), copy)
                ...     restored = SelafinDriver().read(copy)
                >>> restored.global_attributes["title"]
                'Rouse profile test'
                >>> restored.global_attributes["start_date"]
                '1993-07-14T06:30:15'
                >>> restored.global_attributes["n_boundary_nodes"]
                3

            - The boundary numbering is mesh metadata, so it goes back into the ``IPOBO``
              record rather than becoming a sixth result; the written file still declares
              the one variable it started with:

                >>> import os
                >>> import tempfile
                >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
                ...     SelafinDriver,
                ...     write_selafin,
                ... )
                >>> with tempfile.TemporaryDirectory() as directory:
                ...     source = os.path.join(directory, "source.slf")
                ...     write_selafin(
                ...         source,
                ...         node_x=[0.0, 1.0, 1.0, 0.0],
                ...         node_y=[0.0, 0.0, 1.0, 1.0],
                ...         ikle=[[1, 2, 3], [1, 3, 4]],
                ...         variable_names=["WATER DEPTH     M               "],
                ...         times=[0.0],
                ...         values=[[[0.5, 0.25, 0.75, 0.125]]],
                ...         ipobo=[1, 2, 0, 3],
                ...     )
                ...     copy = os.path.join(directory, "copy.slf")
                ...     SelafinDriver().write(SelafinDriver().read(source), copy)
                ...     restored = SelafinDriver().read(copy)
                >>> restored.variable_names
                ['WATER DEPTH', 'boundary_node_number']
                >>> restored.get("boundary_node_number").data.tolist()
                [1, 2, 0, 3]
        """
        mesh = dataset.mesh
        connectivity = getattr(mesh, "face_node_connectivity", None)
        if connectivity is None or connectivity.data is None:
            raise FileFormatNotSupportedError(
                f"cannot write {path!r}: the dataset has no face connectivity to write "
                f"as a Selafin mesh"
            )
        faces = np.asarray(connectivity.data)
        base = connectivity.original_start_index
        if base is None:
            base = 1
        ikle = [[int(v) + base for v in row if int(v) != -1] for row in faces]
        widths = {len(row) for row in ikle}
        if len(widths) != 1:
            raise FileFormatNotSupportedError(
                f"cannot write {path!r} to Selafin: every element must have the same node "
                f"count, but the mesh mixes widths {sorted(widths)}"
            )
        # The reader added the origin, so writing absolute coordinates against a zero
        # origin preserves the values -- but only at float64. At float32 the whole mantissa
        # goes on the large constant: a projected mesh loses decimetres. Subtracting the
        # origin again and declaring it keeps the offsets small, which is how the source
        # file stored them losslessly in the first place.
        origin = self._coordinate_origin(dataset)
        node_x = (np.asarray(mesh.node_x, dtype=np.float64) - origin[0]).tolist()
        node_y = (np.asarray(mesh.node_y, dtype=np.float64) - origin[1]).tolist()
        if "node_z" in dataset.variable_names:
            z = dataset.get("node_z")
            if z is not None and z.location == "node":
                warnings.warn(
                    "Selafin stores only node x/y, so the dataset's node_z (bed "
                    "elevation) is dropped on write; it is geometry, not a result series.",
                    stacklevel=2,
                )
        node_pairs = [
            (name, dataset.get(name))
            for name in dataset.variable_names
            if dataset.get(name).location == "node"
            and name not in ("node_z", _BOUNDARY_VARIABLE)
        ]
        names = [self._selafin_name(name, var.units) for name, var in node_pairs]
        if node_pairs:
            stamps = dataset.time_values
            times = (
                [float(t) for t in stamps]
                if stamps is not None and len(stamps)
                else [0.0]
            )
        else:
            times = []
        values = self._node_series(node_pairs, len(times))
        ipobo = self._boundary_numbering(dataset, len(node_x), path)
        write_selafin(
            path,
            node_x=node_x,
            node_y=node_y,
            ikle=ikle,
            variable_names=names,
            times=times,
            values=values,
            double=self._resolve_precision(options.get("precision")),
            title=dataset.global_attributes.get("title"),
            origin=origin,
            ipobo=ipobo,
            date=self._date_fields(dataset.global_attributes.get("start_date"), path),
        )

    @staticmethod
    def _boundary_numbering(
        dataset: MeshDataset, n_node: int, path: str
    ) -> list[int] | None:
        """Return the ``IPOBO`` numbering to write, or ``None`` when the dataset has none.

        A dataset read from a Selafin file carries the numbering on the
        ``boundary_node_number`` node variable, and writing it back is what makes the
        output usable by TELEMAC, which needs it to attach boundary conditions. A mesh
        built in memory has none, and it cannot be derived -- the boundary *ordering* is
        not recoverable from the node set -- so zeros are written and the caller is warned
        rather than left to discover it from the model.

        Args:
            dataset: The dataset being written.
            n_node: The node count the numbering must match.
            path: The output path, for the warning message.

        Returns:
            list[int] | None: one entry per node, or ``None`` to write zeros.
        """
        reason = f"the dataset carries no {_BOUNDARY_VARIABLE!r}"
        if _BOUNDARY_VARIABLE in dataset.variable_names:
            variable = dataset.get(_BOUNDARY_VARIABLE)
            location = getattr(variable, "location", None)
            array = np.asarray(variable.data).reshape(-1)
            if location != "node":
                # IPOBO is one entry per node by definition; a same-named face or edge
                # field is a different quantity and writing it would mislabel it.
                reason = (
                    f"the dataset's {_BOUNDARY_VARIABLE!r} is {location}-located, and "
                    f"IPOBO is one entry per node"
                )
            elif array.size != n_node:
                # Say which it is. Reporting "carries no numbering" for a numbering that is
                # merely the wrong size sends the reader looking for the wrong problem.
                reason = (
                    f"the dataset's {_BOUNDARY_VARIABLE!r} has {array.size} entries for "
                    f"{n_node} nodes"
                )
            else:
                return [int(v) for v in array]
        warnings.warn(
            f"{path!r} is written with an all-zero IPOBO record: {reason}, and a boundary "
            f"numbering cannot be derived from the topology. TELEMAC needs it to attach "
            f"boundary conditions, so the file is geometry only.",
            stacklevel=3,
        )
        return None

    @staticmethod
    def _coordinate_origin(dataset: MeshDataset) -> tuple[int, int]:
        """Return the ``IPARAM[2]``/``IPARAM[3]`` origin the coordinates are offsets from.

        Selafin stores an integer origin and float offsets from it, and the reader folds
        the two together into absolute coordinates. Writing those absolute values against
        a zero origin is exact at ``float64`` and lossy at ``float32``: a projected mesh
        spends the mantissa on the constant rather than the detail. Taking the origin back
        out restores the split the source file used.

        Non-integer or missing attributes answer ``(0, 0)``, which is the right answer for
        a mesh built in memory -- its coordinates are simply absolute.

        Args:
            dataset: The dataset being written.

        Returns:
            tuple[int, int]: the x and y origin to declare and subtract.
        """
        origin = []
        for key in ("x_origin", "y_origin"):
            value = dataset.global_attributes.get(key, 0)
            try:
                origin.append(int(value))
            except (TypeError, ValueError):
                origin.append(0)
        return origin[0], origin[1]

    @staticmethod
    def _date_fields(start_date: object, path: str) -> list[int] | None:
        """Convert an ISO-8601 ``start_date`` attribute into the six-int date record.

        Parsed with :meth:`datetime.datetime.fromisoformat` rather than split on
        separators: splitting on ``"-"`` cannot read a negative year, and nothing but a
        real parser agrees with :meth:`_SelafinReader._start_date_iso`, which is this
        function's inverse. The two have to agree exactly -- the reader's own output must
        survive a write -- and the only way to guarantee that is for both to go through
        ``datetime``.

        An unusable value is dropped rather than written, because a half-written date
        record would desynchronise every record after it. Dropping it is announced: unlike
        an absent ``start_date``, a present-but-unusable one means the caller believes the
        file carries a date that it will not.

        Args:
            start_date: The dataset's ``start_date`` attribute, or ``None``.
            path: The output path, for the warning message.

        Returns:
            list[int] | None: ``[year, month, day, hour, minute, second]``, or ``None``.
        """
        if start_date is None:
            return None
        if isinstance(start_date, datetime):
            stamp = start_date
        elif isinstance(start_date, str):
            try:
                stamp = datetime.fromisoformat(start_date)
            except ValueError as error:
                warnings.warn(
                    f"{path!r} is written without a date record: its 'start_date' "
                    f"attribute {start_date!r} is not an ISO-8601 instant ({error}).",
                    stacklevel=4,
                )
                return None
        else:
            warnings.warn(
                f"{path!r} is written without a date record: its 'start_date' attribute "
                f"is a {type(start_date).__name__}, not an ISO-8601 string.",
                stacklevel=4,
            )
            return None
        return [
            stamp.year,
            stamp.month,
            stamp.day,
            stamp.hour,
            stamp.minute,
            stamp.second,
        ]

    @staticmethod
    def _node_series(node_pairs, n_times: int) -> list:
        """Build the per-step value blocks for the node variables.

        Each 2-D (time-varying) variable must share the dataset's time axis; a static
        (1-D) field is repeated on every step. A variable whose own time axis differs is
        refused with a clear error rather than being silently truncated or index-erroring.

        Args:
            node_pairs: ``(name, variable)`` pairs of the node-located variables to write.
            n_times: The dataset's time-step count (the shared Selafin time axis).

        Returns:
            list: One block per time step, each a list of per-variable node value lists.

        Raises:
            FileFormatNotSupportedError: If a time-varying variable's first-axis length is
                not ``n_times``.
        """
        prepared = []
        for name, var in node_pairs:
            array = np.asarray(var.data, dtype=np.float64)
            if array.ndim == 2 and array.shape[0] != n_times:
                raise FileFormatNotSupportedError(
                    f"cannot write node variable {name!r} to Selafin: its time axis has "
                    f"{array.shape[0]} steps but the dataset's is {n_times}"
                )
            prepared.append(array)
        return [
            [(array[step] if array.ndim == 2 else array).tolist() for array in prepared]
            for step in range(n_times)
        ]

    @staticmethod
    def _resolve_precision(precision) -> bool:
        """Resolve the ``precision`` option to the double-precision flag ``write_selafin`` takes.

        Returns:
            bool: ``True`` for double (the default, lossless), ``False`` for single.

        Raises:
            FileFormatNotSupportedError: On any value other than ``"single"``/``"double"``.
        """
        if precision in (None, "double"):
            return True
        if precision == "single":
            return False
        raise FileFormatNotSupportedError(
            f"unknown Selafin precision {precision!r}; use 'single' or 'double'"
        )

    @staticmethod
    def _selafin_name(name: str, units) -> str:
        """Pack a variable name and unit into the 32-character Selafin name record.

        The first 16 characters are the name, the next 16 the unit, each space-padded and
        truncated -- the inverse of the reader's by-position split.
        """
        label = f"{name[:_NAME_HALF]:<{_NAME_HALF}}"
        unit = f"{(units or '')[:_NAME_HALF]:<{_NAME_HALF}}"
        return label + unit


def build_minimal_fixture_selafin(path: str) -> None:
    """Write the smallest valid Selafin file, for the cross-driver invariant harness.

    A little-endian, single-precision file: the unit square split into two
    counter-clockwise triangles, with one variable (``WATER DEPTH``) over two timesteps.
    Named per driver because the ``telemac`` module may hold more than one.

    Args:
        path: Where to write the file.

    Examples:
        - The file reads back as four nodes, two faces and one variable:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     SelafinDriver,
            ...     build_minimal_fixture_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.slf")
            ...     build_minimal_fixture_selafin(path)
            ...     dataset = SelafinDriver().read(path)
            >>> dataset.mesh.n_node
            4
            >>> dataset.variable_names
            ['WATER DEPTH']
    """
    write_selafin(
        path,
        # Named, because the writer's default title is no longer this one: the committed
        # tests/data/minimal.slf is compared byte for byte against this builder's output.
        title="pyramids-mesh fixture",
        node_x=[0.0, 1.0, 1.0, 0.0],
        node_y=[0.0, 0.0, 1.0, 1.0],
        ikle=[[1, 2, 3], [1, 3, 4]],
        variable_names=["WATER DEPTH     M               "],
        times=[0.0, 3600.0],
        values=[
            [[0.5, 0.25, 0.75, 0.125]],
            [[1.5, 1.25, 1.75, 1.125]],
        ],
    )


def _derive_iparam(
    boundary: np.ndarray, has_date: bool, origin: tuple[int, int] = (0, 0)
) -> list[int]:
    """Build the ``IPARAM`` array a written file's own records imply.

    ``IPARAM[0]`` is 1 by convention, ``IPARAM[2]``/``IPARAM[3]`` are the coordinate
    origin the written offsets are measured from, ``IPARAM[7]`` is the boundary-point
    count, and ``IPARAM[9]`` flags a following date record. Each is derived from what is
    actually written rather than taken on trust, so the header cannot claim a boundary
    count, an origin or a date that the records do not carry.

    The rest stays zero deliberately. ``IPARAM[6]`` (plane count) and ``IPARAM[8]``
    (interface points) describe a layered or partitioned file, and nothing written through
    here is either -- carrying a source file's plane count forward would label a
    single-layer mesh as part of a 3D run it is no longer part of.

    Args:
        boundary: The ``IPOBO`` numbering about to be written.
        has_date: Whether a date record follows ``IPARAM``.
        origin: The ``(x, y)`` origin the node coordinates are offsets from.

    Returns:
        list[int]: the ten-entry ``IPARAM`` array.

    Examples:
        - A mesh with two boundary nodes and no date record:

            >>> import numpy as np
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import _derive_iparam
            >>> _derive_iparam(np.array([1, 0, 2, 0]), False)
            [1, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    """
    params = [0] * _IPARAM_COUNT
    params[0] = 1
    params[_IPARAM_BOUNDARY_POINTS] = int(np.count_nonzero(boundary))
    params[_IPARAM_DATE_FLAG] = 1 if has_date else 0
    params[_IPARAM_X_ORIGIN], params[_IPARAM_Y_ORIGIN] = int(origin[0]), int(origin[1])
    return params


def write_selafin(
    path: str,
    *,
    node_x: list[float],
    node_y: list[float],
    ikle: list[list[int]],
    variable_names: list[str],
    times: list[float],
    values: list[list[list[float]]],
    endian: str = "<",
    double: bool = False,
    iparam: list[int] | None = None,
    title: str | None = None,
    ipobo: list[int] | None = None,
    date: list[int] | None = None,
    origin: tuple[int, int] = (0, 0),
) -> None:
    """Write a Selafin file, with every variant the driver and its tests need.

    Deliberately on the module surface rather than hidden in the tests: it is the only
    writer of this binary format in the package, and the test variants (big-endian,
    double precision, coordinate origin, 3D, parallel) all come from its keyword
    arguments so one implementation of the record layout is exercised.

    ``title``, ``ipobo`` and ``date`` are what :meth:`SelafinDriver.write` restores from a
    dataset, so a file read and written again keeps its own identity instead of being
    stamped with this function's defaults. When ``iparam`` is not given it is derived:
    ``IPARAM[7]`` is set from ``ipobo``'s non-zero count and ``IPARAM[9]`` is the flag for
    the date record, so the header cannot claim one thing and the records show another.

    Args:
        path: Where to write the file.
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        ikle: Element connectivity, one row of ``NDP`` 1-based node ids per element.
        variable_names: One 32-character name/unit record per variable.
        times: The time value of each timestep.
        values: ``values[t][v]`` is variable ``v``'s array at timestep ``t``.
        endian: ``"<"`` little-endian or ``">"`` big-endian.
        double: Whether to write ``float64`` rather than ``float32``.
        iparam: The 10-int IPARAM array. Defaults to one derived from ``ipobo`` and
            ``date``; pass it explicitly to write a file the derivation would not produce,
            as the 3D and parallel fixtures do.
        title: The 72-character free-text title. Defaults to ``"pyramids-mesh"``.
        ipobo: The boundary-node numbering, one entry per node. Defaults to all zeros,
            which declares a mesh with no boundary nodes.
        date: ``[year, month, day, hour, minute, second]`` for the date record. Written
            only when the effective ``IPARAM[9]`` is 1, and required when an explicit
            ``iparam`` sets that flag -- the two may not disagree.
        origin: The ``(x, y)`` origin to declare in ``IPARAM[2]``/``IPARAM[3]``. The node
            coordinates passed here are the offsets from it, already relative, so this
            function neither adds nor subtracts it. Keeping a large constant in the origin
            is what lets ``double=False`` store a projected mesh without losing
            centimetres to the float32 mantissa.

    Raises:
        ValueError: If ``values`` does not match the timestep and variable counts, or if
            ``ipobo`` or ``date`` has the wrong length.

    Examples:
        - A named title and a boundary numbering come back off the file, and ``IPARAM`` is
          derived from them rather than passed separately, so the header cannot claim one
          thing while the records show another:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     SelafinDriver,
            ...     write_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "flume.slf")
            ...     write_selafin(
            ...         path,
            ...         node_x=[0.0, 1.0, 1.0, 0.0],
            ...         node_y=[0.0, 0.0, 1.0, 1.0],
            ...         ikle=[[1, 2, 3], [1, 3, 4]],
            ...         variable_names=["WATER DEPTH     M               "],
            ...         times=[0.0],
            ...         values=[[[0.5, 0.25, 0.75, 0.125]]],
            ...         title="flume test",
            ...         ipobo=[1, 2, 0, 3],
            ...     )
            ...     dataset = SelafinDriver().read(path)
            >>> dataset.global_attributes["title"], dataset.global_attributes["n_boundary_nodes"]
            ('flume test', 3)

        - Left out, neither is declared: the file gets the neutral default title, no date
          record and an all-zero ``IPOBO``, which is a mesh with no boundary nodes rather
          than one claiming boundary nodes it does not have:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     SelafinDriver,
            ...     write_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "plain.slf")
            ...     write_selafin(
            ...         path,
            ...         node_x=[0.0, 1.0, 1.0, 0.0],
            ...         node_y=[0.0, 0.0, 1.0, 1.0],
            ...         ikle=[[1, 2, 3], [1, 3, 4]],
            ...         variable_names=["WATER DEPTH     M               "],
            ...         times=[0.0],
            ...         values=[[[0.5, 0.25, 0.75, 0.125]]],
            ...     )
            ...     dataset = SelafinDriver().read(path)
            >>> dataset.global_attributes["title"]
            'pyramids-mesh'
            >>> "start_date" in dataset.global_attributes
            False
            >>> dataset.variable_names
            ['WATER DEPTH']
    """
    if len(values) != len(times):
        raise ValueError("values must hold one entry per timestep")
    npoin = len(node_x)
    if ipobo is not None and len(ipobo) != npoin:
        raise ValueError(
            f"ipobo must hold one entry per node: got {len(ipobo)} for {npoin} nodes"
        )
    if date is not None and len(date) != _DATE_FIELDS:
        raise ValueError(f"date must hold {_DATE_FIELDS} fields, got {len(date)}")
    if ipobo is not None:
        # Every record in the file is int32. A numbering that does not fit wraps silently
        # (2**31 becomes -2147483648), which would look like a plausible boundary number
        # rather than a lost one, so it is refused.
        out_of_range = [v for v in ipobo if not _INT32_MIN <= int(v) <= _INT32_MAX]
        if out_of_range:
            raise ValueError(
                f"ipobo entries must fit in the int32 the record stores; "
                f"{out_of_range[:3]} do not"
            )
    if date is not None and iparam is not None and iparam[_IPARAM_DATE_FLAG] != 1:
        # The flag and the record have to agree, and only the caller knows which it meant.
        # Writing the date under a zero flag leaves a record no reader will look for, and
        # dropping it loses what was asked for -- so neither is guessed.
        raise ValueError(
            "a date was given but the explicit iparam clears the date flag "
            f"(IPARAM[9] = {iparam[_IPARAM_DATE_FLAG]}); set it to 1 or drop the date"
        )
    if date is None and iparam is not None and iparam[_IPARAM_DATE_FLAG] == 1:
        raise ValueError(
            "the explicit iparam sets the date flag (IPARAM[9] = 1) but no date was "
            "given; pass date= or clear the flag"
        )
    ndp = len(ikle[0])
    float_code = f"{endian}{'f8' if double else 'f4'}"
    marker = _FORMAT_DOUBLE if double else _FORMAT_SINGLE
    title_record = f"{(title or 'pyramids-mesh'):<72}{marker:<8}"[:_TITLE_LENGTH]
    boundary = (
        np.asarray(ipobo, dtype=np.int64)
        if ipobo is not None
        else np.zeros(npoin, dtype=np.int64)
    )
    params = (
        list(iparam)
        if iparam is not None
        else _derive_iparam(boundary, date is not None, origin)
    )

    def record(payload: bytes) -> bytes:
        word = np.array([len(payload)], dtype=f"{endian}i4").tobytes()
        return word + payload + word

    with open(path, "wb") as handle:
        handle.write(record(title_record.encode("latin-1")))
        handle.write(
            record(np.array([len(variable_names), 0], f"{endian}i4").tobytes())
        )
        for name in variable_names:
            handle.write(record(f"{name:<32}"[:_NAME_LENGTH].encode("latin-1")))
        handle.write(record(np.array(params, dtype=f"{endian}i4").tobytes()))
        if params[_IPARAM_DATE_FLAG] == 1:
            # date is guaranteed present here: the flag is either derived from it or
            # validated against it above, so there is nothing to invent.
            handle.write(record(np.array(date, f"{endian}i4").tobytes()))
        handle.write(
            record(np.array([len(ikle), npoin, ndp, 1], dtype=f"{endian}i4").tobytes())
        )
        flat = np.array(ikle, dtype=f"{endian}i4").reshape(-1)
        handle.write(record(flat.tobytes()))
        handle.write(record(boundary.astype(f"{endian}i4").tobytes()))
        handle.write(record(np.array(node_x, dtype=float_code).tobytes()))
        handle.write(record(np.array(node_y, dtype=float_code).tobytes()))
        for step_values, time in zip(values, times):
            handle.write(record(np.array([time], dtype=float_code).tobytes()))
            for variable in step_values:
                handle.write(record(np.array(variable, dtype=float_code).tobytes()))


class _SelafinReader:
    """One Selafin file, read header-first with the data slabs left for a loader."""

    def __init__(self, path: str, *, lazy: bool) -> None:
        self.path = path
        self.lazy = lazy
        try:
            self.endian = detect_endianness(path, expect_first_record=_TITLE_LENGTH)
        except ValueError as error:
            # detect_endianness raises ValueError for a file it does not recognise; a
            # driver reports an unreadable file as the format error, so the raw probe
            # error is translated rather than leaked.
            raise FileFormatNotSupportedError(
                f"{path!r} is not a Selafin file: {error}"
            ) from error
        self.double = False
        self.names: list[str] = []
        self.iparam = np.zeros(_IPARAM_COUNT, dtype=np.int64)
        self.nelem = 0
        self.npoin = 0
        self.ndp = 0
        self.header_records = 0
        self.title = ""
        self.date: np.ndarray | None = None
        self.ipobo: np.ndarray | None = None

    def build(self) -> ParsedMesh:
        """Read the header, then index the data records and build the parsed mesh.

        Returns:
            ParsedMesh: The topology and the node-located variables.

        Raises:
            FileFormatNotSupportedError: On a parallel sub-domain file, a layered 3D
                file, or a file truncated or corrupt against the counts its header
                promises -- the last translated from the Fortran-record helpers' bare
                ``ValueError``.
        """
        try:
            node_x, node_y, faces = self._read_header()
            offsets = record_offsets(self.path, self.endian)
            times, variables = self._read_data(offsets)
        except ValueError as error:
            # The Fortran-record helpers signal a truncated or corrupt file with a bare
            # ValueError; the driver contract is to report an unreadable file as the
            # format error, so it is translated here rather than leaked to the caller.
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable Selafin file: {error}"
            ) from error
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # IKLE is 1-based; the validator subtracts the base. Recorded for a writer.
            start_index=1,
            original_start_index=1,
            crs=None,
            variables=variables,
            global_attributes=self._global_attributes(len(times)),
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _global_attributes(self, n_timesteps: int) -> dict:
        """Return the dataset attributes, including the header fields worth keeping.

        The title is the run's own name for itself, and ``start_date`` is the instant the
        time values are offsets from -- without it a series of seconds has no anchor. Both
        are absent from the file often enough that each is omitted rather than faked:
        ``title`` only when the record held text, ``start_date`` only when ``IPARAM[9]``
        declares a date record.

        Args:
            n_timesteps: How many whole timesteps the file holds.

        Returns:
            dict: The global attributes for the built dataset.
        """
        attributes = {"source": "TELEMAC Selafin", "n_timesteps": n_timesteps}
        if self.title:
            attributes["title"] = self.title
        start_date = self._start_date_iso()
        if start_date is not None:
            attributes["start_date"] = start_date
        # Only when there is a numbering to count. A mesh whose IPOBO is all zeros has no
        # boundary nodes declared, and gets neither this key nor the node variable, so the
        # two cannot disagree.
        n_boundary = 0 if self.ipobo is None else int(np.count_nonzero(self.ipobo))
        if n_boundary:
            attributes["n_boundary_nodes"] = n_boundary
        # The origin the stored offsets were measured from. The node coordinates are
        # absolute, so this is not needed to place the mesh -- it is kept so a writer can
        # restore the integer-origin/small-offset split, which is what keeps
        # precision="single" from spending the mantissa on a large constant.
        x_origin = int(self.iparam[_IPARAM_X_ORIGIN])
        y_origin = int(self.iparam[_IPARAM_Y_ORIGIN])
        if x_origin or y_origin:
            attributes["x_origin"] = x_origin
            attributes["y_origin"] = y_origin
        return attributes

    def _start_date_iso(self) -> str | None:
        """Return the start instant as an ISO-8601 string, or ``None`` when absent.

        The date record is six ints -- year, month, day, hour, minute, second -- and a file
        without the ``IPARAM[9]`` flag has none at all. The record is **validated as a
        calendar instant** rather than formatted blindly, because formatting is not
        parsing: a partial record such as ``[1993, 0, 0, 12, 0, 0]`` formats cleanly into
        ``'1993-00-00T12:00:00'``, which looks like a timestamp and which no date parser
        accepts. Publishing that under a name documented as ISO-8601 would hand every
        downstream ``datetime.fromisoformat`` a crash on a file that read without complaint.
        :class:`datetime.datetime` is the validator, so month 0, day 0, hour 24, a negative
        field and a year outside 1-9999 are all rejected by construction.

        An all-zero record is TELEMAC's "no date" placeholder and is common, so it answers
        ``None`` quietly. Any other unusable record is a surprise worth a warning, since
        something was there and is being dropped.

        Returns:
            str | None: ``"YYYY-MM-DDTHH:MM:SS"``, or ``None`` when the file declares no
                usable date.
        """
        if self.date is None:
            return None
        year, month, day, hour, minute, second = (int(v) for v in self.date)
        if (year, month, day) == (0, 0, 0):
            return None
        try:
            if (hour, minute, second) == (24, 0, 0):
                # ISO-8601's end-of-day hour, which datetime() refuses and
                # datetime.fromisoformat() -- the writer's parser -- rolls into the next
                # day. Normalised here so the two agree on it rather than one dropping
                # what the other accepts.
                stamp = datetime(year, month, day) + timedelta(days=1)
            else:
                stamp = datetime(year, month, day, hour, minute, second)
        except (ValueError, OverflowError) as error:
            warnings.warn(
                f"{self.path!r} declares a date record that is not a calendar instant "
                f"({[year, month, day, hour, minute, second]}: {error}); no 'start_date' "
                f"is reported, so the time values stay relative.",
                stacklevel=5,
            )
            return None
        return stamp.isoformat()

    def _read_header(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read every header record in order, up to the first timestep.

        Record order is fixed, so the reads are sequential and each guard fires as soon as
        the field it judges is in hand: the parallel guard right after ``IPARAM``, and the
        3D guard only after the dimensions record, since it is keyed on ``NDP``. The title,
        the date record and ``IPOBO`` are kept on the reader as they go past rather than
        being read and dropped, because the writer restores them.

        Returns:
            tuple: ``(node_x, node_y, faces)`` -- coordinates with the IPARAM origin
                applied, and the 1-based connectivity reshaped to ``(NELEM, NDP)``.

        Raises:
            FileFormatNotSupportedError: If the file declares a parallel sub-domain, if it
                is a layered 3D mesh, or if a header record is missing.
        """
        with open(self.path, "rb") as handle:
            title_bytes = self._read_raw(handle)
            title_text = title_bytes.decode("latin-1")
            self.double = _FORMAT_DOUBLE in title_text
            # The first 72 bytes are free text and the last 8 the format marker; the text
            # is space-padded, so it is stripped rather than kept at full width.
            self.title = title_text[: _TITLE_LENGTH - 8].strip()
            nbv = read_record_ints(handle, self.endian, 2)
            n_variables = int(nbv[0] + nbv[1])
            self.names = [
                self._read_raw(handle).decode("latin-1") for _ in range(n_variables)
            ]
            self.iparam = read_record_ints(handle, self.endian, _IPARAM_COUNT).astype(
                np.int64
            )
            self.header_records = 3 + n_variables
            if self.iparam[_IPARAM_DATE_FLAG] == 1:
                self.date = read_record_ints(handle, self.endian, _DATE_FIELDS)
                self.header_records += 1
            self._guard_parallel()
            dims = read_record_ints(handle, self.endian, 4)
            self.nelem, self.npoin, self.ndp = (
                int(dims[0]),
                int(dims[1]),
                int(dims[2]),
            )
            # After the dimensions record, because the guard is keyed on NDP: the plane
            # count alone refuses a single layer extracted from a 3D run.
            self._guard_3d()
            ikle = read_record_ints(handle, self.endian, self.nelem * self.ndp)
            # IPOBO: the boundary-node numbering. An ordering rather than a mask, and not
            # derivable from the topology, so it is kept for a writer to restore.
            self.ipobo = read_record_ints(handle, self.endian, self.npoin).astype(
                np.int64
            )
            node_x = read_record_floats(
                handle, self.endian, self.npoin, double=self.double
            )
            node_y = read_record_floats(
                handle, self.endian, self.npoin, double=self.double
            )
            self.header_records += 5
        faces = ikle.reshape(self.nelem, self.ndp).astype(np.int64)
        node_x = node_x.astype(np.float64) + float(self.iparam[_IPARAM_X_ORIGIN])
        node_y = node_y.astype(np.float64) + float(self.iparam[_IPARAM_Y_ORIGIN])
        return node_x, node_y, faces

    def _read_raw(self, handle) -> bytes:
        """Read one record's raw payload bytes, for the title and name records."""
        payload = _read_record(handle, self.endian)
        if payload is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends inside its header: a record expected here is "
                f"missing"
            )
        return payload

    def _guard_parallel(self) -> None:
        """Refuse a parallel sub-domain file rather than return local indices.

        Keyed on interface points alone. A serial file has none, so this never fires on
        one; a partition has nodes shared with its neighbours, so a nonzero count marks
        local numbering whose connectivity indexes a sub-domain, not the whole mesh. The
        boundary-point field is deliberately ignored: it is a mesh property, nonzero for
        essentially every real mesh, and keying on it would refuse ordinary serial files.
        """
        interface = int(self.iparam[_IPARAM_INTERFACE_POINTS])
        if interface:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a parallel sub-domain file ({interface} interface "
                f"point(s)): its numbering is local to one partition, so the "
                f"connectivity would index a sub-domain rather than the whole mesh. "
                f"Merge the partitions before reading."
            )

    def _guard_3d(self) -> None:
        """Refuse a layered 3D file rather than return a malformed pseudo-2D mesh.

        Keyed on ``NDP``, the element width, which says what the elements in *this* file
        are: 3-node triangles or 4-node quadrilaterals are a 2D surface mesh, and 6-node
        prisms are a layered 3D mesh whose ``NPOIN`` is the full 3D node count. Reading a
        prism file as flat 2D yields every plane's nodes under 6-node "faces" -- plausible
        and wrong -- so it is refused. A full 3D read needs the layered mesh model pyramids
        does not have yet (design section 21.1), and extracting one plane out of a layered
        mesh is a mesh operation, which belongs upstream rather than in a format driver
        (ADR-0005).

        The plane count ``IPARAM[6]`` is deliberately *not* the test, only part of the
        message. A file holding a single layer extracted from a 3D run inherits its
        parent's plane count while its elements are plain triangles: keying on the count
        refused a mesh that reads perfectly well.

        Raises:
            FileFormatNotSupportedError: If ``NDP`` is not a 2D surface element width.
        """
        if self.ndp in _SURFACE_ELEMENT_WIDTHS:
            return
        planes = int(self.iparam[_IPARAM_PLANES])
        if self.ndp == _PRISM_ELEMENT_WIDTH:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a layered 3D Selafin file (NDP=6 prisms over {planes} "
                f"plane(s)); only 2D meshes are read. A 3D read needs the layered mesh "
                f"model pyramids does not have yet, and extracting a single plane is a "
                f"mesh operation that belongs upstream rather than in a format driver."
            )
        # Not 6 either: diagnosing this as "3D" would send the reader looking for a
        # layered file that is not there. NDP 1, 2, 5 and 8 are no element Selafin
        # defines, so the honest report is the width itself.
        raise FileFormatNotSupportedError(
            f"{self.path!r} declares an element width Selafin has no elements for "
            f"(NDP={self.ndp}); 3- and 4-node 2D elements are read, and 6-node prisms are "
            f"refused as layered 3D. A file reaching here is either corrupt or a variant "
            f"this driver has not seen."
        )

    def _read_data(
        self, offsets: list[tuple[int, int]]
    ) -> tuple[np.ndarray, dict[str, ParsedVariable]]:
        """Read the time values eagerly and leave each variable's slab for a loader.

        Returns:
            tuple: the time array, and the parsed variables keyed by name.

        Raises:
            FileFormatNotSupportedError: If the data records do not divide into whole
                timesteps -- a truncated final step.
        """
        n_variables = len(self.names)
        per_step = 1 + n_variables
        remaining = len(offsets) - self.header_records
        if remaining < 0:  # pragma: no cover - a header that read cleanly cannot leave
            # fewer records than it consumed; a defensive guard against an inconsistent
            # index, not reachable from a file that parsed its header.
            return np.empty(0, dtype=np.float64), {}
        n_time = remaining // per_step
        if remaining % per_step:
            warnings.warn(
                f"{self.path!r} ends with a partial timestep ({remaining % per_step} "
                f"trailing record(s)); only the {n_time} whole step(s) are read",
                stacklevel=4,
            )
        times = self._read_times(offsets, n_time, per_step)
        variables = {
            name: self._variable(offsets, index, n_time, per_step, times)
            for index, name in enumerate(self._unique_names())
        }
        boundary = self._boundary_variable()
        if boundary is not None:
            variables[_BOUNDARY_VARIABLE] = boundary
        return times, variables

    def _boundary_variable(self) -> ParsedVariable | None:
        """Return ``IPOBO`` as a static node variable, or ``None`` when it is all zeros.

        A mesh with no boundary numbering -- every entry zero -- gains nothing from an
        array of zeros, so none is attached and the dataset simply has no such variable.
        It is always eager: one int per node is already in hand from the header, so there
        is nothing to defer.

        Returns:
            ParsedVariable | None: the node-located boundary numbering, when the file
                declares one.
        """
        if self.ipobo is None or not np.count_nonzero(self.ipobo):
            return None
        return ParsedVariable(
            location="node",
            shape=(self.npoin,),
            values=self.ipobo.astype(np.int64),
            units=None,
            attributes={
                "long_name": "TELEMAC boundary node numbering (IPOBO)",
                "comment": (
                    "Non-zero on a boundary node. An ordering TELEMAC uses to attach "
                    "boundary conditions, not derivable from the topology, so it is "
                    "carried through a read and written back as the IPOBO record."
                ),
            },
        )

    def _read_times(
        self, offsets: list[tuple[int, int]], n_time: int, per_step: int
    ) -> np.ndarray:
        """Read the one-float time record at the head of each timestep."""
        times = np.empty(n_time, dtype=np.float64)
        for step in range(n_time):
            offset, _length = offsets[self.header_records + step * per_step]
            times[step] = read_payload_floats(
                self.path, offset, 1, endian=self.endian, double=self.double
            )[0]
        return times

    def _unique_names(self) -> list[str]:
        """Return the variable names, de-duplicated so none silently replaces another.

        Two Selafin variables can carry the same 32-character label. Left as-is they
        would collapse to one entry in the variable dict; a positional suffix keeps both.
        """
        seen: dict[str, int] = {}
        unique: list[str] = []
        for raw in self.names:
            label = self._split_name(raw)[0]
            if label in seen:
                seen[label] += 1
                label = f"{label} ({seen[label]})"
            else:
                seen[label] = 1
            unique.append(label)
        return unique

    def _variable(
        self,
        offsets: list[tuple[int, int]],
        index: int,
        n_time: int,
        per_step: int,
        times: np.ndarray,
    ) -> ParsedVariable:
        """Build one node-located variable, lazy or eager per the read mode."""
        _label, unit = self._split_name(self.names[index])
        record_indices = [
            self.header_records + step * per_step + 1 + index for step in range(n_time)
        ]
        slab_offsets = [offsets[record][0] for record in record_indices]
        loader = _SelafinLoader(
            self.path, slab_offsets, self.npoin, endian=self.endian, double=self.double
        )
        shape = (n_time, self.npoin)
        if self.lazy:
            return ParsedVariable(
                location="node",
                shape=shape,
                loader=loader,
                units=unit or None,
                time_values=times,
            )
        return ParsedVariable(
            location="node",
            shape=shape,
            values=loader(),
            units=unit or None,
            time_values=times,
        )

    @staticmethod
    def _split_name(raw: str) -> tuple[str, str]:
        """Split a 32-character record into its name and unit halves by position."""
        return raw[:_NAME_HALF].strip(), raw[_NAME_HALF:_NAME_LENGTH].strip()


class _SelafinLoader:
    """A lazy loader for one variable's whole time series.

    Closes over a path, offsets and widths -- never an open handle, which may be shut by
    the time the load runs (design section 7). Reading it seeks straight to each
    timestep's slab rather than rescanning the file.
    """

    def __init__(
        self,
        path: str,
        offsets: list[int],
        npoin: int,
        *,
        endian: str,
        double: bool,
    ) -> None:
        self.path = path
        self.offsets = offsets
        self.npoin = npoin
        self.endian = endian
        self.double = double

    def __call__(self) -> np.ndarray:
        """Read every timestep's slab and stack them into ``(n_time, npoin)``.

        Raises:
            FileFormatNotSupportedError: If a slab is short or the wrong width -- the same
                failure the eager path reports, translated here so a lazy read surfaces
                the documented type rather than the helper's bare ``ValueError``.
        """
        if not self.offsets:
            return np.empty((0, self.npoin), dtype=np.float64)
        try:
            slabs = [
                read_payload_floats(
                    self.path,
                    offset,
                    self.npoin,
                    endian=self.endian,
                    double=self.double,
                )
                for offset in self.offsets
            ]
        except ValueError as error:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not a readable Selafin file: {error}"
            ) from error
        return np.stack(slabs).astype(np.float64)
