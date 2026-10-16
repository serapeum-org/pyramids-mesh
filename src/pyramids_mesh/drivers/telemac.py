"""TELEMAC Selafin / Serafin driver (task D3).

The first binary driver, and the one that proves the lazy-read path: variable slabs are
left behind a loader that seeks straight to a record, while the topology and the small
time records are read eagerly. Fortran unformatted records with undeclared endianness and
an optional double-precision variant, read through the helpers in ``_binary.py``.

The record layout here was read from MDAL's ``mdal_selafin.cpp``. One detail could not be
checked against a vendor file -- the row-versus-column order of the ``IKLE`` connectivity
block -- so it is read as MDAL reads it (row-major, one element's nodes contiguous) and
called out in the driver's documented limitations. See ``planning/design.md`` section
20.1.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import warnings

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._binary import (
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
# MDAL reads (the TELEMAC docs number them 1-10).
_IPARAM_COUNT = 10
_IPARAM_X_ORIGIN = 2
_IPARAM_Y_ORIGIN = 3
_IPARAM_PLANES = 6
# Interface points: nodes shared between partitions, so nonzero only in a parallel
# sub-domain file and zero in any serial one. This is the single reliable parallel
# marker -- IPARAM[7] (boundary points) is a mesh property, not a parallel flag, and
# MDAL keys on neither (mdal_selafin.cpp reads IPOBO regardless).
_IPARAM_INTERFACE_POINTS = 8
_IPARAM_DATE_FLAG = 9
_DATE_FIELDS = 6


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
        - **A layered 3D file is refused.** ``IPARAM[6]`` is the plane count; 0 and 1 both
          mean 2D (some tools write 1 for 2D), and a value above 1 is a real 3D file whose
          ``NPOIN`` and prism ``IKLE`` cannot be read as a flat 2D mesh without returning
          all planes' nodes under 6-node faces. A full 3D read needs the layered mesh
          model pyramids does not have yet (design section 21.1), so the file is refused
          -- as MDAL refuses it -- rather than yielding a plausible, wrong mesh.
        - **A parallel sub-domain file is refused.** A non-zero interface-point count
          (``IPARAM[8]``) means local numbering within one partition, so the connectivity
          would index a sub-domain rather than the whole mesh. It is keyed on interface
          points alone -- the boundary-point field is a mesh property, nonzero for
          ordinary serial meshes, and keying on it would refuse them.
        - **Selafin variables are node-located, always.** The location is not inferred
          from an array length.
        - **The IKLE row/column order could not be checked against a vendor file.** It is
          read row-major -- one element's ``NDP`` node ids contiguous -- as MDAL reads
          it. A real file that stores it transposed would give a scrambled mesh; this is
          the detail a vendor sample should confirm.
        - There is no CRS in the format, so ``crs`` is ``None``; it is never inferred
          from the coordinate magnitudes.

    Examples:
        - Write the smallest valid Selafin file and read it back. The 1-based ``IKLE``
          arrives 0-based, with the file's base recorded for a writer:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.telemac import (
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
            >>> from pyramids_mesh.drivers.telemac import (
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
    """

    name = "selafin"
    long_name = "TELEMAC Selafin / Serafin"
    suffixes = (".slf", ".srf", ".ser", ".sel")
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh", "read_data", "lazy_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file is a Selafin file, by the endianness probe.

        The first record must declare an 80-byte title in one byte order or the other.
        That probe is cheap and definitive, so it is the whole sniff; a file failing it
        is not Selafin.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the endianness probe succeeds.
        """
        try:
            detect_endianness(path, expect_first_record=_TITLE_LENGTH)
        except ValueError:
            return False
        return True

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
            >>> from pyramids_mesh.drivers.telemac import (
            ...     SelafinDriver,
            ...     build_minimal_fixture_selafin,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.slf")
            ...     build_minimal_fixture_selafin(path)
            ...     dataset = SelafinDriver().read(path)
            >>> dataset.mesh.n_node
            4
            >>> dataset.data_variable_names
            ['WATER DEPTH']
    """
    write_selafin(
        path,
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
) -> None:
    """Write a Selafin file for a fixture, with every variant the tests need.

    Deliberately on the module surface rather than hidden in the tests: it is the only
    writer of this binary format in the package, and the test variants (big-endian,
    double precision, coordinate origin, 3D, parallel) all come from its keyword
    arguments so one implementation of the record layout is exercised.

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
        iparam: The 10-int IPARAM array; defaults to all zeros but ``IPARAM[0] = 1``.

    Raises:
        ValueError: If ``values`` does not match the timestep and variable counts.
    """
    if len(values) != len(times):
        raise ValueError("values must hold one entry per timestep")
    npoin = len(node_x)
    ndp = len(ikle[0])
    float_code = f"{endian}{'f8' if double else 'f4'}"
    marker = _FORMAT_DOUBLE if double else _FORMAT_SINGLE
    title = f"{'pyramids-mesh fixture':<72}{marker:<8}"[:_TITLE_LENGTH]
    params = list(iparam) if iparam is not None else [1] + [0] * (_IPARAM_COUNT - 1)

    def record(payload: bytes) -> bytes:
        word = np.array([len(payload)], dtype=f"{endian}i4").tobytes()
        return word + payload + word

    with open(path, "wb") as handle:
        handle.write(record(title.encode("latin-1")))
        handle.write(
            record(np.array([len(variable_names), 0], f"{endian}i4").tobytes())
        )
        for name in variable_names:
            handle.write(record(f"{name:<32}"[:_NAME_LENGTH].encode("latin-1")))
        handle.write(record(np.array(params, dtype=f"{endian}i4").tobytes()))
        if params[_IPARAM_DATE_FLAG] == 1:
            handle.write(
                record(np.array([2000, 1, 1, 0, 0, 0], f"{endian}i4").tobytes())
            )
        handle.write(
            record(np.array([len(ikle), npoin, ndp, 1], dtype=f"{endian}i4").tobytes())
        )
        flat = np.array(ikle, dtype=f"{endian}i4").reshape(-1)
        handle.write(record(flat.tobytes()))
        handle.write(record(np.zeros(npoin, dtype=f"{endian}i4").tobytes()))
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
            global_attributes={"source": "TELEMAC Selafin", "n_timesteps": len(times)},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_header(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read every header record in order, up to the first timestep.

        Returns:
            tuple: ``(node_x, node_y, faces)`` -- coordinates with the IPARAM origin
                applied, and the 1-based connectivity reshaped to ``(NELEM, NDP)``.

        Raises:
            FileFormatNotSupportedError: If the file declares a parallel sub-domain.
        """
        with open(self.path, "rb") as handle:
            title_bytes = self._read_raw(handle)
            self.double = _FORMAT_DOUBLE in title_bytes.decode("latin-1")
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
                read_record_ints(handle, self.endian, _DATE_FIELDS)
                self.header_records += 1
            self._guard_parallel()
            self._guard_3d()
            dims = read_record_ints(handle, self.endian, 4)
            self.nelem, self.npoin, self.ndp = (
                int(dims[0]),
                int(dims[1]),
                int(dims[2]),
            )
            ikle = read_record_ints(handle, self.endian, self.nelem * self.ndp)
            read_record_ints(handle, self.endian, self.npoin)  # IPOBO, unused
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
        from pyramids_mesh._binary import _read_record

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

        ``IPARAM[6]`` is the plane count. 0 and 1 both mean a 2D mesh -- some tools write
        1 for 2D, which MDAL also accepts -- so only a value above 1 is a real 3D file.
        For one, ``NPOIN`` is the full 3D node count and ``IKLE`` is prism connectivity,
        so reading them as a flat 2D mesh yields all planes' nodes with 6-node "faces":
        plausible and wrong. A full 3D read needs the layered mesh model pyramids does
        not have yet (design section 21.1), and extracting the surface plane cannot be
        done faithfully without a vendor file to verify the plane ordering against, so
        the file is refused -- as MDAL refuses it ("would need additional parsing").
        """
        planes = int(self.iparam[_IPARAM_PLANES])
        if planes > 1:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is a layered 3D Selafin file ({planes} planes); only 2D "
                f"meshes are read. A 3D read needs the layered mesh model pyramids does "
                f"not have yet, and the surface plane cannot be extracted faithfully "
                f"without a vendor file to verify the plane ordering."
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
        if remaining < 0 or per_step == 0:
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
        return times, variables

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
        """Read every timestep's slab and stack them into ``(n_time, npoin)``."""
        if not self.offsets:
            return np.empty((0, self.npoin), dtype=np.float64)
        slabs = [
            read_payload_floats(
                self.path, offset, self.npoin, endian=self.endian, double=self.double
            )
            for offset in self.offsets
        ]
        return np.stack(slabs).astype(np.float64)
