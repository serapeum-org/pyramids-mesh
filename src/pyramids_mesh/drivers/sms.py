"""SMS-family drivers: the 2DM mesh frame and the results-only DAT.

2DM is the reference driver every later one copies (task D1). DAT carries values but no
topology, so it declares ``needs_mesh`` and implements ``read_data`` rather than ``read``
(task D2): a results file is attached to a mesh the caller already has, through
``UgridDataset.attach_data``. See ``planning/design.md`` section 20.1.

The 2DM structure here was verified against the 2DM format specification: a plain-text file
whose first significant line is ``MESH2D``, then one record per line.

``topology_info`` is ``None`` for this driver -- a 2DM file has no NetCDF variables to
name, and nothing is synthesised that ``to_file`` would not write.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

# Imported from a private module because there is no public one: as of pyramids-gis
# 0.68.0 nothing named *Error is re-exported from `pyramids` or `pyramids.base`, while
# the published driver contract ("Exceptions -- reuse, do not invent") tells a driver to
# raise exactly this type. ADR-0005 requires a local workaround to name the upstream
# gap and be removed when it closes: the ask upstream is a public export of
# FileFormatNotSupportedError (and a public eager/lazy MeshVariable constructor, which
# _base.py needs for the same reason). Until then a patch release renaming the private
# module would break this import, so `tests/test_package.py` keeps a canary on it.
from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION, MeshVariable, UgridDataset

from pyramids_mesh._base import BaseDriver, ParsedMesh, ParsedVariable
from pyramids_mesh._text import iter_records, native_id_map

_HEADER = "MESH2D"

# The node count comes from the keyword's second character (read as
# ``line[1]`` and asserts 3, 4 or 6), but the full keyword is matched instead: E2L and
# E3L carry a digit too and are line elements, not faces.
_FACE_NODE_COUNTS = {"E3T": 3, "E4Q": 4, "E6T": 6}

# The whole file is refused on these rather than guessing a node mapping. E3L is a 1D
# element that is rejected for the same reason.
_UNSUPPORTED_ELEMENTS = ("E3L", "E8Q", "E9Q")

_LINE_ELEMENT = "E2L"
_NODE = "ND"
_MATERIALS_TAG = "NUM_MATERIALS_PER_ELEM"

# An upper bound on the declared material columns. Each one allocates an array of
# `n_face` floats and a variable, and the count comes from an untrusted file -- reading
# other people's files is this package's whole job -- so one corrupt line must not be
# able to ask for thousands. The practical limit is single digits.
_MAX_MATERIAL_COLUMNS = 16

# Kept in step with ``iter_records``' default: the header sniff and the parser must
# skip the same lines, or one accepts a file the other refuses.
_COMMENT = "#"

# ``can_open`` must not read the whole file, so the header sniff reads in blocks and
# stops at the first significant line. A single fixed prefix was not enough: a file
# whose comment header ran past it was declined by ``can_open`` and then read fine by
# ``read``, which is the asymmetry the sniff exists to avoid. The limit still bounds the
# work for a file that is all comments, or binary -- which decodes to replacement
# characters rather than raising.
_SNIFF_BLOCK = 4096
_SNIFF_LIMIT = 256 * 1024

# -- DAT cards (task D2) ----------------------------------------------------------
#
# The new dialect opens with DATASET. The old one has no such line and is recognised by
# any of `_OLD_DIALECT_CARDS` appearing in the sniffed prefix, and read by a separate
# parser, and the two differ in TS arity (below).
_DATASET = "DATASET"
_OLD_DIALECT_CARDS = ("SCALAR", "VECTOR", "TS", "TIMEUNITS")
# The old-format group type is keyed on this word appearing in the header line.
_VECTOR_TOKEN = "VECTOR"

_BEGIN_SCALAR = "BEGSCL"
_BEGIN_VECTOR = "BEGVEC"
_END_DATASET = "ENDDS"
_OBJTYPE = "OBJTYPE"
_NODE_COUNT = "ND"
_CELL_COUNT = "NC"
_NAME = "NAME"
_TIME_UNITS = "TIMEUNITS"
_REFERENCE_TIME = "RT_JULIAN"
_TIMESTEP = "TS"

# Cards that are only meaningful inside an open BEGSCL/BEGVEC group. Each one outside a
# group is an error: a NAME or TS with no group to attach to means the file is
# malformed, and guessing a group would invent structure.
_GROUP_ONLY_CARDS = (_NAME, _TIME_UNITS, _TIMESTEP)

# `TS <istat> <time>` carries at least these many fields per dialect. The old dialect
# drops a field, so a parser written only for the new one reads the *time* as the active
# flag and silently yields a one-timestep series -- the trap this split exists to avoid.
_TS_FIELDS_NEW = 2
_TS_FIELDS_OLD = 1

# The smallest valid DAT: one scalar group, two timesteps, four node values each. It
# matches `build_minimal_fixture`'s 2DM -- four nodes, two cells -- so the two can be
# attached to each other in a test without a count mismatch.
_MINIMAL_DAT_FIXTURE = """DATASET
OBJTYPE mesh2d
BEGSCL
ND 4
NC 2
NAME depth
TIMEUNITS SECONDS
TS 0 0.0
0.5
0.6
0.7
0.8
TS 0 3600.0
1.5
1.6
1.7
1.8
ENDDS
"""

# The smallest valid 2DM: the unit square split into two counter-clockwise triangles.
_MINIMAL_FIXTURE = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 1.0 1.0 0.0
ND 4 0.0 1.0 0.0
E3T 1 1 2 3 1
E3T 2 1 3 4 1
"""


class TwoDmDriver(BaseDriver):
    """Reader for the SMS/TUFLOW 2DM mesh-frame format.

    Reads nodes, triangular/quadrilateral/six-node faces and 1D line elements. Node
    elevation becomes a ``node_z`` variable and the material id a face-located
    ``material`` variable, since the pyramids mesh model is planar and carries no
    per-node elevation coordinate.

    Documented limitations:
        - ``E3L``, ``E8Q`` and ``E9Q`` are refused outright;
          treating a mid-side or centre node as a corner would silently distort the
          element.
        - ``E6T`` is read as a real six-node face, not demoted to its corner triangle.
        - A file may hold a surface mesh, an ``E2L`` network, or both. They are named
          separately (``mesh_names`` -> ``["mesh2d", "network1d"]``), so one read
          returns one topology and the caller picks with ``mesh_name``. Reading either
          half warns about what the other leaves behind. ``E2L`` elements are a
          separate 1D topology, never folded into the surface's edge connectivity.
        - A 1D read carries no ``node_z``, variables or global attributes:
          ``Network1d`` has no parameter for them. Each is named in a warning rather
          than dropped quietly. That is a ``pyramids-gis`` limitation to fix upstream.
        - ``node_z`` is omitted entirely when no ``ND`` record carries a z column,
          rather than fabricated as zeros. A file whose ``ND`` records disagree about
          that column is refused instead: a fabricated ``0.0`` cannot be told from a
          real zero datum, so there is nothing safe to fill the gaps with.
        - ``NUM_MATERIALS_PER_ELEM`` is accepted from 1 to 16 columns. Each column
          allocates a face-length array and a variable from a count a file supplied, so
          one corrupt line must not be able to ask for thousands; the practical
          limit is single digits.
        - 2DM carries no CRS, so ``crs`` is always ``None``; it is never guessed from
          the coordinate magnitudes.

    Examples:
        - Write the smallest valid 2DM and read it back. The file numbers its nodes
          from 1, and the connectivity arrives 0-based with the file's base still
          recorded for a writer to restore:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import (
            ...     TwoDmDriver,
            ...     build_minimal_fixture,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.2dm")
            ...     build_minimal_fixture(path)
            ...     dataset = TwoDmDriver().read(path)
            >>> dataset.mesh.n_node, dataset.mesh.n_face
            (4, 2)
            >>> dataset.mesh.face_node_connectivity.data.tolist()
            [[0, 1, 2], [0, 2, 3]]
            >>> dataset.mesh.face_node_connectivity.original_start_index
            1

        - Elevation and the material id come back as variables, because the mesh
          model is planar and carries neither as a coordinate:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import (
            ...     TwoDmDriver,
            ...     build_minimal_fixture,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "domain.2dm")
            ...     build_minimal_fixture(path)
            ...     dataset = TwoDmDriver().read(path)
            >>> sorted(dataset.data_variable_names)
            ['material', 'node_z']
            >>> dataset.get("node_z").location
            'node'
            >>> dataset.get("material").data.tolist()
            [1.0, 1.0]

        - Dispatch checks the suffix and sniffs the header, so the right content
          under a suffix this driver does not claim is still declined:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import (
            ...     TwoDmDriver,
            ...     build_minimal_fixture,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     claimed = os.path.join(directory, "domain.2dm")
            ...     foreign = os.path.join(directory, "domain.txt")
            ...     build_minimal_fixture(claimed)
            ...     build_minimal_fixture(foreign)
            ...     driver = TwoDmDriver()
            ...     (driver.can_open(claimed), driver.can_open(foreign))
            (True, False)

        - A file whose ``ND`` records carry only x and y gets no ``node_z`` at all,
          rather than a column of zeros a consumer could not tell from a real datum:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import TwoDmDriver
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "flat.2dm")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         print("MESH2D", file=handle)
            ...         print("ND 1 0.0 0.0", file=handle)
            ...         print("ND 2 1.0 0.0", file=handle)
            ...         print("ND 3 1.0 1.0", file=handle)
            ...         print("E3T 1 1 2 3 1", file=handle)
            ...     dataset = TwoDmDriver().read(path)
            >>> sorted(dataset.data_variable_names)
            ['material']
    """

    name = "2dm"
    long_name = "SMS/TUFLOW 2DM mesh"
    suffixes = (".2dm",)
    api_version = MESH_DRIVER_API_VERSION
    # `read_data` and `multi_mesh` are declarative upstream: they describe the driver in
    # `info(drivers=True)` and gate nothing, so neither implies a method. `read_data`
    # here means this format carries values -- material and elevation -- that `read`
    # returns with the topology; there is deliberately no `read_data()`. Only a
    # results-only format declares `needs_mesh` and implements that method.
    capabilities = frozenset(
        {"read_mesh", "read_data", "multi_mesh", "write_mesh", "write_data"}
    )

    # Face node counts emit their keyword; the inverse of `_FACE_NODE_COUNTS`, used by
    # the writer to choose E3T/E4Q/E6T from a face's real (non-fill) vertex count.
    _COUNT_TO_KEYWORD = {3: "E3T", 4: "E4Q", 6: "E6T"}

    def mesh_names(self, path: str) -> list[str]:
        """List the topologies the file holds, in surface-then-network order.

        A 2DM may carry a surface mesh (``E3T``/``E4Q``/``E6T``), a 1D network
        (``E2L``), or both. Naming them separately is what lets a caller pick one with
        ``mesh_name``, which is the only way to reach the 1D half of a mixed file.

        Args:
            path: The 2DM file to inspect.

        Returns:
            list[str]: ``["mesh2d"]``, ``["network1d"]``, or both. A file whose
                elements cannot be read at all reports the surface name, leaving the
                error to ``read``.

        Note:
            The ``MESH2D`` header is not checked here, so this answers for any readable
            text file. Dispatch asks ``can_open`` first, which does check it; a caller
            holding a driver instance should do the same before trusting the result.

        Examples:
            - A file carrying both an ``E3T`` face and an ``E2L`` line element names
              two topologies, and either name may then be passed to ``read``:

                >>> import os
                >>> import tempfile
                >>> import warnings
                >>> from pyramids_mesh.drivers.sms import TwoDmDriver
                >>> with tempfile.TemporaryDirectory() as directory:
                ...     path = os.path.join(directory, "mixed.2dm")
                ...     with open(path, "w", encoding="utf-8") as handle:
                ...         print("MESH2D", file=handle)
                ...         print("ND 1 0.0 0.0 0.0", file=handle)
                ...         print("ND 2 1.0 0.0 0.0", file=handle)
                ...         print("ND 3 1.0 1.0 0.0", file=handle)
                ...         print("ND 4 0.0 1.0 0.0", file=handle)
                ...         print("E3T 1 1 2 3 1", file=handle)
                ...         print("E2L 1 3 4 1", file=handle)
                ...     driver = TwoDmDriver()
                ...     names = driver.mesh_names(path)
                ...     with warnings.catch_warnings():
                ...         warnings.simplefilter("ignore")
                ...         surface = driver.read(path)
                ...         channels = driver.read(path, mesh_name="network1d")
                >>> names
                ['mesh2d', 'network1d']
                >>> surface.mesh.face_node_connectivity.data.tolist()
                [[0, 1, 2]]
                >>> channels.n_edge
                1
                >>> channels.edge_node_connectivity.data.tolist()
                [[2, 3]]

            - A file with faces and no ``E2L`` names only the surface, which is the
              answer a caller needs before asking for a topology that is not there:

                >>> import os
                >>> import tempfile
                >>> from pyramids_mesh.drivers.sms import (
                ...     TwoDmDriver,
                ...     build_minimal_fixture,
                ... )
                >>> with tempfile.TemporaryDirectory() as directory:
                ...     path = os.path.join(directory, "domain.2dm")
                ...     build_minimal_fixture(path)
                ...     TwoDmDriver().mesh_names(path)
                ['mesh2d']
        """
        has_faces = False
        has_lines = False
        try:
            for keyword, _ in iter_records(path, comment=_COMMENT):
                upper = keyword.upper()
                if upper in _FACE_NODE_COUNTS:
                    has_faces = True
                elif upper == _LINE_ELEMENT:
                    has_lines = True
                if has_faces and has_lines:
                    # Nothing further can change the answer, and this spares a full
                    # scan of a large mesh for a metadata question.
                    break
        except OSError:
            return [self._default_mesh_name]
        names = []
        if has_faces:
            names.append(self._default_mesh_name)
        if has_lines:
            names.append(self.network_mesh_name)
        return names or [self._default_mesh_name]

    def _sniff(self, path: str) -> bool:
        """Report whether the first significant line is the ``MESH2D`` header.

        Reads a bounded prefix and decodes it leniently, so a binary or empty file
        answers ``False`` instead of raising. A byte-order mark is dropped, and blocks
        are read until the first significant line or ``_SNIFF_LIMIT`` bytes, so a file
        whose comment header runs long is not declined for that alone.

        Each block is split at its last newline and the remainder carried into the next
        one, so only a line that has been read to its end is ever judged; the leftover
        is judged at end of file, where it is a whole line after all. Deciding on the
        buffer's tail instead meant deciding on a line still arriving: with a block edge
        falling inside the first significant line, that declined valid files and
        accepted a ``MESH2DX`` truncated to exactly ``MESH2D``.

        Blank and comment lines are skipped, exactly as
        :func:`~pyramids_mesh._text.iter_records` skips them while parsing. The two must
        agree: a sniff stricter than the parser would decline a file this driver can
        read, and dispatch would then hand it to nobody.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the first significant line starts with ``MESH2D``.
        """
        with open(path, "rb") as handle:
            pending = b""
            consumed = 0
            while consumed < _SNIFF_LIMIT:
                block = handle.read(_SNIFF_BLOCK)
                if not block:
                    # End of file, so whatever is left is a whole line after all.
                    return _header_verdict(pending) is True
                consumed += len(block)
                # Only ever judge complete lines. Deciding on the tail of the buffer
                # means deciding on a line still being read: with a block edge inside
                # the first significant line that declined valid files and accepted a
                # MESH2DX truncated to exactly MESH2D. Splitting on the last newline
                # also keeps the work linear, and is a safe character boundary because
                # no UTF-8 continuation byte can be 0x0A.
                complete, newline, pending = (pending + block).rpartition(b"\n")
                if not newline:
                    continue
                verdict = _header_verdict(complete)
                if verdict is not None:
                    return verdict
        return False

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a 2DM file into a :class:`~pyramids_mesh._base.ParsedMesh`.

        Node and element ids are collected as the file gives them and resolved to
        contiguous indices through :func:`~pyramids_mesh._text.native_id_map` in a
        second pass: 2DM ids may have gaps and may be out of
        order, so ``id == index + 1`` would mis-wire every face of a renumbered file.

        Args:
            path: The 2DM file to read.
            lazy: Ignored -- a 2DM file is plain text and read eagerly.

        Returns:
            ParsedMesh: Nodes, faces or edges, and the material/elevation variables.

        Raises:
            FileFormatNotSupportedError: If the ``MESH2D`` header is missing, an
                unsupported element type appears, the file declares no elements, a
                record is short of the fields its keyword requires, a numeric field is
                not a number, ``NUM_MATERIALS_PER_ELEM`` declares a column count
                outside 1 to 16, or the ``ND`` records disagree about whether they
                carry a z column.
            ValueError: If an element references a node id the file never defined, or
                two nodes, two faces or two line elements share an id.
        """
        scan = _Scan()
        scan.run(path)
        return scan.build(path)

    def write(self, dataset, path: str, **options) -> None:
        """Write a mesh back to a 2DM file.

        Emits ``MESH2D``, one ``ND`` per node and one ``E3T``/``E4Q``/``E6T`` per face,
        choosing the element keyword from the face's **real** vertex count (fill ``-1``
        entries ignored). The connectivity base recorded on the mesh is re-applied, so a
        file read 1-based is written back 1-based.

        Node elevation is taken from a node-located ``node_z`` variable when present, and
        the material column from a face-located ``material`` variable; each defaults to
        ``0`` when absent. Coordinates are written at full ``float64`` precision by
        default, because a fixed number of decimals silently loses position for projected
        coordinates in the millions -- a ``%.6f`` easting of 500000.123456789 comes back
        wrong by metres.

        Args:
            dataset: The :class:`~pyramids.netcdf.ugrid.UgridDataset` to write.
            path: The output ``.2dm`` path.
            **options: ``precision`` sets the coordinate format: an int number of
                significant digits, or the default ``"repr"`` for an exact float64
                round-trip.

        Raises:
            FileFormatNotSupportedError: If a face has a vertex count 2DM has no element
                for (only 3, 4 and 6 are writable), or the dataset carries no faces.
        """
        mesh = dataset.mesh
        connectivity = mesh.face_node_connectivity
        if connectivity is None or connectivity.data is None:
            raise FileFormatNotSupportedError(
                f"cannot write {path!r}: the dataset has no face connectivity to write "
                f"as a 2DM mesh"
            )
        faces = np.asarray(connectivity.data)
        base = connectivity.original_start_index
        if base is None:
            base = 1
        node_x = np.asarray(mesh.node_x, dtype=np.float64)
        node_y = np.asarray(mesh.node_y, dtype=np.float64)
        node_z = self._writer_column(dataset, "node_z", len(node_x))
        material = self._writer_column(dataset, "material", len(faces))
        fmt = _coordinate_formatter(options.get("precision", "repr"))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{_HEADER}\n")
            for index in range(len(node_x)):
                z = 0.0 if node_z is None else float(node_z[index])
                handle.write(
                    f"{_NODE} {index + base} {fmt(node_x[index])} "
                    f"{fmt(node_y[index])} {fmt(z)}\n"
                )
            for row_index, row in enumerate(faces):
                vertices = [int(v) for v in row if int(v) != -1]
                keyword = self._COUNT_TO_KEYWORD.get(len(vertices))
                if keyword is None:
                    raise FileFormatNotSupportedError(
                        f"cannot write a {len(vertices)}-vertex face to 2DM; only 3, 4 "
                        f"and 6 node faces have an element keyword"
                    )
                ids = " ".join(str(v + base) for v in vertices)
                mat = 0 if material is None else int(material[row_index])
                handle.write(f"{keyword} {row_index + base} {ids} {mat}\n")

    @staticmethod
    def _writer_column(dataset, name: str, expected: int):
        """Return a variable's values for the writer, or ``None`` when it is absent.

        The length is checked so a stale variable cannot write a column that does not
        line up with the nodes or faces; a mismatch drops it rather than writing garbage.
        """
        if name not in dataset.data_variable_names:
            return None
        values = np.asarray(dataset.get(name).data)
        if values.shape != (expected,):
            return None
        return values


class DatDriver(BaseDriver):
    """Reader for the SMS ASCII DAT results format.

    DAT carries values and no topology -- it is a results-only format --
    so this driver declares ``needs_mesh`` and implements ``read_data`` instead of
    ``read``. The values attach to a mesh the caller already holds, through
    ``UgridDataset.read_file("domain.2dm").attach_data("depth.dat")``.

    Two dialects are supported. The **new** one opens with a ``DATASET`` card; the
    **old** one has no such line and is recognised by ``SCALAR``, ``VECTOR``, ``TS`` or
    ``TIMEUNITS`` appearing in the file. They differ in ``TS`` arity, which is why the
    distinction is tracked rather than ignored: a parser written only for the new
    dialect reads an old file's *time* as its active flag and silently produces a
    one-timestep series.

    Documented limitations:
        - **The element-count check is the only thing tying a DAT to its mesh.** Nothing
          in the file identifies the mesh it belongs to, so ``ND`` is checked against
          the dataset's node count and ``NC`` against its face count before any value
          is read. A file declaring neither is refused rather than hoped about. Two
          meshes with coincidentally equal counts stay indistinguishable -- a property
          of the format, not of this driver.
        - A ``BEGVEC`` group yields **two scalar variables**, ``<name>_x`` and
          ``<name>_y``, rather than one two-component variable: ``MeshVariable`` carries
          a location and a shape over elements and time, with no component axis for a
          vector. Both keep the group's name as a prefix so they stay recognisable as
          one quantity.
        - Values are **node-located** when ``ND`` is declared, which is what SMS
          writes. A file declaring only ``NC`` is read as face-located. With both
          declared the group is node-located, the SMS convention for
          ``BEGSCL``/``BEGVEC``; ``OBJTYPE`` is recorded but no SMS writer puts a
          location in it.
        - The per-step active mask becomes a **separate face-located variable**
          ``<name>_active``, never folded into the values as NaN (design section 21.2).
          It is deliberately not attached through ``ParsedVariable.active``: that gives
          the sibling its parent's location and shape, and a DAT active block counts
          *cells* while a node-located group counts *nodes*.
        - **The active block is read right after the ``TS`` card, before the values.**
          This is the one part of the layout no vendor file was
          available to confirm, so a mismatch is made loud rather than silent -- the
          flags must all be ``0`` or ``1``, and a block of real values in their place
          is refused. A file whose node and cell counts are equal defeats that check,
          which is recorded here because nothing in the data can catch it.
        - ``RT_JULIAN`` is carried as a ``reference_time`` entry in **each variable's
          attributes**. It cannot reach the dataset's global attributes: ``attach_data``
          merges the variables a driver returns and keeps its own ``global_attributes``
          untouched, so a driver has no way to add one.
        - DAT carries no CRS and no topology, so neither is produced here.

    Examples:
        - Attach a DAT to the mesh it belongs to. Each group arrives as a variable
          carrying the file's time axis:

            >>> import os
            >>> import tempfile
            >>> from pyramids.netcdf.ugrid import UgridDataset
            >>> from pyramids_mesh.drivers.sms import (
            ...     build_minimal_fixture,
            ...     build_minimal_fixture_dat,
            ... )
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     mesh_path = os.path.join(directory, "domain.2dm")
            ...     dat_path = os.path.join(directory, "depth.dat")
            ...     build_minimal_fixture(mesh_path)
            ...     build_minimal_fixture_dat(dat_path)
            ...     attached = UgridDataset.read_file(mesh_path).attach_data(dat_path)
            >>> sorted(attached.data_variable_names)
            ['depth', 'material', 'node_z']
            >>> depth = attached.get("depth")
            >>> depth.location, depth.shape
            ('node', (2, 4))
            >>> depth.data.tolist()
            [[0.5, 0.6, 0.7, 0.8], [1.5, 1.6, 1.7, 1.8]]

        - ``read`` is not how a results-only format is reached, and says so rather than
          failing obscurely:

            >>> from pyramids_mesh.drivers.sms import DatDriver
            >>> try:
            ...     DatDriver().read("depth.dat")
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            dat carries values but no mesh, so it has no read()

        - A count disagreeing with the mesh is refused, reporting both numbers, because
          nothing else in the file ties it to a topology:

            >>> import os
            >>> import tempfile
            >>> from pyramids.netcdf.ugrid import UgridDataset
            >>> from pyramids_mesh.drivers.sms import build_minimal_fixture
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     mesh_path = os.path.join(directory, "domain.2dm")
            ...     dat_path = os.path.join(directory, "wrong.dat")
            ...     build_minimal_fixture(mesh_path)
            ...     with open(dat_path, "w", encoding="utf-8") as handle:
            ...         print("DATASET", file=handle)
            ...         print("BEGSCL", file=handle)
            ...         print("ND 9", file=handle)
            ...         print("NAME depth", file=handle)
            ...         print("ENDDS", file=handle)
            ...     try:
            ...         UgridDataset.read_file(mesh_path).attach_data(dat_path)
            ...     except ValueError as error:
            ...         print(str(error).split(": ")[-1])
            declares ND 9 but the mesh has 4 nodes
    """

    name = "dat"
    long_name = "SMS ASCII DAT results"
    suffixes = (".dat",)
    api_version = MESH_DRIVER_API_VERSION
    # `needs_mesh` is what makes `read_data` the entry point rather than `read`: the
    # format carries no topology, so the mesh has to come from the caller.
    capabilities = frozenset({"needs_mesh", "read_data"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file looks like either DAT dialect.

        The new dialect is a ``DATASET`` first significant line. The old one has no such
        card, so a bounded prefix is scanned for the cards that mark the old format.
        Scanning a prefix keeps ``can_open`` cheap on a large results
        file, at the cost of missing a dialect marker buried past the limit -- the same
        bound the 2DM sniff accepts.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when either dialect is recognised.
        """
        with open(path, "rb") as handle:
            prefix = handle.read(_SNIFF_LIMIT)
        significant = [
            stripped
            for line in prefix.decode("utf-8-sig", errors="replace").splitlines()
            if (stripped := line.strip()) and not stripped.startswith(_COMMENT)
        ]
        if not significant:
            return False
        if significant[0].split()[0].upper() == _DATASET:
            return True
        # The old dialect has no header card, so it is recognised by the cards it must
        # contain. Matching the first token only keeps a file that merely mentions one
        # of those words in a comment or a dataset name from qualifying.
        return any(
            line.split()[0].upper() in _OLD_DIALECT_CARDS for line in significant
        )

    def read(
        self, path: str, *, mesh_name: str | None = None, lazy: bool = True
    ) -> UgridDataset:
        """Refuse the mesh-reading entry point, naming the one to use instead.

        Args:
            path: The results file, named in the message.
            mesh_name: Unused.
            lazy: Unused.

        Returns:
            UgridDataset: Never. The signature matches the driver protocol so the
                refusal is a clear message rather than a ``TypeError``.

        Raises:
            ValueError: Always. A results-only format has no topology to return, and an
                empty mesh would be a plausible-looking wrong answer.
        """
        raise ValueError(
            f"{self.name} carries values but no mesh, so it has no read(); attach it to "
            f"a mesh you already have with "
            f"UgridDataset.read_file(mesh_path).attach_data({path!r})"
        )

    def read_data(
        self, path: str, dataset: UgridDataset, *, lazy: bool = True
    ) -> dict[str, MeshVariable]:
        """Read a DAT file's groups as variables for a mesh the caller supplies.

        The declared element counts are checked against ``dataset`` before any value is
        read; the class's documented limitations explain why that check is the whole
        safeguard.

        Args:
            path: The DAT file to read.
            dataset: The mesh the values will attach to, for the element-count check.
            lazy: Accepted for the driver protocol and ignored. DAT is plain text with
                no record index to seek, so there is nothing to defer, and the driver
                does not declare ``lazy_data``.

        Returns:
            dict[str, MeshVariable]: One variable per scalar group, two per vector
                group, plus one ``<name>_active`` face variable per group that carries
                an active mask. The old dialect carries a single group named after the
                file, read node-located with no active block (the old-format model).

        Raises:
            FileFormatNotSupportedError: If the file matches neither dialect, a card
                appears outside the group it belongs to, a group opens while one is open
                or ends when none is, a value block is short or non-numeric, two groups
                would produce the same variable name, or an active block holds anything
                but ``0`` and ``1``.
            ValueError: If a declared element count disagrees with ``dataset``, or the
                file declares neither count.
        """
        n_node = int(dataset.mesh.n_node)
        n_face = int(dataset.mesh.n_face)
        if _first_significant_card(path) == _DATASET:
            scan = _DatScan(path, n_node=n_node, n_face=n_face)
            scan.run()
            variables = scan.build()
        else:
            # No DATASET header: the old dialect, read by a separate
            # parser -- one group whose type comes from the header line, ND, and bare
            # TS <time> blocks, with no BEGSCL/ENDDS/NC/NAME cards.
            variables = _DatOldScan(path, n_node=n_node).build()
        return self._build_data_variables(variables, dataset.mesh_name)


def build_minimal_fixture_dat(path: str) -> None:
    """Write the smallest valid DAT file, matching the 2DM fixture's element counts.

    Named per driver rather than ``build_minimal_fixture`` because this module holds two
    drivers; the invariant harness looks for ``build_minimal_fixture_<driver name>``
    first and falls back to the shared name.

    Args:
        path: Where to write the file.

    Examples:
        - One scalar group over two timesteps, with a value per node of the 2DM
          fixture's four:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import build_minimal_fixture_dat
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.dat")
            ...     build_minimal_fixture_dat(path)
            ...     with open(path, encoding="utf-8") as handle:
            ...         lines = handle.read().splitlines()
            >>> lines[0]
            'DATASET'
            >>> len([line for line in lines if line.startswith("TS ")])
            2
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_MINIMAL_DAT_FIXTURE)


def _first_significant_card(path: str) -> str | None:
    """Return the upper-cased first significant card of a DAT file, or ``None``.

    The dialect turns on this alone: a ``DATASET`` first line is
    the new format, anything else is the old one. Blank and comment lines are skipped, so
    a leading comment does not change the verdict.
    """
    for keyword, _fields in iter_records(path, comment=_COMMENT):
        return keyword.upper()
    return None


class _DatOldScan:
    """The pre-DATASET DAT dialect: one group from the header, then TS value blocks.

    This dialect uses a parser separate from the new format: the
    first line sets the single group's type (vector if it contains ``VECTOR``, else
    scalar), ``ND`` gives the node count, and each ``TS <time>`` starts a timestep whose
    values follow one row per node. There is no ``BEGSCL``/``BEGVEC``, no ``ENDDS``, no
    ``NC`` and no ``NAME`` -- the group is named after the file, by convention.

    The values are node-located, and the declared ``ND`` is checked against the mesh
    before any value is read, the same safeguard the new dialect applies: a results file
    carries nothing else tying it to a mesh.
    """

    def __init__(self, path: str, *, n_node: int) -> None:
        self.path = path
        self.n_node = n_node
        self.vector = False
        self.declared_nd: int | None = None
        self.time_units: str | None = None
        self.times: list[float] = []
        self.values: list[np.ndarray] = []

    def build(self) -> dict[str, ParsedVariable]:
        """Read the file and return its variables, keyed by name.

        Raises:
            FileFormatNotSupportedError: On a malformed or short record, or a file with
                no ``TS`` block (no values to attach).
            ValueError: If ``ND`` is absent or disagrees with the mesh node count.
        """
        records = iter_records(self.path, comment=_COMMENT)
        try:
            header, _ = next(records)
        except StopIteration:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is empty: an old-format DAT starts with a type header"
            ) from None
        # The type is keyed on the word VECTOR appearing anywhere in the header line.
        self.vector = _VECTOR_TOKEN in header.upper()
        for keyword, fields in records:
            self._consume(keyword.upper(), fields, records)
        self._check_nd()
        if not self.times:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has no {_TIMESTEP} block, so it carries no values"
            )
        return self._variables()

    def _consume(
        self,
        upper: str,
        fields: list[str],
        records: Iterator[tuple[str, list[str]]],
    ) -> None:
        """Dispatch one old-format card; an unknown card is ignored.

        ``SCALAR``/``VECTOR`` are type hints the header already settled, so they do
        nothing here.
        """
        if upper == _NODE_COUNT:
            self.declared_nd = self._single_int(upper, fields)
        elif upper == _TIME_UNITS:
            _require_fields(upper, fields, 1, self.path)
            self.time_units = fields[0]
        elif upper == _TIMESTEP:
            self._read_timestep(fields, records)

    def _single_int(self, upper: str, fields: list[str]) -> int:
        """Parse a one-field integer card such as ``ND``."""
        _require_fields(upper, fields, 1, self.path)
        return _as_int(fields[0], upper, self.path)

    def _check_nd(self) -> None:
        """Check the declared node count against the mesh, before any value is used.

        Raises:
            ValueError: If ``ND`` is absent (nothing ties the file to a mesh) or does not
                match the mesh's node count.
        """
        if self.declared_nd is None:
            raise ValueError(
                f"{self.path!r} declares no {_NODE_COUNT}, so nothing can be checked "
                f"against the mesh; attaching it unchecked would be a guess"
            )
        if self.declared_nd != self.n_node:
            raise ValueError(
                f"{self.path!r} does not match this mesh: declares {_NODE_COUNT} "
                f"{self.declared_nd} but the mesh has {self.n_node} nodes"
            )

    def _read_timestep(
        self, fields: list[str], records: Iterator[tuple[str, list[str]]]
    ) -> None:
        """Read one ``TS <time>`` card and the node value rows that follow it.

        The node count is checked before the first value is read. The old dialect carries
        no active flag, so a block of values always follows
        the time directly.
        """
        self._check_nd()
        _require_fields(_TIMESTEP, fields, _TS_FIELDS_OLD, self.path)
        self.times.append(_as_float(fields[0], _TIMESTEP, self.path))
        columns = 2 if self.vector else 1
        block = np.empty((self.n_node, columns), dtype=np.float64)
        for row in range(self.n_node):
            try:
                keyword, extra = next(records)
            except StopIteration:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} ends inside a {_TIMESTEP} block: expected "
                    f"{self.n_node} rows, found {row}"
                ) from None
            tokens = [keyword, *extra]
            if len(tokens) < columns:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short {_TIMESTEP} row: expected {columns} "
                    f"value(s), found {len(tokens)}"
                )
            for column in range(columns):
                block[row, column] = _as_float(tokens[column], _TIMESTEP, self.path)
        self.values.append(block)

    def _variables(self) -> dict[str, ParsedVariable]:
        """Build the single group's variables, named after the file.

        A vector group becomes ``<name>_x``/``<name>_y``, as in the new dialect; a scalar
        group is the bare name. The name is the file stem, the convention for an
        old-format group (it has no ``NAME`` card).
        """
        import os

        base = os.path.splitext(os.path.basename(self.path))[0] or "dataset"
        times = np.array(self.times, dtype=np.float64)
        stacked = np.stack(self.values)
        attributes = (
            {"time_units": self.time_units} if self.time_units is not None else {}
        )
        if self.vector:
            return {
                f"{base}_{suffix}": ParsedVariable(
                    location="node",
                    shape=stacked[:, :, offset].shape,
                    values=stacked[:, :, offset],
                    time_values=times,
                    attributes=dict(attributes),
                )
                for offset, suffix in ((0, "x"), (1, "y"))
            }
        return {
            base: ParsedVariable(
                location="node",
                shape=stacked[:, :, 0].shape,
                values=stacked[:, :, 0],
                time_values=times,
                attributes=dict(attributes),
            )
        }


def build_minimal_fixture(path: str) -> None:
    """Write the smallest valid 2DM file, for the cross-driver invariant harness.

    This is test support on the installed surface, deliberately: the harness discovers
    it by name on each driver's module, which is what lets a new driver join the
    invariant suite by being registered rather than by anyone writing new tests. It
    writes the same seven lines as the committed ``tests/data/minimal.2dm``, and a test
    asserts the two match so neither can drift.

    Args:
        path: Where to write the file.

    Examples:
        - The fixture is the unit square split into two counter-clockwise
          triangles -- four nodes and two elements, the smallest thing that is
          still a mesh:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.sms import build_minimal_fixture
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.2dm")
            ...     build_minimal_fixture(path)
            ...     with open(path, encoding="utf-8") as handle:
            ...         lines = handle.read().splitlines()
            >>> lines[0]
            'MESH2D'
            >>> len([line for line in lines if line.startswith("ND")])
            4
            >>> [line for line in lines if line.startswith("E3T")]
            ['E3T 1 1 2 3 1', 'E3T 2 1 3 4 1']
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_MINIMAL_FIXTURE)


class _Scan:
    """One 2DM file's records, collected in file order then resolved to indices."""

    def __init__(self) -> None:
        self.node_ids: list[int] = []
        self.node_x: list[float] = []
        self.node_y: list[float] = []
        self.node_z: list[float] = []
        self.face_ids: list[int] = []
        self.face_vertices: list[list[int]] = []
        self.face_keywords: list[str] = []
        self.face_extras: list[list[str]] = []
        self.edge_ids: list[int] = []
        self.edge_vertices: list[list[int]] = []
        self.materials_per_element = 1
        # Whether any ND record actually carried a z column. A file without one gets no
        # node_z at all rather than a column of zeros, which a consumer could not tell
        # from a real zero datum.
        self.saw_elevation = False
        self.missing_elevation = False

    def run(self, path: str) -> None:
        """Stream the file once, collecting records without resolving any ids.

        Args:
            path: The 2DM file to read.

        Raises:
            FileFormatNotSupportedError: On a missing header or an unsupported element.
        """
        header_seen = False
        for keyword, fields in iter_records(path, comment=_COMMENT):
            upper = keyword.upper()
            if not header_seen:
                if upper != _HEADER:
                    raise FileFormatNotSupportedError(
                        f"{path!r} is not a 2DM file: expected a {_HEADER} header, "
                        f"found {keyword!r}"
                    )
                header_seen = True
                continue
            self._consume(upper, fields, path)
        if not header_seen:
            raise FileFormatNotSupportedError(
                f"{path!r} is empty: a 2DM file starts with a {_HEADER} header"
            )

    def _consume(self, upper: str, fields: list[str], path: str) -> None:
        """Dispatch one record onto the matching collector."""
        if upper in _UNSUPPORTED_ELEMENTS:
            raise FileFormatNotSupportedError(
                f"{path!r} uses the {upper} element, which is not supported; such a "
                f"file is refused rather than guessing a node mapping"
            )
        if upper == _MATERIALS_TAG:
            _require_fields(upper, fields, 1, path)
            declared = _as_int(fields[0], upper, path)
            if not 1 <= declared <= _MAX_MATERIAL_COLUMNS:
                raise FileFormatNotSupportedError(
                    f"{path!r} declares {upper} {declared}, which is not a usable "
                    f"column count: expected 1 to {_MAX_MATERIAL_COLUMNS}"
                )
            self.materials_per_element = declared
        elif upper == _NODE:
            self._add_node(fields, path)
        elif upper in _FACE_NODE_COUNTS:
            self._add_face(upper, fields, path)
        elif upper == _LINE_ELEMENT:
            _require_fields(upper, fields, 3, path)
            self.edge_ids.append(_as_int(fields[0], upper, path))
            self.edge_vertices.append(
                [_as_int(value, upper, path) for value in fields[1:3]]
            )

    def _add_node(self, fields: list[str], path: str) -> None:
        """Collect one ``ND <id> <x> <y> <z>`` record.

        Args:
            fields: The record's fields, without the keyword.
            path: The source path, for the error message.

        Raises:
            FileFormatNotSupportedError: If the record is short of an id and both
                coordinates, or if it disagrees with an earlier ``ND`` record about the
                z column -- in either direction, since whichever kind came first, a
                datum would have to be invented for the records without one. A file
                where *no* record carries z is fine: ``node_z`` is then omitted.
        """
        _require_fields(_NODE, fields, 3, path)
        self.node_ids.append(_as_int(fields[0], _NODE, path))
        self.node_x.append(_as_float(fields[1], _NODE, path))
        self.node_y.append(_as_float(fields[2], _NODE, path))
        if len(fields) > 3:
            self.saw_elevation = True
            self.node_z.append(_as_float(fields[3], _NODE, path))
        else:
            self.missing_elevation = True
            self.node_z.append(0.0)
        if self.saw_elevation and self.missing_elevation:
            # Mixed records would mean inventing a datum for the ones without a z, and
            # a fabricated 0.0 is indistinguishable from a real zero elevation. The
            # all-missing case is fine -- node_z is simply omitted -- but a file that
            # cannot make up its mind is malformed.
            raise FileFormatNotSupportedError(
                f"{path!r} has ND records that disagree about the z column; elevation "
                f"would have to be invented for the ones without it"
            )

    def _add_face(self, upper: str, fields: list[str], path: str) -> None:
        """Collect one face record, keeping its trailing columns unsliced.

        The material and aux columns are stored verbatim because
        ``NUM_MATERIALS_PER_ELEM`` may appear after the elements it applies to; the
        split happens once the whole file has been read. Fields are only ever indexed
        from the start, since the column count varies with the aux columns.

        Args:
            upper: The element keyword, upper-cased.
            fields: The record's fields, without the keyword.
            path: The source path, for the error message.

        Raises:
            FileFormatNotSupportedError: If the record carries fewer vertex ids than
                the element type declares.
        """
        count = _FACE_NODE_COUNTS[upper]
        _require_fields(upper, fields, 1 + count, path)
        self.face_ids.append(_as_int(fields[0], upper, path))
        self.face_vertices.append(
            [_as_int(value, upper, path) for value in fields[1 : 1 + count]]
        )
        # The keyword travels with the trailing columns so a malformed material value
        # can name the record it came from rather than a placeholder.
        self.face_keywords.append(upper)
        self.face_extras.append(fields[1 + count :])

    def build(self, path: str) -> ParsedMesh:
        """Resolve collected ids to indices and assemble the parsed mesh.

        Args:
            path: The source path, for error messages.

        Returns:
            ParsedMesh: The assembled mesh or 1D network.

        Raises:
            FileFormatNotSupportedError: If the file declares no elements at all.
            ValueError: If an element references an undefined node id, or if two nodes,
                two faces or two line elements share an id. Nothing in a 2DM references
                an element by id, so the element maps are built for their duplicate
                check alone.
        """
        if not self.face_vertices and not self.edge_vertices:
            raise FileFormatNotSupportedError(
                f"{path!r} declares no elements: no E3T/E4Q/E6T face and no E2L line"
            )
        node_map = native_id_map(self.node_ids)
        # Built for their duplicate-id check only: nothing in a 2DM references an
        # element by id, but two elements sharing one means a malformed file, and a
        # face and a line element are held to the same standard.
        native_id_map(self.face_ids)
        native_id_map(self.edge_ids)
        faces = self._resolve_faces(node_map, path)
        edges = self._resolve_edges(node_map, path)
        return self._assemble(path, faces, edges)

    def _resolve_faces(self, node_map: dict[int, int], path: str) -> np.ndarray | None:
        """Map face vertex ids to 0-based indices, padding short rows with ``-1``."""
        if not self.face_vertices:
            return None
        max_nodes = max(len(vertices) for vertices in self.face_vertices)
        faces = np.full((len(self.face_vertices), max_nodes), -1, dtype=np.int64)
        for row, vertices in enumerate(self.face_vertices):
            for column, native in enumerate(vertices):
                faces[row, column] = _index_of(native, node_map, path)
        return faces

    def _resolve_edges(self, node_map: dict[int, int], path: str) -> np.ndarray | None:
        """Map E2L vertex ids to 0-based indices."""
        if not self.edge_vertices:
            return None
        edges = np.empty((len(self.edge_vertices), 2), dtype=np.int64)
        for row, vertices in enumerate(self.edge_vertices):
            for column, native in enumerate(vertices):
                edges[row, column] = _index_of(native, node_map, path)
        return edges

    def _assemble(
        self, path: str, faces: np.ndarray | None, edges: np.ndarray | None
    ) -> ParsedMesh:
        """Build the :class:`ParsedMesh` from resolved connectivity."""
        variables: dict[str, ParsedVariable] = {}
        if faces is not None:
            variables.update(self._material_variables(len(faces), path))
        return ParsedMesh(
            node_x=np.array(self.node_x, dtype=np.float64),
            node_y=np.array(self.node_y, dtype=np.float64),
            faces=faces,
            # E2L elements are a separate 1D topology, not this surface's edges, so
            # they never reach `Mesh2d.edge_node_connectivity`.
            network_edges=edges,
            fill_value=-1,
            # Connectivity is already 0-based: it was resolved through the id map, so
            # there is nothing for the validator to subtract. The file's own 1-based
            # numbering is recorded separately so a writer can restore it.
            start_index=0,
            original_start_index=1,
            node_z=(
                np.array(self.node_z, dtype=np.float64) if self.saw_elevation else None
            ),
            crs=None,
            variables=variables,
            global_attributes={"Conventions": "UGRID-1.0", "source": "SMS 2DM"},
            topology_dimension=1 if faces is None else 2,
            mesh_name="mesh2d",
        )

    def _material_variables(self, n_face: int, path: str) -> dict[str, ParsedVariable]:
        """Split the trailing columns into one face variable per material column.

        An element short of the declared material columns contributes ``0.0`` for the
        missing ones rather than failing the read, since a writer omitting a trailing
        column is common and the alternative is refusing an otherwise sound file.

        The columns are float64 even though a material is an integer id: the id is
        exact in float64, writers are not consistent about emitting ``7`` rather than
        ``7.0``, and refusing a file over that spelling would be worse than carrying it.

        Args:
            n_face: The face count, for each variable's shape.
            path: The source path, for error messages.

        Returns:
            dict[str, ParsedVariable]: ``material`` plus ``material_2``.. as declared.
        """
        variables: dict[str, ParsedVariable] = {}
        for index in range(self.materials_per_element):
            column = [
                (
                    _as_float(extras[index], keyword, path)
                    if len(extras) > index
                    else 0.0
                )
                for keyword, extras in zip(self.face_keywords, self.face_extras)
            ]
            name = "material" if index == 0 else f"material_{index + 1}"
            variables[name] = ParsedVariable(
                location="face",
                shape=(n_face,),
                values=np.array(column, dtype=np.float64),
            )
        return variables


def _coordinate_formatter(precision):
    """Return a function formatting a float for a 2DM coordinate column.

    ``"repr"`` (the default) uses Python's shortest round-tripping representation, which
    reproduces a float64 exactly; an integer requests that many significant digits. Full
    precision is the default because a fixed number of decimals loses position for
    projected coordinates in the millions.

    Args:
        precision: ``"repr"`` or an int number of significant digits.

    Returns:
        A callable mapping a float to its formatted string.

    Raises:
        ValueError: If ``precision`` is neither ``"repr"`` nor a positive integer.
    """
    if precision == "repr":
        # repr(float(...)) gives the shortest exact round-trip; repr on a numpy scalar
        # would print "np.float64(...)", so coerce to a Python float first.
        return lambda value: repr(float(value))
    if isinstance(precision, int) and precision > 0:
        return lambda value: f"{value:.{precision}g}"
    raise ValueError(
        f"precision must be 'repr' or a positive integer, not {precision!r}"
    )


def _header_verdict(chunk: bytes) -> bool | None:
    """Judge the first significant line of a chunk of complete lines.

    Args:
        chunk: Bytes holding whole lines only.

    Returns:
        bool | None: Whether the first significant line is the ``MESH2D`` header, or
            ``None`` when the chunk holds no significant line and the answer is still
            undecided. The first token is compared for equality, exactly as the parser
            compares it, so the sniff is neither stricter nor looser than ``read``.
    """
    # `utf-8-sig` drops a byte-order mark, which Windows editors add and which would
    # otherwise make the first token '﻿MESH2D'.
    for line in chunk.decode("utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(_COMMENT):
            return stripped.split()[0].upper() == _HEADER
    return None


def _as_int(value: str, keyword: str, path: str) -> int:
    """Parse one field as an integer, naming the record and file if it is not one.

    Args:
        value: The field text.
        keyword: The record's keyword, for the error message.
        path: The source path, for the error message.

    Returns:
        int: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not an integer.
    """
    try:
        return int(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected an integer, found "
            f"{value!r}"
        ) from error


def _as_float(value: str, keyword: str, path: str) -> float:
    """Parse one field as a float, naming the record and file if it is not one.

    Args:
        value: The field text.
        keyword: The record's keyword, for the error message.
        path: The source path, for the error message.

    Returns:
        float: The parsed value.

    Raises:
        FileFormatNotSupportedError: If the field is not a number.
    """
    try:
        return float(value)
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected a number, found "
            f"{value!r}"
        ) from error


def _require_fields(keyword: str, fields: list[str], needed: int, path: str) -> None:
    """Check a record carries at least the fields its keyword requires.

    Without this a truncated line fails with a bare ``IndexError`` deep in the parser,
    which says nothing about which record of which file is malformed.

    Args:
        keyword: The record's keyword, for the error message.
        fields: The record's fields, without the keyword.
        needed: The minimum number of fields the keyword requires.
        path: The source path, for the error message.

    Raises:
        FileFormatNotSupportedError: If fewer than ``needed`` fields are present.
    """
    if len(fields) < needed:
        raise FileFormatNotSupportedError(
            f"{path!r} has a malformed {keyword} record: expected at least {needed} "
            f"fields after the keyword, found {len(fields)}"
        )


def _index_of(native: int, node_map: dict[int, int], path: str) -> int:
    """Return the 0-based index of a native node id.

    Args:
        native: The node id as the file wrote it.
        node_map: The id-to-index mapping built from the ``ND`` records.
        path: The source path, for the error message.

    Returns:
        int: The node's 0-based index.

    Raises:
        ValueError: If the file never defined that node id.
    """
    try:
        return node_map[native]
    except KeyError:
        raise ValueError(
            f"{path!r} references node id {native}, which no ND record defines"
        ) from None


class _DatGroup:
    """One ``BEGSCL``/``BEGVEC`` group's records, collected in file order."""

    def __init__(self, vector: bool) -> None:
        self.vector = vector
        self.name: str | None = None
        self.time_units: str | None = None
        self.times: list[float] = []
        # One entry per timestep, each ``(n_element, components)``.
        self.values: list[np.ndarray] = []
        # One entry per timestep when the group carries active masks, else empty.
        self.active: list[np.ndarray] = []


class _DatScan:
    """One DAT file's groups, checked against the mesh before any value is read."""

    def __init__(self, path: str, *, n_node: int, n_face: int) -> None:
        self.path = path
        self.n_node = n_node
        self.n_face = n_face
        self.declared_nd: int | None = None
        self.declared_nc: int | None = None
        self.objtype: str | None = None
        self.reference_time: float | None = None
        self.counts_checked = False
        self.groups: list[_DatGroup] = []
        self.current: _DatGroup | None = None

    def run(self) -> None:
        """Stream the file once, reading each group's values as its ``TS`` cards appear.

        Raises:
            FileFormatNotSupportedError: If a group is left unterminated, the file holds
                no group at all, or any card is malformed or misplaced.
        """
        records = iter_records(self.path, comment=_COMMENT)
        for keyword, fields in records:
            self._consume(keyword.upper(), fields, records)
        if self.current is not None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} ends with a group still open: no {_END_DATASET} card"
            )
        if not self.groups:
            raise FileFormatNotSupportedError(
                f"{self.path!r} holds no {_BEGIN_SCALAR}/{_BEGIN_VECTOR} group, so "
                f"there is no data in it to attach"
            )

    def _consume(
        self,
        upper: str,
        fields: list[str],
        records: Iterator[tuple[str, list[str]]],
    ) -> None:
        """Dispatch one card onto the matching collector.

        A card this driver does not model is ignored rather than refused: DAT has
        accumulated optional cards across
        SMS versions, and refusing an unknown one would reject files that read fine.
        """
        if upper == _DATASET:
            # The header card; the dialect was already decided by read_data (a DATASET
            # file is why this new-format scan was chosen), so nothing to do here.
            pass
        elif upper == _OBJTYPE:
            # Recorded for completeness. No SMS writer puts a location in it, so it is
            # deliberately not used to resolve one -- see the class's limitations.
            self.objtype = fields[0] if fields else None
        elif upper in (_BEGIN_SCALAR, _BEGIN_VECTOR):
            self._open_group(upper)
        elif upper == _NODE_COUNT:
            self.declared_nd = self._single_int(upper, fields)
        elif upper == _CELL_COUNT:
            self.declared_nc = self._single_int(upper, fields)
        elif upper == _REFERENCE_TIME:
            _require_fields(upper, fields, 1, self.path)
            self.reference_time = _as_float(fields[0], upper, self.path)
        elif upper in _GROUP_ONLY_CARDS:
            self._consume_group_card(upper, fields, records)
        elif upper == _END_DATASET:
            self._close_group()

    def _consume_group_card(
        self,
        upper: str,
        fields: list[str],
        records: Iterator[tuple[str, list[str]]],
    ) -> None:
        """Handle a card that only means anything inside an open group."""
        group = self._require_group(upper)
        if upper == _NAME:
            _require_fields(upper, fields, 1, self.path)
            # Joined rather than taking one field: a group name may contain spaces, and
            # quotes are stripped because SMS writes them inconsistently.
            group.name = " ".join(fields).strip('"')
        elif upper == _TIME_UNITS:
            _require_fields(upper, fields, 1, self.path)
            group.time_units = fields[0]
        else:
            self._read_timestep(group, fields, records)

    def _single_int(self, upper: str, fields: list[str]) -> int:
        """Parse a one-field integer card such as ``ND`` or ``NC``."""
        _require_fields(upper, fields, 1, self.path)
        return _as_int(fields[0], upper, self.path)

    def _open_group(self, upper: str) -> None:
        """Start a group, refusing one opened while another is still open."""
        if self.current is not None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} opens a {upper} group while another is still open; "
                f"this is refused rather than guessing where the first one ended"
            )
        self.current = _DatGroup(vector=upper == _BEGIN_VECTOR)

    def _close_group(self) -> None:
        """End the open group, refusing an ``ENDDS`` with nothing open."""
        if self.current is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has an {_END_DATASET} card with no open group"
            )
        self.groups.append(self.current)
        self.current = None

    def _require_group(self, upper: str) -> _DatGroup:
        """Return the open group, refusing a group-only card found outside one."""
        if self.current is None:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a {upper} card outside a "
                f"{_BEGIN_SCALAR}/{_BEGIN_VECTOR} group, so there is nothing for it to "
                f"describe"
            )
        return self.current

    def _read_timestep(
        self,
        group: _DatGroup,
        fields: list[str],
        records: Iterator[tuple[str, list[str]]],
    ) -> None:
        """Read one ``TS`` card and the value block that belongs to it.

        The element counts are checked here, before the first value is consumed, which
        is the point of checking them at all.

        A new-format ``TS`` is ``<istat> <time>``; the old dialect's ``TS <time>`` is
        read by :class:`_DatOldScan`, which ``read_data`` routes a no-``DATASET`` file to,
        so this path is only ever the new arity.
        """
        self._check_counts()
        _require_fields(_TIMESTEP, fields, _TS_FIELDS_NEW, self.path)
        istat = _as_int(fields[0], _TIMESTEP, self.path)
        time = _as_float(fields[1], _TIMESTEP, self.path)
        group.times.append(time)
        if istat:
            group.active.append(self._pull_active(records))
        _location, count = self._element_count()
        columns = 2 if group.vector else 1
        group.values.append(self._pull_block(records, count, columns, _TIMESTEP))

    def _pull_active(self, records: Iterator[tuple[str, list[str]]]) -> np.ndarray:
        """Read the per-cell active block that follows a ``TS`` card with ``istat`` set.

        Raises:
            FileFormatNotSupportedError: If any flag is not ``0`` or ``1``. That is the
                only check available on the block's position: this driver reads it
                before the values, and a block of real values landing
                here would otherwise be taken as flags and the values shifted by a whole
                block. The check cannot fire when the node and cell counts are equal.
        """
        block = self._pull_block(records, self.n_face, 1, "active")
        flags = block[:, 0]
        if not bool(np.isin(flags, (0.0, 1.0)).all()):
            offenders = np.unique(flags[~np.isin(flags, (0.0, 1.0))])[:3].tolist()
            raise FileFormatNotSupportedError(
                f"{self.path!r} has an active block holding values other than 0 and 1 "
                f"({offenders}); it is read immediately after the TS card, before the "
                f"values, and this file does not match that layout"
            )
        return flags.astype(np.int8)

    def _pull_block(
        self,
        records: Iterator[tuple[str, list[str]]],
        rows: int,
        columns: int,
        what: str,
    ) -> np.ndarray:
        """Read ``rows`` lines of ``columns`` numbers each from the record stream.

        Args:
            records: The file's remaining records.
            rows: How many lines the block holds.
            columns: How many numbers each line holds.
            what: What the block belongs to, for the error message.

        Returns:
            np.ndarray: The block, shape ``(rows, columns)``, float64.

        Raises:
            FileFormatNotSupportedError: If the file ends mid-block, a line holds fewer
                numbers than the block needs, or a field is not a number.
        """
        block = np.empty((rows, columns), dtype=np.float64)
        for row in range(rows):
            try:
                keyword, fields = next(records)
            except StopIteration:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} ends inside a {what} block: expected {rows} rows, "
                    f"found {row}"
                ) from None
            tokens = [keyword, *fields]
            if len(tokens) < columns:
                raise FileFormatNotSupportedError(
                    f"{self.path!r} has a short {what} row: expected {columns} "
                    f"value(s), found {len(tokens)}"
                )
            for column in range(columns):
                block[row, column] = _as_float(tokens[column], what, self.path)
        return block

    def _check_counts(self) -> None:
        """Check the declared element counts against the mesh, once.

        Raises:
            ValueError: If a declared count disagrees with the mesh, or the file
                declares neither. Nothing else in a DAT ties it to a topology, so an
                unchecked attach produces a plausible, wrong result.
        """
        if self.counts_checked:
            return
        if self.declared_nd is None and self.declared_nc is None:
            raise ValueError(
                f"{self.path!r} declares neither {_NODE_COUNT} nor {_CELL_COUNT}, so "
                f"nothing can be checked against the mesh; a DAT carries no other "
                f"identifier tying it to a topology, and attaching it unchecked would "
                f"be a guess"
            )
        if self.declared_nd is not None and self.declared_nd != self.n_node:
            raise ValueError(
                f"{self.path!r} does not match this mesh: declares {_NODE_COUNT} "
                f"{self.declared_nd} but the mesh has {self.n_node} nodes"
            )
        if self.declared_nc is not None and self.declared_nc != self.n_face:
            raise ValueError(
                f"{self.path!r} does not match this mesh: declares {_CELL_COUNT} "
                f"{self.declared_nc} but the mesh has {self.n_face} faces"
            )
        self.counts_checked = True

    def _element_count(self) -> tuple[str, int]:
        """Return the location and element count a group's values are indexed by.

        ``ND`` wins when both are declared: ``BEGSCL``/``BEGVEC`` groups are
        node-located in SMS, and both counts have already been checked against the mesh
        so neither can be wrong -- only ambiguous.
        """
        if self.declared_nd is not None:
            return "node", self.declared_nd
        # `_check_counts` has refused the case where neither is declared.
        return "face", int(self.declared_nc or 0)

    def build(self) -> dict[str, ParsedVariable]:
        """Turn the collected groups into parsed variables.

        Returns:
            dict[str, ParsedVariable]: One variable per scalar group, two per vector
                group, plus a face-located ``<name>_active`` for each group carrying
                active masks.

        Raises:
            ValueError: If a declared count disagrees with the mesh, or the file
                declares neither count.
            FileFormatNotSupportedError: If a group carries no ``TS`` card at all, so
                there are no values in it to attach, or two groups would produce the
                same variable name.
        """
        self._check_counts()
        location, _count = self._element_count()
        variables: dict[str, ParsedVariable] = {}
        for index, group in enumerate(self.groups):
            for name, variable in self._group_variables(group, index, location):
                if name in variables:
                    # A malformed file, not a mesh mismatch, so it is the format error:
                    # `attach_data` passes a ValueError through only when the message
                    # names the mesh, and wraps anything else as a driver failure.
                    raise FileFormatNotSupportedError(
                        f"{self.path!r} produces the variable name {name!r} twice; "
                        f"name the groups distinctly, since one would otherwise "
                        f"silently replace the other"
                    )
                variables[name] = variable
        return variables

    def _group_variables(
        self, group: _DatGroup, index: int, location: str
    ) -> list[tuple[str, ParsedVariable]]:
        """Build one group's variables, splitting a vector group into components."""
        if not group.times:
            raise FileFormatNotSupportedError(
                f"{self.path!r} has a group with no {_TIMESTEP} card, so it carries no "
                f"values to attach"
            )
        # Fall back to a positional name rather than refusing: a nameless group is
        # readable, and the position is the only other thing identifying it.
        base = group.name or f"dataset_{index + 1}"
        times = np.array(group.times, dtype=np.float64)
        stacked = np.stack(group.values)
        attributes = self._attributes(group)
        built: list[tuple[str, ParsedVariable]] = []
        if group.vector:
            # Two scalars rather than one two-component variable: MeshVariable has no
            # component axis. See the class's documented limitations.
            for offset, suffix in ((0, "x"), (1, "y")):
                built.append(
                    (
                        f"{base}_{suffix}",
                        ParsedVariable(
                            location=location,
                            shape=stacked[:, :, offset].shape,
                            values=stacked[:, :, offset],
                            time_values=times,
                            attributes=dict(attributes),
                        ),
                    )
                )
        else:
            built.append(
                (
                    base,
                    ParsedVariable(
                        location=location,
                        shape=stacked[:, :, 0].shape,
                        values=stacked[:, :, 0],
                        time_values=times,
                        attributes=dict(attributes),
                    ),
                )
            )
        if group.active:
            masks = np.stack(group.active)
            built.append(
                (
                    f"{base}_active",
                    ParsedVariable(
                        # The mask counts cells while a node-located group counts nodes,
                        # so it cannot ride along as ParsedVariable.active -- that would
                        # give it the parent's location and shape.
                        location="face",
                        shape=masks.shape,
                        values=masks,
                        # The time axis is kept (the mask varies per step), but the
                        # group's reference_time/time_units are not: a wet/dry flag is
                        # not a timed physical quantity, so carrying a Julian reference
                        # time on it is misleading metadata.
                        time_values=times,
                    ),
                )
            )
        return built

    def _attributes(self, group: _DatGroup) -> dict[str, object]:
        """Collect the metadata a DAT declares that has nowhere else to go.

        ``attach_data`` keeps the dataset's own ``global_attributes`` and merges only
        the variables a driver returns, so a file-level value such as ``RT_JULIAN`` can
        only travel on the variables themselves.
        """
        attributes: dict[str, object] = {}
        if self.reference_time is not None:
            attributes["reference_time"] = self.reference_time
        if group.time_units is not None:
            attributes["time_units"] = group.time_units
        return attributes
