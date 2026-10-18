"""FVCOM native grid reader (task C8).

An FVCOM native grid is not one file but a **trio of ASCII files** sharing a basename:

- ``<base>_grd.dat`` -- the topology. A ``Node Number = N`` line, a ``Cell Number = M``
  line, then ``M`` cell lines ``<cellid> <n1> <n2> <n3> <flag>`` (every element is a
  triangle), then ``N`` node lines ``<nodeid> <x> <y> <z>``.
- ``<base>_dep.dat`` -- the bathymetry. A ``Node Number = N`` line, then ``N`` lines
  ``<x> <y> <depth>`` in node order; the depth column becomes ``node_z``.
- ``<base>_obc.dat`` -- the open boundary. An ``OBC Node Number = K`` line, then ``K``
  lines ``<seq> <nodeid> <type>``; the node ids ride along as mesh metadata.

Node and cell ids are 1-based and resolved to 0-based, the base recorded for a writer.
Everything is plain text, so there is no declared dependency beyond numpy. The driver
owns only the file parsing; the mesh object it builds is ``pyramids.netcdf.ugrid``'s.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import warnings

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The component that defines the grid; the suffix that marks it within the trio.
_GRD_SUFFIX = "_grd.dat"
_DEP_SUFFIX = "_dep.dat"
_OBC_SUFFIX = "_obc.dat"

# The header cards the first two significant lines of a grd file must carry, lowered.
_NODE_CARD = "node number"
_CELL_CARD = "cell number"


def _significant_lines(path: str) -> list[str]:
    """Return the file's non-blank lines, stripped and in order.

    Args:
        path: The text file to read.

    Returns:
        list[str]: Every line that is not blank, whitespace trimmed.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        return [stripped for line in handle if (stripped := line.strip())]


def _header_count(line: str, card: str) -> int:
    """Return the integer a ``<card> = <n>`` header line declares.

    The spacing around ``=`` varies between FVCOM writers (``Node Number = 871`` and
    ``OBC Node Number =21`` both occur), so the value is taken as whatever follows the
    first ``=``.

    Args:
        line: The header line to read.
        card: The card name expected before the ``=``, for the error message only.

    Returns:
        int: The declared count.

    Raises:
        FileFormatNotSupportedError: If the line is not a ``<card> = <int>`` header.
    """
    _, separator, value = line.partition("=")
    try:
        if not separator:
            raise ValueError
        return int(value.strip())
    except ValueError as error:
        raise FileFormatNotSupportedError(
            f"expected an FVCOM '{card} = <int>' header, found {line!r}"
        ) from error


class FvcomDriver(BaseDriver):
    """Reader for the FVCOM native grid trio.

    An FVCOM native grid is the ``<base>_grd.dat`` / ``<base>_dep.dat`` /
    ``<base>_obc.dat`` trio. ``read_file`` reaches the driver either by pointing it at
    the ``<base>_grd.dat`` file, or by passing a directory holding one with
    ``driver="fvcom"``. The matching ``_dep.dat`` and ``_obc.dat`` are found beside the
    grid file; both are optional, and a grid read on its own still yields the topology.

    Because ``.dat`` is a generic suffix other formats also use, ``can_open`` sniffs the
    ``Node Number`` / ``Cell Number`` header so it never steals a foreign ``.dat``.

    Documented limitations:
        - **The input is the ``_grd.dat`` file (or its directory).** The matching
          ``_dep.dat`` / ``_obc.dat`` are resolved by basename beside it; either may be
          absent.
        - **Every element is a triangle.** The grid stores three node ids per cell; a
          fill column is never needed.
        - **Node and cell ids are 1-based** -- resolved to 0-based, the base recorded for
          a writer.
        - **Depth comes from ``_dep.dat``.** Its ``depth`` column becomes a ``node_z``
          node variable; the ``z`` column of ``_grd.dat`` is ignored. Without a
          ``_dep.dat`` there is no ``node_z``.
        - **Open-boundary nodes become metadata.** The ``_obc.dat`` node ids are carried
          (0-based) on the dataset's global attributes, not as a topology of their own.
        - There is no CRS in the files read here, so ``crs`` is ``None``.

    Example:
        Read an FVCOM grid by its ``_grd.dat`` file (shown, not run as a doctest -- it
        needs the real trio on disk)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("path/to/chn_grd.dat")
            depth = mesh.get("node_z")
    """

    name = "fvcom"
    long_name = "FVCOM native grid"
    # FVCOM is a trio; read_file reaches it through the _grd.dat component.
    suffixes = (".dat",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is an FVCOM grid file or a directory holding one.

        Overrides the base check, which only ever sniffs a file: an FVCOM grid is
        reached through its ``_grd.dat`` component or through the directory that holds
        it. ``.dat`` is a generic suffix, so the grid header is sniffed to decline a
        foreign ``.dat``. Never raises -- a sniff error is warned and declined.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a ``_grd.dat`` file, or a directory containing one, whose
                header is an FVCOM grid; ``False`` otherwise.
        """
        grd = self._grd_path(path)
        if grd is None:
            return False
        try:
            return self._sniff(grd)
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            warnings.warn(
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False

    @staticmethod
    def _grd_path(path: str) -> str | None:
        """Resolve the ``_grd.dat`` file from a grid file or a directory holding one.

        Args:
            path: A ``_grd.dat`` file, or a directory containing one.

        Returns:
            str | None: The grid-file path, or ``None`` when ``path`` is neither.
        """
        if os.path.isfile(path) and path.lower().endswith(_GRD_SUFFIX):
            return path
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                if entry.lower().endswith(_GRD_SUFFIX):
                    return os.path.join(path, entry)
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether the first two significant lines are the FVCOM grid header.

        Args:
            path: The ``_grd.dat`` file to inspect.

        Returns:
            bool: ``True`` when a ``Node Number`` line is followed by a ``Cell Number``
                line.
        """
        lines = _significant_lines(path)
        if len(lines) < 2:
            return False
        return lines[0].lower().startswith(_NODE_CARD) and lines[1].lower().startswith(
            _CELL_CARD
        )

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an FVCOM grid trio into a :class:`ParsedMesh`.

        Args:
            path: The ``_grd.dat`` file, or a directory holding one.
            lazy: Unused -- an FVCOM grid is small ASCII and read whole.

        Returns:
            ParsedMesh: Nodes, the 1-based triangle connectivity, node depth from
                ``_dep.dat`` and the open-boundary node ids from ``_obc.dat``.

        Raises:
            FileFormatNotSupportedError: If ``path`` resolves to no ``_grd.dat`` file, or
                a file is not the FVCOM table its header promises.
        """
        grd = self._grd_path(path)
        if grd is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an FVCOM '_grd.dat' file or a directory holding one"
            )
        return _FvcomReader(grd).build()


class _FvcomReader:
    """One FVCOM grid trio: read the grid, its depth and its open boundary."""

    def __init__(self, grd_path: str) -> None:
        self.grd_path = grd_path
        base = grd_path[: -len(_GRD_SUFFIX)]
        self.dep_path = base + _DEP_SUFFIX
        self.obc_path = base + _OBC_SUFFIX

    def build(self) -> ParsedMesh:
        """Read the trio and assemble the triangular mesh.

        Returns:
            ParsedMesh: The surface mesh, with node depth and open-boundary metadata.
        """
        node_x, node_y, faces = self._read_grid()
        node_z = self._read_depth(len(node_x))
        attributes = self._read_open_boundary()
        attributes["source"] = "FVCOM native grid"
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            start_index=1,  # 1-based ids; validation re-bases and records the base
            node_z=node_z,
            crs=None,
            global_attributes=attributes,
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _read_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read ``_grd.dat`` into node coordinates and the 1-based triangle table.

        Returns:
            tuple: ``(node_x, node_y, faces)`` -- float64 coordinates and the
                ``(n_cell, 3)`` int64 connectivity as written (1-based).

        Raises:
            FileFormatNotSupportedError: If the file is shorter than its header promises.
        """
        lines = _significant_lines(self.grd_path)
        if len(lines) < 2:
            raise FileFormatNotSupportedError(
                f"{self.grd_path!r} is not an FVCOM grid: no 'Node/Cell Number' header"
            )
        n_node = _header_count(lines[0], "Node Number")
        n_cell = _header_count(lines[1], "Cell Number")
        cell_lines = lines[2 : 2 + n_cell]
        node_lines = lines[2 + n_cell : 2 + n_cell + n_node]
        if len(cell_lines) != n_cell or len(node_lines) != n_node:
            raise FileFormatNotSupportedError(
                f"{self.grd_path!r} is truncated: header promises {n_cell} cells and "
                f"{n_node} nodes, found {len(cell_lines)} and {len(node_lines)}"
            )
        faces = np.array([line.split()[1:4] for line in cell_lines], dtype=np.int64)
        coords = np.array([line.split()[1:4] for line in node_lines], dtype=np.float64)
        return coords[:, 0].copy(), coords[:, 1].copy(), faces

    def _read_depth(self, n_node: int) -> np.ndarray | None:
        """Read the ``depth`` column of ``_dep.dat`` in node order, or ``None``.

        Args:
            n_node: The node count the grid declared, for a length check.

        Returns:
            np.ndarray | None: The float64 depth per node, or ``None`` when no
                ``_dep.dat`` sits beside the grid.

        Raises:
            FileFormatNotSupportedError: If the file holds fewer depth rows than nodes.
        """
        if not os.path.isfile(self.dep_path):
            return None
        lines = _significant_lines(self.dep_path)
        rows = lines[1:]  # the first significant line is the 'Node Number = N' header
        if len(rows) < n_node:
            raise FileFormatNotSupportedError(
                f"{self.dep_path!r} holds {len(rows)} depth rows, fewer than the "
                f"{n_node} nodes the grid declares"
            )
        depth = np.array([row.split()[2] for row in rows[:n_node]], dtype=np.float64)
        return depth

    def _read_open_boundary(self) -> dict[str, object]:
        """Read the open-boundary node ids of ``_obc.dat`` as 0-based metadata.

        Returns:
            dict[str, object]: ``open_boundary_nodes`` (0-based int64 ids) and
                ``open_boundary_node_count``, or an empty dict when no ``_obc.dat`` sits
                beside the grid.
        """
        if not os.path.isfile(self.obc_path):
            return {}
        lines = _significant_lines(self.obc_path)
        count = _header_count(lines[0], "OBC Node Number")
        rows = lines[1 : 1 + count]
        nodes = np.array([row.split()[1] for row in rows], dtype=np.int64) - 1
        return {
            "open_boundary_nodes": nodes,
            "open_boundary_node_count": int(nodes.size),
        }
