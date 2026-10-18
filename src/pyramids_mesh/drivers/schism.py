"""SCHISM ``hgrid.gr3`` reader (task C2).

SCHISM is an unstructured-grid hydrodynamic model. Its horizontal grid is the ASCII
``hgrid.gr3`` -- a node list ``(id, x, y, depth)``, a mixed triangle/quad element table
``(id, i34, n1, n2, n3[, n4])`` and open/land boundary sections. The layout is the one
ADCIRC's ``fort.14`` uses, so the node- and element-block parser is shared through
:mod:`pyramids_mesh.drivers._gr3`; the two differences this driver adds are quad elements
(``i34 == 4``) and the boundary node-id lists, which are carried as metadata.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

from typing import TextIO

import numpy as np

from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh
from pyramids_mesh.drivers._gr3 import read_gr3_grid, sniff_gr3


class SchismDriver(BaseDriver):
    """Reader for the SCHISM ``hgrid.gr3`` horizontal grid.

    The mesh comes from ``hgrid.gr3``: node coordinates, the node ``depth`` (becomes a
    ``node_z`` variable) and the 1-based element table, which may hold both triangles
    (``i34 == 3``) and quads (``i34 == 4``). A mixed grid is returned as a
    quad-width table whose triangle rows are fill-padded, the form pyramids expects. The
    open- and land-boundary node-id lists are parsed and carried as dataset metadata.

    Documented limitations:
        - **Horizontal grid only.** ``hgrid.gr3`` carries the 2-D mesh and its node
          depth; SCHISM's vertical grid (``vgrid.in``) and results (``schout_*.nc``) are
          separate files and are not read here.
        - **Node and element ids are 1-based** -- resolved to 0-based, the base recorded
          for a writer. Boundary node-id lists are likewise converted to 0-based so they
          index straight into the mesh.
        - **No CRS.** ``.gr3`` declares no authority (the coordinates may be lon/lat or a
          projected system), so ``crs`` is ``None`` rather than guessed. ``depth`` is
          carried verbatim as ``node_z``.
        - **Boundaries become metadata, not topology.** The open/land node lists are
          stored on ``global_attributes`` (``open_boundary_nodes`` /
          ``land_boundary_nodes``); they are not built into edge connectivity.

    Example:
        Read a SCHISM grid and reach its depth and boundary lists (shown, not run as a
        doctest -- it needs a real ``hgrid.gr3``)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("run/hgrid.gr3")
            bed = mesh.get("node_z")                       # SCHISM node depth
            ocean = mesh.global_attributes["open_boundary_nodes"]  # 0-based node ids
    """

    name = "schism"
    long_name = "SCHISM"
    suffixes = (".gr3",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def _sniff(self, path: str) -> bool:
        """Report whether the file opens as a ``.gr3`` grid.

        Shares :func:`pyramids_mesh.drivers._gr3.sniff_gr3` with the ADCIRC reader, since
        the header and first node line are identical between the two formats.

        Args:
            path: The file to inspect.

        Returns:
            bool: ``True`` when the header and first node line are grid-shaped.
        """
        return sniff_gr3(path)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read an ``hgrid.gr3`` into a :class:`ParsedMesh`.

        Args:
            path: The ``.gr3`` file.
            lazy: Unused -- the whole grid is read eagerly, as a ``.gr3`` is small and
                carries no deferred result series.

        Returns:
            ParsedMesh: The surface mesh, ``node_z`` and the boundary node lists as
                metadata.

        Raises:
            FileFormatNotSupportedError: If an element is neither a triangle nor a quad.
        """
        with open(path, "r", encoding="latin-1") as handle:
            node_x, node_y, node_z, faces = read_gr3_grid(
                handle, allow_quads=True, source=path
            )
            open_boundaries, land_boundaries = _read_boundaries(handle)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=faces,
            fill_value=-1,
            # .gr3 node/element ids are 1-based; validate re-bases and records the base.
            start_index=1,
            node_z=node_z,
            crs=None,
            global_attributes={
                "source": "SCHISM hgrid.gr3",
                "n_open_boundaries": len(open_boundaries),
                "n_land_boundaries": len(land_boundaries),
                "open_boundary_nodes": open_boundaries,
                "land_boundary_nodes": land_boundaries,
            },
            topology_dimension=2,
            mesh_name="mesh2d",
        )


def _read_boundaries(handle: TextIO) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read the open- then land-boundary node lists from an open handle.

    The handle is positioned just after the element block. Each section opens with an
    ``<n> = Number of ... boundaries`` line and a redundant total-node-count line, then
    one ``<count> ...`` header plus ``count`` node-id lines per boundary. A ``.gr3`` with
    no boundary section (a bathymetry/nodal-value grid) simply reaches end-of-file, and
    both groups come back empty.

    Args:
        handle: The open text handle, positioned after the element block.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: The open- and land-boundary node-id
            arrays, one int64 array of 0-based node ids per boundary.
    """
    open_boundaries = _read_boundary_group(handle)
    land_boundaries = _read_boundary_group(handle)
    return open_boundaries, land_boundaries


def _read_boundary_group(handle: TextIO) -> list[np.ndarray]:
    """Read one boundary section (open or land) into a list of node-id arrays.

    Args:
        handle: The open handle, positioned at the section's count line.

    Returns:
        list[np.ndarray]: One int64 array of 0-based node ids per boundary; empty when
            the section is absent.
    """
    header = handle.readline()
    if not header.strip():
        return []
    try:
        count = int(header.split()[0])
    except (IndexError, ValueError):
        return []
    handle.readline()  # total-node-count line, recomputed from the lists below
    groups: list[np.ndarray] = []
    for _ in range(count):
        size = int(handle.readline().split()[0])
        ids = [int(handle.readline().split()[0]) - 1 for _ in range(size)]
        groups.append(np.asarray(ids, dtype=np.int64))
    return groups
