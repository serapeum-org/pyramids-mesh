# SCHISM — `hgrid.gr3` horizontal grid

The SCHISM reader. SCHISM is an unstructured-grid hydrodynamic model whose horizontal
grid is the ASCII `hgrid.gr3`: a node list `(id, x, y, depth)`, a mixed triangle/quad
element table `(id, i34, n1, n2, n3[, n4])` and open/land boundary sections. The layout
is the one ADCIRC's `fort.14` uses, so the two readers share the node- and element-block
parser.

| | |
|---|---|
| Driver name | `schism` |
| Suffixes | `.gr3` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("run/hgrid.gr3")

bed = mesh.get("node_z")                                   # SCHISM node depth
ocean = mesh.global_attributes["open_boundary_nodes"]      # 0-based node-id arrays
land = mesh.global_attributes["land_boundary_nodes"]
```

## Documented limitations

- **Horizontal grid only.** `hgrid.gr3` carries the 2-D mesh and its node depth; SCHISM's
  vertical grid (`vgrid.in`) and results (`schout_*.nc`) are separate files and are not
  read here.
- **Triangles and quads.** `i34` is `3` (triangle) or `4` (quad); a mixed grid is returned
  as a quad-width face table whose triangle rows are fill-padded, the form pyramids
  expects. Any other element type is refused rather than guessed.
- **Node and element ids are 1-based** — resolved to 0-based, the base recorded for a
  writer. Boundary node-id lists are likewise converted to 0-based so they index straight
  into the mesh.
- **No CRS.** `.gr3` declares no authority (the coordinates may be lon/lat or a projected
  system), so `crs` is `None` rather than guessed. `depth` is carried verbatim as
  `node_z`.
- **Boundaries become metadata, not topology.** The open/land node lists are stored on
  `global_attributes` (`open_boundary_nodes` / `land_boundary_nodes`, with the counts
  `n_open_boundaries` / `n_land_boundaries`); they are not built into edge connectivity.

## API

::: pyramids_mesh.drivers.schism.SchismDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
