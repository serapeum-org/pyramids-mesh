# MODFLOW 6 DISU

Reader for the **MODFLOW 6 DISU** (discretization by unstructured) grid.

MODFLOW 6's DISU package describes a fully-unstructured groundwater grid by its connection
graph. When the grid also declares vertices, its horizontal footprint is an unstructured
set of polygonal cells — identical in shape to a DISV grid. This driver reads the ASCII
`.disu` input file and builds that footprint as a surface mesh; it reuses the DISV
block-parsing helpers and refuses a connectivity-only DISU, which carries no geometry.

| | |
|---|---|
| **Driver name** | `modflow6_disu` |
| **Suffixes** | `.disu` (the ASCII grid input of a MODFLOW 6 model) |
| **Capabilities** | `read_mesh` |
| **Returns** | `UgridDataset` |

## The ASCII grammar

DISU shares DISV's `BEGIN`/`END` block grammar. The geometry lives in three blocks; the
`GRIDDATA` and `CONNECTIONDATA` blocks are skipped:

```
BEGIN DIMENSIONS
  NODES <nodes>      number of cells -- the face count
  NJA   <nja>        length of the connection vector (the DISU signature)
  NVERT <nvert>      number of vertices -- the node count (0 when geometry is absent)
END DIMENSIONS

BEGIN VERTICES
  <iv> <x> <y>       one 1-based vertex per line
END VERTICES

BEGIN CELL2D
  <icell2d> <xc> <yc> <ncvert> <icvert...>   1-based vertex ids, clockwise
END CELL2D
```

`VERTICES` becomes `node_x` / `node_y`; each `CELL2D` row's `icvert` list becomes one
fill-padded, 1-based face row; `xc` / `yc` become the face centres. A `VERTICES` or
`CELL2D` block may defer its rows to an external file with an `OPEN/CLOSE <fname>`
directive, which is read relative to the `.disu` file.

## Dispatch

A `.disu` file dispatches by its suffix; `can_open` confirms the DISU `DIMENSIONS` block
(`NODES` and `NJA`) is present rather than only matching the name:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model/keating.disu", driver="modflow6_disu")
footprint = mesh.mesh      # NODES polygons, one horizontal footprint
```

## Documented limitations

- **Vertices required.** A connectivity-only DISU (`NVERT` absent or zero, no `VERTICES` /
  `CELL2D`) has no geometry to place; it is refused with a clear error rather than
  approximated from the connection graph.
- **One horizontal footprint.** The mesh is the `NODES` cells of `CELL2D`. The `GRIDDATA`
  arrays (`TOP` / `BOT` / `AREA` / `IDOMAIN`) and the `CONNECTIONDATA` connection graph are
  not modelled — this package reads the mesh file format, not the full groundwater model.
- **ASCII only.** The binary `.disu.grb` grid is not read; point the reader at the ASCII
  `.disu` input.
- **Vertex ids are 1-based.** They are resolved to 0-based and the base recorded for a
  writer. Short (polygonal) face rows are padded; the quad test fixture pads nothing.
- **Cells are clockwise.** DISU lists `icvert` clockwise; pyramids reverses each face to
  counter-clockwise on read.
- There is no CRS or node elevation in a `.disu` file, so `crs` and `node_z` are `None`;
  the cell centres are carried as the face centres.

## API

::: pyramids_mesh.drivers.modflow.Modflow6DisuDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
