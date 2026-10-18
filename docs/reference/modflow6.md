# MODFLOW 6 DISV

Reader for the **MODFLOW 6 DISV** (discretization by vertices) grid (task C5).

MODFLOW 6's DISV package describes a layered groundwater grid whose horizontal footprint is
an unstructured set of polygonal cells. This driver reads the recommended ASCII `.disv`
input file — one file of a MODFLOW 6 model directory — and builds that single horizontal
cell layer as a surface mesh.

| | |
|---|---|
| **Driver name** | `modflow6_disv` |
| **Suffixes** | `.disv` (the ASCII grid input of a MODFLOW 6 model) |
| **Capabilities** | `read_mesh` |

## The ASCII grammar

Three keyword blocks matter; the intervening `GRIDDATA` block (`TOP`/`BOTM`/`IDOMAIN`) is
skipped:

```
BEGIN DIMENSIONS
  NLAY <nlay>        number of layers
  NCPL <ncpl>        number of cells per layer -- the face count
  NVERT <nvert>      number of vertices -- the node count
END DIMENSIONS

BEGIN VERTICES
  <iv> <x> <y>       one 1-based vertex per line
END VERTICES

BEGIN CELL2D
  <icell2d> <xc> <yc> <ncvert> <icvert...>   1-based vertex ids, clockwise
END CELL2D
```

`VERTICES` becomes `node_x` / `node_y`; each `CELL2D` row's `icvert` list becomes one
fill-padded, 1-based face row; `xc` / `yc` become the face centres.

## Dispatch

A `.disv` file dispatches by its suffix; `can_open` confirms the DISV blocks are present
rather than only matching the name:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model/tri_model.disv", driver="modflow6_disv")
footprint = mesh.mesh      # NCPL polygons, one horizontal layer
```

## Documented limitations

- **One horizontal layer.** A DISV grid is `NLAY` layers sharing a single horizontal
  footprint of `NCPL` cells; the mesh is that footprint. The vertical layering and the
  `TOP` / `BOTM` / `IDOMAIN` arrays in `GRIDDATA` are not modelled — this package reads the
  mesh file format, not the full groundwater model.
- **ASCII only (for now).** The binary companion grid `.disv.grb` (ASCII header `GRID
  DISV`, a text metadata block naming the arrays, then binary `VERTICES` / `IAVERT` /
  `JAVERT`) is **not read in v1**. The driver claims only the `.disv` suffix, so a lone
  `.grb` is not dispatched to it. Reading the binary grid as a fallback when only it is
  present is a possible future addition.
- **Vertex ids are 1-based.** They are resolved to 0-based and the base recorded for a
  writer. A short (polygonal) face row is padded; a triangular grid has no padding.
- **Cells are clockwise.** DISV lists `icvert` clockwise; pyramids reverses each face to
  counter-clockwise on read.
- There is no CRS or node elevation in a `.disv` file, so `crs` and `node_z` are `None`;
  the cell centres are carried as the face centres.
