# DHI FEFLOW — classic ASCII `.fem` mesh

Reader for the **DHI FEFLOW** classic ASCII `.fem` finite-element mesh.

FEFLOW is DHI's finite-element groundwater model. Its classic ASCII `.fem` file is a
keyword deck: a sequence of all-caps section headers each followed by the section's numeric
rows. This driver reads the mesh those sections describe — the node coordinates and the
element-to-node connectivity — and builds a surface mesh. The physics, boundary-condition,
material and time-series sections are not interpreted.

| | |
|---|---|
| Driver name | `feflow` |
| Suffixes | `.fem` (shared with Nastran — see below) |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

## The geometry sections

```
PROBLEM:                            the two statements can_open keys on
CLASS (v.7.501...)
...
DIMENS
  <n_nodes> <n_elements> <nodes_per_element> ...
SCALE
  <sx>, <ox>, <sy>, <oy>, <sz>, <oz>     per-axis (scale, offset) pairs
NODE
  <n1> <n2> <n3[ n4 ...]>            n_elements rows of 1-based node ids
COOR
  <x1>, <x2>, ... <y1>, <y2>, ...   one flat stream: all x, then all y[, then z]
```

- **`DIMENS`** gives the counts and the element valence (`3` triangle, `4` quad or
  tetrahedron, `6` prism, `8` hexahedron).
- **`SCALE`** maps the stored coordinates to real ones: `real = stored * scale + offset`,
  per axis. A 2D mesh leaves the z pair at zero.
- **`NODE`** is the element-to-node connectivity, 1-based, `nodes_per_element` ids per row.
- **`COOR`** is one flat, comma-separated stream (wrapped ~12 values per line) of all node
  x, then all y, then — for a 3D mesh — all z. Whether z is present is decided by the
  stream length: `len(COOR) / n_nodes` is `2` or `3`.

## Dispatch and the shared `.fem` suffix

The `.fem` suffix is **shared** with the Nastran bulk-data format. `can_open` therefore
requires the FEFLOW signature — the first two non-blank statements being `PROBLEM:` then
`CLASS` — and declines a Nastran deck (which opens on `$` comment lines and `BEGIN BULK`):

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model/disv_nodes.fem", driver="feflow")
surface = mesh.mesh      # the finite-element mesh
```

## Documented limitations

- **Geometry only.** Only `DIMENS` / `SCALE` / `NODE` / `COOR` are read; the physics,
  boundary-condition, material and time-series sections are not interpreted — this package
  reads the mesh file format, not the full model.
- **Node ids are 1-based** — resolved to 0-based, the base recorded for a writer. A
  mixed-valence deck pads short rows; a single-valence deck (the triangular test fixture)
  pads nothing.
- **No CRS.** A `.fem` file carries no spatial-reference authority, so `crs` is `None`. The
  `SCALE` transform is applied to the coordinates as the file specifies; a 3D deck carries
  the elevation as `node_z`.
- **`.fem` is shared with Nastran.** `can_open` keys on the `PROBLEM:` / `CLASS` signature
  so a Nastran deck sharing the suffix is declined rather than mis-parsed.

## API

::: pyramids_mesh.drivers.feflow.FeflowDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
