# RiverFlow2D — `.FED` mesh

Reader for the **RiverFlow2D** (Hydronia) `.FED` triangular mesh (task C29,
issue #67). A `.FED` is a single ASCII file: a count header, a node block, then a
triangular element block. The layout below was confirmed against the 16 meshes shipped in
Hydronia's *ExampleProjects* package (Hoh River, Brumadinho tailings, DamBreach, …):

```text
NELEM NNODES 5 5                                                   # 4 integers
IN  X  Y  ZB  INITWSE  MINERODELEV  BCTYPE  BCFILENAME             # NNODES node lines
IE  N1 N2 N3  MANNINGN  ELZB  ELINITWSE  ELMINERODELEV  <reserved> # NELEM element lines
```

Node and element ids are 1-based and consecutive; every element is a counter-clockwise
triangle. The header's trailing two fields are unused dummies, and a trailing element
column is reserved (`0.0` in every shipped mesh).

| | |
|---|---|
| **Driver name** | `riverflow2d` |
| **Suffixes** | `.fed` |
| **Capabilities** | `read_mesh` |
| **Extra required** | none |
| **Returns** | `MeshDataset` |

## Dispatch

```python
from pyramids.netcdf.ugrid import MeshDataset

mesh = MeshDataset.read_file("model.fed", driver="riverflow2d")
bed = mesh.get("node_z")        # node-located bed elevation (ZB)
n = mesh.get("manning_n")       # face-located Manning's n (MANNINGN)
```

## What becomes what

- `X`, `Y` → node coordinates; `ZB` → a node-located `node_z` variable.
- `N1 N2 N3` → the triangular `face_node_connectivity` (1-based, re-based to 0-based).
- The per-cell data in a `.FED` is **element-located**, so these are **face** variables:
  `MANNINGN` → `manning_n`, `ELINITWSE` → `init_wse`, `ELMINERODELEV` → `min_erode_elev`.

## Documented limitations

- **Triangles only.** A line that does not carry three node ids is refused rather than
  guessed.
- **Ids are 1-based and consecutive.** The format numbers nodes `1..NNODES` in order; a file
  that breaks that is refused rather than silently mis-wired. The base is recorded for a
  writer.
- **Per-cell data is element-located.** `manning_n`, `init_wse` and `min_erode_elev` are
  face variables; the per-node columns of the same name are sentinel in every shipped mesh
  and are not surfaced. The `-9999` sentinel is recorded as `nodata` on `min_erode_elev`.
- **Boundary conditions are not here.** They live in the companion `.TBA` file, which this
  reader does not consult.
- **Results are not read here.** RiverFlow2D writes results as VTK/HDF5, read by the `vtk`
  and `xmdf`/`xdmf` drivers in this package.
- **No CRS.** A `.FED` carries projected metric coordinates but declares no authority, so
  `crs` is `None` rather than guessed.

## API

::: pyramids_mesh.drivers.river_hydraulics.riverflow2d.RiverFlow2dDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
