# SRH-2D

Reader for the **SRH-2D** (USBR/Aquaveo) ASCII mesh format (task C12).

An SRH-2D model stores its mesh as a **trio of ASCII files** sharing one basename stem. This
driver is the geometry/control reader; it reads the first and consults the other two for
metadata:

- `<case>.srhgeom` — the mesh. `Elem <id> n1 n2 n3[ n4 …]` lines carry mixed
  triangle/quad/polygon faces by 1-based node id, and `Node <id> x y z` lines the node
  coordinates and bed elevation. The `Name`, `GridUnit` and `NodeString` cards are read for
  metadata only.
- `<case>.srhhydro` — the control deck: `Grid "<case>.srhgeom"`, `HydroMat "<case>.srhmat"`,
  `ManningsN <material-id> <n>`, `BC` and `SimTime`.
- `<case>.srhmat` — the element-to-material-zone map, one `Material <id> <elem …>` block per
  zone (element ids continue onto the following unprefixed lines).

| | |
|---|---|
| **Driver name** | `srh2d` |
| **Suffixes** | `.srhgeom`, `.srhhydro` (either member of the trio) |
| **Capabilities** | `read_mesh` |

## Dispatch

Point `read_file` at either the `.srhgeom` or the `.srhhydro`; the driver resolves the rest
of the trio from the `Grid` / `HydroMat` cards, falling back to the shared basename stem:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("Muncie.srhgeom", driver="srh2d")
elevation = mesh.get("node_z")      # node-located bed elevation
zone = mesh.get("material")         # face-located material-zone id
```

## Documented limitations

- **The input is a file set.** A `.srhhydro` whose geometry cannot be resolved (no `Grid`
  card and no `<stem>.srhgeom` sibling) is declined by `can_open`.
- **Faces are mixed-width polygons.** Triangles, quads and larger polygons share one
  fill-padded connectivity array; short rows are padded with `-1`.
- **Node and element ids are resolved through an id map**, so a gapped or out-of-order file
  stays correctly wired rather than assuming `id == index + 1`. The ids are 1-based, which is
  recorded for a writer to restore.
- **Bed elevation becomes a `node_z` variable**, since the pyramids mesh model is planar and
  carries no per-node elevation coordinate.
- **Material and Manning's n are carried only when clean.** The `material` face variable is
  attached when the `.srhmat` assigns every element exactly one known zone; the `manning_n`
  face variable is attached on top of that when every assigned zone has a `ManningsN` value in
  the `.srhhydro`. An inconsistent or incomplete map is dropped rather than guessed.
- **Results are not read here.** SRH-2D writes its results as XMDF or VTK, which the `xmdf`
  and VTK drivers already read; this is the geometry/control reader.
- **SRH-2D carries no CRS**, so `crs` is always `None`. The `GridUnit` card is recorded as a
  `grid_unit` global attribute but never interpreted as a projection.
