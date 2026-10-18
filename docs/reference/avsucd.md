# AVS-UCD — unstructured cell data

The AVS-UCD (`.inp`/`.avs`) reader. The format — written by AVS and by LANL's LaGriT — opens
with a five-integer header (`n_nodes n_cells n_nodedata n_celldata 0`), then one row per
node (`node_id x y z`) and one per cell (`cell_id material_id cell_type v1..vn`).

| | |
|---|---|
| Driver name | `avsucd` |
| Suffixes | `.inp`, `.avs` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("mesh.avs")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Element handling

Each cell's `cell_type` keyword (`tri`/`quad`/`tet`/`pyr`/`prism`/`hex`) names its topology.
Following the Exodus reader's precedent, every area/volume cell maps its node list to a
fill-padded face row (a `tet` becomes a 4-node face, exactly as Exodus maps `TETRA`); `pt`
and `line` cells are dropped. The `material_id`/`cell_type` columns and the optional
node-data and cell-data sections are discarded — this is a geometry reader.

## Documented limitations

- Volume cells are mapped as faces rather than boundary-extracted (derivation is pyramids'
  concern — ADR-0005); this keeps a solid mesh readable rather than refusing it.
- There is no CRS in the format — engineering coordinates, no datum — so `crs` is `None`.
- **The `.inp` suffix is shared with Abaqus.** `can_open` requires the AVS-UCD signature
  (the first significant line is exactly five integers), so an Abaqus `.inp` — whose first
  keyword line begins with `*` — is declined.

## API

::: pyramids_mesh.drivers.avsucd.AvsUcdDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
