# FLAC3D — `.f3grid` grid

The FLAC3D (`.f3grid`) reader. The Itasca geomechanics grid is a plain-text file of
gridpoints and zones: `G <id> <x> <y> <z>` rows give node coordinates and
`Z <zonetype> <id> <gp1>..<gpN>` rows give each zone's gridpoint list; `*` begins a comment.

| | |
|---|---|
| Driver name | `flac3d` |
| Suffixes | `.f3grid` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.f3grid")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Element handling

FLAC3D zones are solid cells — `T4` (tet, 4), `W6` (wedge, 6), `B8` (brick, 8), `P5`
(pyramid, 5). Following the Exodus reader's precedent, each zone maps its gridpoint list to
a fill-padded face row (padded to the widest zone) rather than being boundary-extracted
(connectivity derivation is pyramids' concern — ADR-0005). `ZGROUP`/`FGROUP` tables are read
past and ignored.

## Documented limitations

- **Gridpoint ids are arbitrary labels**, resolved to contiguous 0-based indices with the
  1-based base recorded for a writer, so a gapped or reordered file wires correctly.
- **Node elevation** — the `z` column becomes a `node_z` variable when any value is
  non-zero, matching the Exodus and gmsh readers.
- There is no CRS in the format — engineering coordinates, no datum — so `crs` is `None`.
- The mesh is a solid volume mapped cell-by-cell to faces; it is not a true 2-D surface.

## API

::: pyramids_mesh.drivers.flac3d.Flac3dDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
