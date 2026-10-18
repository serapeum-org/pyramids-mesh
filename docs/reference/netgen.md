# Netgen — volume mesh

The Netgen reader. It reads the Netgen `.vol` ASCII file (and its gzip-compressed
`.vol.gz` form): a `mesh3d` preamble with a `dimension` and a `geomtype` setting, then one
section per topology kind — each a keyword line, a count line, and that many rows. The
file is parsed with the standard library alone — a line-record walk, through
`gzip` for a compressed file — so the driver declares no dependency and never imports a
mesh library.

| | |
|---|---|
| Driver name | `netgen` |
| Suffixes | `.vol`, `.vol.gz` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.vol")   # or domain.vol.gz
```

## Sections

Netgen lists each topology kind in its own section. The reader turns the surface
triangles into faces and skips the rest:

| Section | Row | Used for |
|---|---|---|
| `points` | `x y z` | node coordinates (`node_x`, `node_y`, `node_z`) |
| `surfaceelements` | `surfnr bcnr domin domout np p1…pnp` | the `np` trailing ids become a face |
| `volumeelements` | `matnr np p1…p4` | dropped (volumetric) |
| `materials`, `bcnames`, `edgesegmentsgi2`, `identifications`, … | — | skipped (metadata) |

A `surfaceelements` row carries four leading columns — the surface number, the boundary
condition, and the two adjacent domains — then the node count `np` and that many 1-based
node ids. Only the trailing ids are kept.

## Documented limitations

- **Only the surface becomes a mesh.** The `surfaceelements` triangles become faces; the
  `volumeelements` tetrahedra are dropped rather than boundary-extracted. A faithful
  volume read needs the layered mesh model pyramids does not have yet, and reconstructing
  the boundary from the tetrahedra could not be verified against a sample, so the surface
  the file already lists is kept instead.
- **The optional trailing sections are metadata, not topology.** `materials`, `bcnames`,
  `identifications`, `edgesegmentsgi2` and the rest of the Netgen table are skipped; only
  `points` and `surfaceelements` are read.
- **The node `z` column is carried as a `node_z` variable**, not as a coordinate — the
  mesh model is planar. Unlike the 2DM and VTK readers, a flat `z = 0` is kept rather than
  omitted: Netgen is a 3D format whose points always declare a real elevation.
- **Node ids are 1-based.** The connectivity arrives 0-based with the base `1` recorded for
  a writer. Values are eager — a `.vol` file is plain text with nothing to defer. There is
  no CRS in the format, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.netgen.NetgenDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
