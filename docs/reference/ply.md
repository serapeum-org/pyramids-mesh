# PLY — Stanford polygon

The Stanford PLY reader. ASCII and both binary byte orders, hand-parsed — a geometry
interchange format with elevation per vertex and no time axis.

| | |
|---|---|
| Driver name | `ply` |
| Suffixes | `.ply` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("surface.ply")
```

## Documented limitations

- **A round trip is not geometrically faithful.** `z` survives as a `node_z` variable,
  but pyramids' mesh model is planar, so a consumer reading the geometry sees the
  `(x, y)` projection. This is inherent to carrying a 3D surface in a 2D model (design §6).
- **The whole file is read eagerly.** A PLY from a 3D scanner can be large; there is no
  record index to seek, so every vertex property and face is read at once. The driver
  does not declare `lazy_data`.
- **Extra vertex properties become node variables.** Colours, normals and the like are
  carried as their own node-located variables rather than dropped.
- Faces may be triangles, quads or mixed; short rows pad to the widest with `-1`. A face
  of fewer than three vertices (an edge or a point) is refused.
- PLY carries no CRS and no units, so `crs` is `None`. Indices are 0-based and stay so.

## API

::: pyramids_mesh.drivers.interchange.PlyDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
