# OFF — Object File Format

The OFF (Object File Format) reader. An ASCII `OFF` magic line, a `nVerts nFaces nEdges`
count line, then `x y z` vertices and `k v0 … v{k-1}` faces — hand-parsed, a geometry
interchange format with elevation per vertex and no time axis.

> **No CRS, no georeferencing.** OFF carries no coordinate reference system or spatial
> reference of any kind, so `crs` is always `None`. That makes it a low-value format for a
> GIS stack; it is provided for mesh interchange only.

| | |
|---|---|
| Driver name | `off` |
| Suffixes | `.off` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("surface.off")
```

## Documented limitations

- **A round trip is not geometrically faithful.** `z` survives as a `node_z` variable,
  but pyramids' mesh model is planar, so a consumer reading the geometry sees the
  `(x, y)` projection. This is inherent to carrying a 3D surface in a 2D model (design §6).
- **OFF carries no CRS and no georeferencing**, so `crs` is `None`.
- **The whole file is read eagerly** — a line-oriented text format has no record index to
  seek — so the driver does not declare `lazy_data`.
- **Per-face colour is ignored.** A face line may carry per-face RGB/RGBA colour after its
  vertex ids; those trailing numbers have no home on a planar mesh and are read past.
- A `#` begins a comment that runs to the end of the line (whole-line or trailing a
  vertex), and blank lines are ignored.
- Faces may be triangles, quads or higher n-gons; short rows pad to the widest with `-1`.
  A face of fewer than three vertices is refused.
- The declared edge count (the third header field) is advisory and ignored.
- Face indices are 0-based in OFF and stay so. The coloured and higher-dimensional magic
  variants (`COFF`, `4OFF`, `nOFF`) are not supported and are declined.

## API

::: pyramids_mesh.drivers.off.OffDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
