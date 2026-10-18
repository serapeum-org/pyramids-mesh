# OBJ — Wavefront OBJ

The Wavefront OBJ reader. ASCII `v x y z` vertices and 1-based `f` faces, hand-parsed — a
geometry interchange format with elevation per vertex and no time axis.

> **No CRS, no georeferencing.** OBJ carries no coordinate reference system or spatial
> reference of any kind, so `crs` is always `None`. That makes it a low-value format for a
> GIS stack; it is provided for mesh interchange only.

| | |
|---|---|
| Driver name | `obj` |
| Suffixes | `.obj` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("surface.obj")
```

## Documented limitations

- **A round trip is not geometrically faithful.** `z` survives as a `node_z` variable,
  but pyramids' mesh model is planar, so a consumer reading the geometry sees the
  `(x, y)` projection. This is inherent to carrying a 3D surface in a 2D model (design §6).
- **OBJ carries no CRS and no georeferencing**, so `crs` is `None`.
- **The whole file is read eagerly** — a line-oriented text format has no record index to
  seek — so the driver does not declare `lazy_data`.
- Faces may be triangles, quads or higher n-gons; short rows pad to the widest with `-1`.
  A face of fewer than three vertices is refused.
- A face token may be a bare index or an `index/texture/normal` triple (`f v/vt/vn`); only
  the vertex index before the first `/` is used. Texture coordinates, vertex normals,
  grouping and material cards are read past and not carried.
- Face indices are 1-based in OBJ and arrive 0-based, with the base recorded for a writer.

## API

::: pyramids_mesh.drivers.obj.ObjDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
