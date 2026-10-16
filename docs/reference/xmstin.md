# XMS TIN — SMS triangulated network

The XMS / SMS TIN reader. A text triangulated irregular network: a `VERT` block of
`x y z` vertices and a `TRI` block of 1-based triangles.

| | |
|---|---|
| Driver name | `xmstin` |
| Suffixes | `.tin` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("surface.tin")
```

## Documented limitations

- **A round trip is not geometrically faithful**, for the same reason as PLY: the
  elevation survives as `node_z` data, but the mesh geometry is the planar projection
  (design §6).
- Triangle vertex indices are **1-based** and arrive 0-based, with the base recorded for
  a writer.
- A vertex's optional lock flag is read for width but not carried — it is an
  editing-state marker the mesh model has no place for.
- XMS TIN carries no CRS, so `crs` is `None`.

## Esri TIN

The task that produced this reader also named **Esri TIN**, which is deliberately *not*
implemented: it is an undocumented directory of binary files, reverse-engineered by MDAL,
and there is no vendor sample to test an honest parser against. It is tracked for when a
real ArcGIS TIN directory is available.

## API

::: pyramids_mesh.drivers.interchange.XmsTinDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
