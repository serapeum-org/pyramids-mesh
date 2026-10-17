# Esri TIN — directory of `.adf` files

The Esri TIN reader. An Esri TIN is a directory of big-endian binary `.adf` components, not
a single file. Read through plain binary parsing and the vendored GDAL for the projection
(no declared dependency).

| | |
|---|---|
| Driver name | `esri_tin` |
| Suffixes | `.adf` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset` |

A directory has no suffix, so `read_file` cannot dispatch a bare directory path. Reach the
driver either way:

```python
from pyramids.netcdf.ugrid import UgridDataset

# point at any .adf component of the TIN
mesh = UgridDataset.read_file("path/to/tin/tdenv9.adf")

# or pass the directory with an explicit driver
mesh = UgridDataset.read_file("path/to/tin", driver="esri_tin")

elevation = mesh.get("node_z")
```

## Documented limitations

- **The input is a directory of `.adf` files.** `read_file` reaches it through any `.adf`
  component, or through `driver="esri_tin"` on the directory. A raster `.adf` (an Esri
  GRID), whose directory has no triangle table, is declined.
- **Masked triangles are dropped.** The convex-hull frame triangles flagged in `tmsk.adf`
  are excluded, so the mesh is the surface the TIN represents; the frame nodes they
  referenced stay in the node list, unreferenced.
- **Triangle indices are 1-based** — resolved to 0-based, the base recorded for a writer.
  Every face is a triangle; Esri's clockwise triangles are reversed to counter-clockwise.
- **The projection is read from `prj.adf`.** The Esri-dialect WKT is morphed from Esri
  form; it resolves to an EPSG code when one is named, else the normalised WKT, else
  `None` — never a guess.
- Node elevation (`tnz.adf`) becomes a `node_z` variable.

## API

::: pyramids_mesh.drivers.esri.EsriTinDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
