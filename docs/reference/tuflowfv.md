# TUFLOW FV — results NetCDF (2D)

The TUFLOW FV reader, reading the **2D slice**: the surface mesh and its per-cell
variables. Read through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `tuflowfv` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `UgridDataset` |

TUFLOW FV results use `.nc`, which is not UGRID, so they are reached through `read_file`'s
NetCDF-suffix fallback (`pyramids-core` 0.69.0).

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("results.nc")
depth = mesh.get("D")      # face-located, over time
wet = mesh.get("stat")     # per-step wet/dry status
```

## Documented limitations

- **A layered 3D file is refused.** When the file has more 3D cells than 2D cells
  (`NumCells3D` > `NumCells2D`), its variables stack over vertical levels this package
  does not model yet (design §21.1); it is refused rather than flattened. A single-layer
  file reads normally, its `NumCells3D` variables mapped to faces.
- **`cell_node` is 1-based with `0` padding** — a `0` marks a triangle in the quad-width
  table and becomes `-1` fill; the base is recorded for a writer.
- **`stat` is carried as a face-located variable**, not folded into the values as NaN
  (design §21.2): the per-step wet/dry status stays inspectable.
- Values are lazy; the loader reopens the file per read. There is no CRS in the format
  read here, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.tuflowfv.TuflowFvDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
