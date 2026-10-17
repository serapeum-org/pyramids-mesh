# 3Di — results NetCDF

The 3Di reader. 3Di stores each 2D cell by its corner coordinates rather than a node list,
so the driver deduplicates the corners into shared nodes and builds face connectivity.
Read through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `3di` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `UgridDataset` |

3Di results use `.nc`, which is not UGRID. `UgridDataset.read_file` tries its built-in
UGRID reader first and, finding no topology, falls back to a driver claiming the suffix
(`pyramids-gis` 0.69.0's dispatch) — which is how this driver is reached.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("results_3di.nc")
water_level = mesh.get("Mesh2D_s1")   # face-located, over time
```

## Documented limitations

- **Corner dedup uses a rounded key, not float equality.** Shared corners are matched
  after rounding to a fixed tolerance; raw float comparison would leave duplicate nodes
  and split edges. The unique nodes are sorted, so the node order is deterministic.
- **A coupled 1D network is reported, not represented.** A file with `Mesh1D*` arrays
  carries a 1D model coupled to the 2D mesh; pyramids has no coupling relation (design
  §10), so the 2D mesh is returned and the 1D part is announced in a warning rather than
  dropped silently.
- **Flow-line variables are not shipped** — a result on the `nMesh2D_lines` dimension
  lives on flow links between cells, which the mesh model has no place for; only per-cell
  (face) variables are read.
- Values are lazy; the loader reopens the file per read. CRS comes only from a validated
  grid-mapping attribute, else `None`.

## API

::: pyramids_mesh.drivers.threedi.ThreeDiDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
