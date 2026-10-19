# ICON — icosahedral grid NetCDF

The ICON reader. ICON (the DWD/MPI weather and climate model) stores its unstructured
triangular grid as NetCDF: cell centres in `clon`/`clat`, triangle vertices in
`vlon`/`vlat`, and the 1-based `vertex_of_cell` table (shape `nv=3` × `ncells`). The
driver builds the surface mesh — the vertices become the nodes and `vertex_of_cell` the
triangular face connectivity — and ships `cell_area` as a static face variable. Read
through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `icon` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

ICON grids use `.nc`, which is not UGRID. `UgridDataset.read_file` tries its built-in
UGRID reader first and, finding no topology, falls back to a driver claiming the suffix
(`pyramids-core` 0.69.0's dispatch) — which is how this driver is reached.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("icon_grid.nc")
area = mesh.get("cell_area")   # face-located cell areas
```

## Documented limitations

- **Coordinates are kept in radians.** ICON stores longitude and latitude in radians, and
  this driver preserves them verbatim: node and face-centre coordinates come back in
  radians, not degrees. Converting to degrees is a projection concern that belongs in the
  GIS layer upstream (pyramids), not in a file-format driver, so the native unit is carried
  unchanged.
- **No CRS.** ICON declares a sphere (`grid_mapping_name = lat_long_on_sphere`,
  `semi_major_axis` ≈ 6371229 m) but no EPSG authority, and radian coordinates do not match
  a standard degree-based geographic CRS, so `crs` is `None` rather than a misleading
  geographic CRS.
- **`vertex_of_cell` is 1-based** — the `nv=3` × `ncells` table is transposed to
  `ncells` × 3, resolved to 0-based, and the base recorded for a writer.
- **MPAS is declined.** A `.nc` carrying MPAS's `verticesOnCell` array or the MPAS
  convention is not claimed, so the ICON and MPAS drivers never both open one file.
- Values are eager: an ICON grid is a static mesh, so `cell_area` is read at parse time
  rather than behind a loader.

## API

::: pyramids_mesh.drivers.icon.IconDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
