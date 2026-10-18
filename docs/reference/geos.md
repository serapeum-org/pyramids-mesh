# GEOS — cubed-sphere NetCDF

The GEOS reader. GEOS (the NASA Global Modeling and Assimilation Office model) writes its
native cubed-sphere grid as NetCDF-4 with a logically structured corner grid per cube face:
`corner_lons`/`corner_lats` have shape `(6, ny, nx)` — six faces, each an `ny` × `nx`
lattice of corner points in degrees — and the optional `lons`/`lats` (`(6, ny-1, nx-1)`) are
the cell centres. The driver ravels the corner lattice into a node list and reconstructs the
quad face-node connectivity by index arithmetic, so the structured corner grid becomes an
unstructured surface mesh. Read through the vendored GDAL multidimensional API (no declared
dependency).

| | |
|---|---|
| Driver name | `geos` |
| Suffixes | `.nc`, `.nc4` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

A GEOS grid is not UGRID, so `read_file` reaches the driver through its NetCDF-suffix
fallback:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("test-c12.native.nc4")
mesh.mesh.n_face          # the cubed-sphere cells
```

## Documented limitations

- **Connectivity is computed, not read.** GEOS stores a structured corner lattice with no
  explicit connectivity, so the quad faces are reconstructed by index arithmetic — the four
  corners of each `(face, j, i)` cell.
- **Corners are not deduplicated across the ravel.** A face edge shared between cube panels
  carries two coincident nodes, one per panel.
- **Coordinates are carried in the file's degrees.** GEOS writes longitude and latitude in
  degrees, carried verbatim.
- **No CRS.** The grid is lon/lat on a sphere with no EPSG authority, so `crs` is `None`
  rather than guessed.
- **Only the topology is read.** The model's data fields are not shipped as variables; this
  driver declares `read_mesh` only.

## API

::: pyramids_mesh.drivers.geos.GeosDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
