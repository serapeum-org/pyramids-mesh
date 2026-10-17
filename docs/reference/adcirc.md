# ADCIRC — `fort.14` grid + `fort.63.nc` results

The ADCIRC reader. ADCIRC is a coastal-circulation model whose mesh is the ASCII grid
`fort.14` (a node list then a triangular element table) and whose scalar results are
`fort.63` / `fort.63.nc` (a sea-surface-elevation time series on the nodes). The NetCDF
results are read through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `adcirc` |
| Suffixes | `.14` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `UgridDataset` |

`fort.14`/`fort.63` have no file suffix and `fort.63.nc` ends `.nc`, so — as with the Esri
TIN directory format — `read_file` reaches the driver through the `fort.14` file or its
directory:

```python
from pyramids.netcdf.ugrid import UgridDataset

# point at the fort.14 grid (its fort.63.nc sibling, if present, is attached)
mesh = UgridDataset.read_file("run/fort.14")

# or pass the run directory with an explicit driver
mesh = UgridDataset.read_file("run", driver="adcirc")

zeta = mesh.get("zeta")       # node-located sea-surface elevation, over time
bed = mesh.get("node_z")      # ADCIRC bathymetric depth
```

## Documented limitations

- **The input is a `fort.14` file (or its directory).** `fort.14` has no suffix, so
  `read_file` reaches it through the `fort.14` path, or through `driver="adcirc"` on the
  directory.
- **Triangular elements only.** ADCIRC grids are triangulations; a non-triangular element
  type is refused rather than guessed.
- **Element ids are 1-based** — resolved to 0-based, the base recorded for a writer.
- **Results come from `fort.63.nc` only.** A plain ASCII `fort.63` is not read here; without
  a `fort.63.nc` sibling the bare mesh is returned. Each `(time, node)` array in the NetCDF
  becomes a node-located variable over the `time` axis; values are lazy.
- **No CRS.** `fort.14` carries lon/lat but declares no authority, so `crs` is `None` rather
  than guessed. `depth` is ADCIRC bathymetric depth (positive down), carried verbatim as
  `node_z`.

## API

::: pyramids_mesh.drivers.adcirc.AdcircDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
