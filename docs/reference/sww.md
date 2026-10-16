# SWW — ANUGA

The ANUGA SWW reader, and the package's first NetCDF-schema driver. It reads through the
vendored GDAL multidimensional API — the same substrate the HDF5 drivers use, activated
by `import pyramids`, with no declared `netCDF4` or `h5py`.

| | |
|---|---|
| Driver name | `sww` |
| Suffixes | `.sww` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

dataset = UgridDataset.read_file("model.sww")
stage = dataset.get("stage").data   # deferred until this line
```

`.sww` is not one of pyramids' NetCDF suffixes, so a plain `read_file` reaches this driver
through the registry. A SWW that happened to be named `.nc` would not — that is the
upstream dispatch limitation tracked for the 3Di and TUFLOW FV drivers, not a concern
here.

## What it reads

`x`, `y` (node coordinates), `volumes` (triangle connectivity), `elevation` or `z` (node
elevation), `time`, and per-timestep node results such as `stage`, `xmomentum`,
`ymomentum`.

## Documented limitations

- **The connectivity base is asserted, not assumed.** ANUGA is 0-based, and an off-by-one
  shifts every face by one node while still producing a plausible-looking mesh. The base
  is taken from `volumes.min()` and must be `0` or `1`; anything else is refused.
- **A variable's location is resolved by its NetCDF dimension name, not its length.** In a
  tiny file the node and triangle counts can coincide, so a variable on the points
  dimension is node-located and one on the volumes dimension is face-located — decided by
  name, where length would be ambiguous.
- **A variable on neither the point nor the volume dimension is skipped**, with a warning
  naming it. The mesh model has nowhere to put a value indexed by something else.
- **`stage` is carried as the file writes it** — absolute water level, not depth. It is
  not renamed or turned into a derived quantity.
- **CRS comes only from a validated attribute.** SWW usually stores a zone or projection
  label rather than a full CRS; a value is used only when the vendored PROJ accepts it,
  and is otherwise left `None` rather than guessed from a UTM zone.
- Node elevation becomes a `node_z` variable, never a coordinate (design §6).
- **Every variable is lazy**, elevation included: the topology is read eagerly to build
  the mesh, and each variable is left behind a loader that reopens the file and reads its
  one array.

## API

::: pyramids_mesh.drivers.anuga.SwwDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
