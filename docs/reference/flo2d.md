# FLO-2D

Reader for the **FLO-2D** floodplain-grid format (task D10).

A FLO-2D project is a **directory** of interdependent files. This driver reads two of them:

- `DEPFP.OUT` — one line per element, `<id> <x> <y> <max floodplain depth>`. FLO-2D elements
  are square cells on a regular grid; the line gives each cell centre.
- `TIMDEP.HDF5` — the time-varying output, an HDF5 tree whose `TIMDEP NETCDF OUTPUT RESULTS`
  group holds one sub-group per quantity, each with `Values` `(n_time, n_element)`, `Times`,
  `Mins` and `Maxs`.

| | |
|---|---|
| **Driver name** | `flo2d` |
| **Suffixes** | `.out` (the `DEPFP.OUT` component; the input is really a directory) |
| **Capabilities** | `read_mesh`, `read_data`, `lazy_data` |

## Dispatch

A directory has no suffix, so point `read_file` at the `DEPFP.OUT` component, or pass the
directory with `driver="flo2d"`:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("flo2d_project", driver="flo2d")
depth = mesh.get("FLOW DEPTH")      # face-located, over time
```

## Documented limitations

- **The input is a directory.** `DEPFP.OUT` (element centres) and `TIMDEP.HDF5` (time series)
  must both be present; a lone component elsewhere is declined by `can_open`.
- **Square cells become quad faces.** Each element is a square whose side is the grid spacing;
  four corner nodes are built per cell and shared corners deduplicated. The element centres are
  carried as the face centres.
- **Only 2D scalar results are shipped.** Each `TIMDEP` quantity whose `Values` is
  `(n_time, n_element)` becomes a face-located variable over its `Times` axis; a vector quantity
  (e.g. `Velocity`, with a third axis) is skipped — its `Velocity MAG` magnitude is the scalar
  form. A genuinely 3D result is not modelled yet.
- **Values are lazy** — the loader reopens the file per read, closing over a path.
- `Mins`/`Maxs` ride along as per-timestep `minimum`/`maximum` attributes, and the `DEPFP.OUT`
  max-depth column is a static `max_floodplain_depth` face variable.
- There is no elevation or CRS in the files read here, so `node_z` and `crs` are `None`.

HDF5 is read through the GDAL multidimensional API over the GDAL vendored in `pyramids-core`
(activated by `import pyramids`) — no declared dependency and no `h5py`.
