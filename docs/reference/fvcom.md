# FVCOM native grid — trio of `.dat` files

The FVCOM native grid reader. An FVCOM grid is not a single file but a trio of ASCII files
sharing a basename: `<base>_grd.dat` (topology), `<base>_dep.dat` (bathymetry) and
`<base>_obc.dat` (open boundary). Read through plain text parsing — no declared dependency
beyond numpy.

| | |
|---|---|
| Driver name | `fvcom` |
| Suffixes | `.dat` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

`.dat` is a generic suffix, so the driver sniffs the `Node Number` / `Cell Number` header
and is reached through the `_grd.dat` component or the directory holding it:

```python
from pyramids.netcdf.ugrid import UgridDataset

# point at the _grd.dat component of the trio
mesh = UgridDataset.read_file("path/to/chn_grd.dat")

# or pass the directory with an explicit driver
mesh = UgridDataset.read_file("path/to/grid_dir", driver="fvcom")

depth = mesh.get("node_z")
```

## Documented limitations

- **The input is the `_grd.dat` file (or its directory).** The matching `_dep.dat` and
  `_obc.dat` are resolved by basename beside it; either may be absent, and a grid read on
  its own still yields the topology. A foreign `.dat` is declined by the header sniff.
- **Every element is a triangle.** The grid stores three node ids per cell, so no fill
  padding is ever needed.
- **Node and cell ids are 1-based** — resolved to 0-based, the base recorded for a writer.
  The clockwise-wound triangles are reversed to counter-clockwise.
- **Depth comes from `_dep.dat`.** Its `depth` column becomes a `node_z` node variable;
  the `z` column of `_grd.dat` is ignored. Without a `_dep.dat` there is no `node_z`.
- **Open-boundary nodes become metadata.** The `_obc.dat` node ids are carried (0-based) on
  the dataset's global attributes as `open_boundary_nodes`, not as a topology of their own.
- There is no CRS in the files read here, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.fvcom.FvcomDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
