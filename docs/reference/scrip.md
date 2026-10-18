# SCRIP — remapping-grid NetCDF

The SCRIP reader. SCRIP (the Spherical Coordinate Remapping and Interpolation Package)
describes a grid for remapping as NetCDF: every cell is given by its corner coordinates
directly in `grid_corner_lat`/`grid_corner_lon` (shape `grid_size` × `grid_corners`), with
`grid_center_lat`/`grid_center_lon` the cell centres. There is no shared node list — the
same corner is repeated by every cell that touches it — so the driver deduplicates the
corners into a node table and builds face-node connectivity into it. Read through the
vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `scrip` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

A SCRIP grid is not UGRID, so `read_file` reaches the driver through its NetCDF-suffix
fallback:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("outCSne8.nc")   # corners -> shared nodes
mesh.mesh.n_face                               # the SCRIP cells
```

## Documented limitations

- **Corner dedup uses a rounded key, not float equality.** Shared corners are matched
  after rounding to a fixed tolerance; raw float comparison would leave duplicate nodes and
  split edges. The unique nodes are sorted, so the same file gives the same node order every
  run.
- **A repeated corner marks padding.** SCRIP fills a cell with fewer corners than the
  `grid_corners` width by repeating a corner; a corner that duplicates an earlier one in the
  same cell becomes `-1` fill, so a triangle in a quad-width table yields a fill-padded
  three-node face. SCRIP repeats a trailing corner, so the padding lands in the tail of the
  row as pyramids requires.
- **Coordinates are carried in the file's unit, degrees by default.** A `units` attribute
  naming radians is honoured and the coordinates converted to degrees; otherwise they are
  left as the file gave them.
- **No CRS.** The grid is lon/lat on a sphere with no EPSG authority, so `crs` is `None`
  rather than guessed.
- **`grid_imask` is not used to drop cells.** Every cell is kept; this driver declares
  `read_mesh` only and ships no variables.

## API

::: pyramids_mesh.drivers.scrip.ScripDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
