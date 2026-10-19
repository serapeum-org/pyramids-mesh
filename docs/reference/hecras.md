# HEC-RAS 2D — plan HDF5

Reader for the **HEC-RAS 2D** plan format (task D11).

A HEC-RAS plan file is an HDF5 tree holding one or more named 2D flow areas, each an
unstructured mesh of polygonal cells, plus unsteady results. The mesh nodes are the
`FacePoints`; each cell's polygon is reconstructed from its ordered faces
(`Cells Face and Orientation` into `Faces FacePoint Indexes`). The per-cell maximum/minimum
water-surface summary in the plan file becomes face-located variables.

| | |
|---|---|
| **Driver name** | `hecras` |
| **Suffixes** | `.hdf`, `.hdf5` |
| **Capabilities** | `read_mesh`, `read_data`, `lazy_data` |
| **Extra required** | none |
| **Returns** | `UgridDataset` |

`.hdf` is shared with other HDF5 formats, so the sniff keys on the HEC-RAS group layout
(`Geometry/2D Flow Areas` with a `Faces FacePoint Indexes` geometry), not the suffix.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.p01.hdf")
wse = mesh.get("Maximum Water Surface")   # face-located, per cell
```

## Documented limitations

- **Only the first 2D flow area is read.** A plan file may hold several; selecting a second
  2D mesh by name is not wired through the read template yet, so the first area is read and
  the others are named in a warning rather than dropped silently.
- **Ghost/boundary cells are excluded.** HEC-RAS appends perimeter cells (a single face
  each) after the real cells; `Cell Info` gives the real-cell count, and only those closed
  polygons become faces. The per-cell summary arrays, sized to include the ghost cells, are
  sliced to the real cells to match.
- **Summary results only.** The plan file carries the maximum/minimum water-surface
  summary; the per-timestep series lives in a separate, geometry-stripped `*.timeseries.hdf`
  companion, which cannot be paired into a single read here.
- **Face velocities are not exposed** — they are edge-located (one per face, not per cell);
  this reader builds the cell mesh only.
- **Values are lazy** — the loader reopens the file per read, closing over a path.
- There is no CRS in the arrays read here, so `crs` is `None`.

HDF5 is read through the GDAL multidimensional API over the GDAL vendored in `pyramids-core`
(activated by `import pyramids`) — no declared dependency and no `h5py`.

## API

::: pyramids_mesh.drivers.hecras.HecRasDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
