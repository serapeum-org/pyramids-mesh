# XDMF — XML index over HDF5

The XDMF reader. The `.xdmf`/`.xmf` file is an XML index naming the mesh topology and
geometry and, for a time series, the per-step result slabs; every heavy array lives in a
companion HDF5 file (`file.h5:/dataset`). The XML is parsed with the standard library and
the HDF5 arrays through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `xdmf` |
| Suffixes | `.xdmf`, `.xmf` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("out.xdmf")
phi = mesh.get("phi")      # node-located, over time
a = mesh.get("a")          # cell (face) located, over time
```

## Documented limitations

- **Only 2D surface topologies are read** (`Triangle`, `Quadrilateral`). A volumetric
  topology (`Tetrahedron`/`Hexahedron`/…) is a 3D mesh whose vertical structure is not
  modelled yet (design §21.1); it, and the `Mixed`/higher-order/`Polyline` types, are
  refused rather than flattened.
- **Vector attributes are skipped** — the 2D scalar case ships first; a `Vector` attribute
  (e.g. velocity components) is left out rather than mislabelled as a scalar.
- **Values are lazy** — each result variable's loader reopens the HDF5 payload per read and
  stacks the per-step slabs, closing over paths, never an open handle (design §7); shapes
  come from the XML without forcing a read.
- `GeometryType="XYZ"` contributes a `node_z` variable; `XY` leaves it unset. There is no
  CRS in the format, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.xdmf.XdmfDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
