# VTK / VTU — unstructured grid

The VTK reader. It reads the unstructured-grid format in both spellings: the legacy
`.vtk` ASCII file (`# vtk DataFile Version` magic, `DATASET UNSTRUCTURED_GRID`) and the
XML `.vtu` file (`<VTKFile type="UnstructuredGrid">`). Both are parsed with the standard
library alone — a text/token walk for the legacy form and `xml.etree.ElementTree` for the
XML form — so the driver declares no dependency and never imports `vtk`, `meshio` or any
other mesh library.

| | |
|---|---|
| Driver name | `vtk` |
| Suffixes | `.vtk`, `.vtu` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("surface.vtk")   # or surface.vtu
```

## Cell types

VTK tags every cell with a type code. The reader turns the 2D surface cells into faces and
drops the rest:

| Code | Cell | Mapped to |
|---|---|---|
| `5` | triangle | 3-node face |
| `9` | quad | 4-node face |
| `22` | quadratic triangle | 3 corner nodes (linearised) |
| `23` | quadratic quad | 4 corner nodes (linearised) |
| `10`, `24` | tetrahedron, quadratic tetrahedron | dropped (volumetric) |
| `12` | hexahedron | dropped (volumetric) |

A quadratic cell lists its corner nodes first and its mid-edge nodes after, so taking the
leading corners linearises it without moving the corner positions.

## Documented limitations

- **Only 2D surface cells become a mesh.** A volumetric cell (`10`/`12`/`24`/…) is
  dropped, and a grid with no surface cell at all is refused rather than returned as an
  empty mesh.
- **Only ASCII DataArrays are read.** An `appended` or `binary` (base64) `.vtu` payload is
  refused for v1, documented, rather than mis-decoded.
- **Only single-component `SCALARS` / DataArrays become variables.** A multi-component
  scalar, and the `VECTORS`/`NORMALS`/`TENSORS`/`FIELD` attribute kinds, are consumed so
  parsing stays aligned but left out of the result — the 2D scalar case ships first.
- **`POINT_DATA` is node-located, `CELL_DATA` face-located.** Cell scalars are sliced to
  the surface cells, so a grid mixing surface and volumetric cells keeps the right column.
- Point `(x, y, z)` with any non-zero `z` contributes a `node_z` variable; a grid flat at
  `z = 0` gets none. Values are eager — VTK is plain text/XML with nothing to defer. There
  is no CRS in the format, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.vtk.VtkDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
