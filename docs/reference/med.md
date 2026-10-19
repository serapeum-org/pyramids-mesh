# MED-file — Salome / code_aster

The MED-file reader. A `.med` file is a plain HDF5 container. A mesh lives under
`ENS_MAA/<mesh>/<step>` with its node coordinates in `NOE/COO` — a flat, coordinate-major
block (all x, then all y, then all z) — and its elements in `MAI/<geomcode>` groups, one per
element type (`TR3`, `QU4`, `TE4`, `HE8`, …), each with a flat, 1-based, coordinate-major
`NOD` connectivity list. Because `.med` is not auto-sniffed by GDAL, the file is opened by
forcing the HDF5 multidimensional driver, over the GDAL vendored in the `pyramids-core` wheel
(no declared dependency, no `h5py`/`med` library).

| | |
|---|---|
| Driver name | `med` |
| Suffixes | `.med` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("cylinder.med")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Documented limitations

- **Only the first mesh (and its first step) is read.** A multi-mesh file is read for its
  first `ENS_MAA` entry only; selecting another is not wired through the read template yet.
- **Surface groups are preferred; volumetric groups are a fallback.** When a mesh carries
  any surface group (`TR*`/`QU*`), only those become faces — so a 3D volume mesh with a
  surface skin returns the surface. A mesh carrying only volumetric groups (e.g.
  `TE4`/`HE8`) maps those as faces verbatim, each cell's node list taken as a polygon rather
  than expanded into its bounding faces: a genuine 3D mesh is not modelled yet (design
  §21.1), and winding is judged in the `xy` plane like any other face. Boundary extraction
  is pyramids' job (ADR-0005), not this reader's.
- **Segment (`SE2`/`SE3`) groups are skipped** — they are 1D element types, not surface
  faces.
- **Connectivity is 1-based with mixed widths padded by `0`** — a short row in a
  mixed-element table (triangles among quads) is padded with `0`, never a valid 1-based id,
  and the normaliser rewrites it to `-1` fill; the `1` base is recorded for a writer.
- **Families, fields and results are not read** — the `FAM`/`NUM` arrays and any `CHA` field
  groups a MED file carries are out of scope for this reader.
- The `NOE/COO` z column (a 3D file) contributes a `node_z` variable; a 2D file leaves it
  unset. There is no CRS in the arrays read here, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.med.MedDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
