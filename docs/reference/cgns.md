# CGNS/HDF5 — CFD General Notation System

The CGNS reader. A `.cgns` file is a plain HDF5 container holding a SIDS tree: a
`CGNSBase_t` base, one or more `Zone_t` zones, each with `GridCoordinates`
(`CoordinateX`/`CoordinateY`/`CoordinateZ`) and one or more `Elements_t` sections. A section
names an element-type code (`TRI_3=5`, `QUAD_4=7`, `TETRA_4=10`, `HEXA_8=17`, …) and carries
a flat, 1-based `ElementConnectivity` list plus an `ElementRange`. Because `.cgns` is not
auto-sniffed by GDAL, the file is opened by forcing the HDF5 multidimensional driver, over
the GDAL vendored in the `pyramids-core` wheel (no declared dependency, no `h5py`/`CGNS`
library).

| | |
|---|---|
| Driver name | `cgns` |
| Suffixes | `.cgns` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("grid.cgns")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Documented limitations

- **Only the first base and first zone are read.** A multi-base or multi-zone file is read
  for its first `CGNSBase_t` / `Zone_t` only; selecting another is not wired through the
  read template yet.
- **Surface sections are preferred; volumetric sections are a fallback.** When a zone
  carries any surface section (`TRI_*`/`QUAD_*`), only those become faces — so a 3D volume
  mesh with boundary patches returns its surface skin. A zone carrying only volumetric
  sections (e.g. `HEXA_8`) maps those as faces verbatim, each cell's node list taken as a
  polygon rather than expanded into its bounding faces: a genuine 3D mesh is not modelled
  yet (design §21.1), and winding is judged in the `xy` plane like any other face. Boundary
  extraction is pyramids' job (ADR-0005), not this reader's.
- **`MIXED` sections and 0D/1D codes are skipped.** A `MIXED` section interleaves a
  per-element type code with the node lists and has no constant width, so it is dropped
  rather than misread; `NODE`/`BAR` sections are not surface faces and are skipped too.
- **Connectivity is 1-based with mixed widths padded by `0`** — a short row in a
  mixed-element table (triangles among quads) is padded with `0`, never a valid 1-based id,
  and the normaliser rewrites it to `-1` fill; the `1` base is recorded for a writer.
- **Results and flow solutions are not read** — only the mesh is built; a `FlowSolution_t`
  an analysis file may carry is out of scope for this reader.
- `CoordinateZ` (a 3D grid) contributes a `node_z` variable; a planar grid leaves it unset.
  There is no CRS in the arrays read here, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.cgns.CgnsDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
