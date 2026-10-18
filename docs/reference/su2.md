# SU2 — `.su2` native mesh

The SU2 reader. SU2 is the native ASCII mesh of the SU2 open-source CFD/multiphysics
suite: a flat, section-keyed text file holding a dimension card, a VTK-typed interior
element table, a node table, and named boundary markers.

| | |
|---|---|
| Driver name | `su2` |
| Suffixes | `.su2` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.su2")   # resolved to this package's su2 driver
mesh.mesh.n_face
```

## File shape

```
NDIME= 2                      dimension: 2 or 3
NELEM= 25                     interior element table
 9  0  1  7  6  0             <vtk-type> n1 n2 … [element-id]
 …
NPOIN= 36                     node coordinates
 0.0  0.0  0                  x y [z] [point-id]
 …
NMARK= 4                      boundary markers
MARKER_TAG= bottom
MARKER_ELEMS= 5
 3  0  1                      line/edge elements (VTK type 3)
 …
```

The VTK cell codes are triangle `5`, quadrilateral `9`, line `3`, tetra `10` and hexa
`12`. A node table row indexes nodes directly — SU2 is **0-based**.

## Documented limitations

- **Surface meshes only.** Triangle (code 5) and quadrilateral (code 9) elements become
  the fill-padded `face_node_connectivity`; a mixed-width file pads the short rows with
  `-1`. A volume element type (tetra 10, hexa 12, or any other 3D cell) in the `NELEM`
  table is refused rather than flattened to a face — there is no committed 3D sample to
  validate such a read against.
- **0-based, no id map.** SU2 element rows index the node table directly, so the
  connectivity is already 0-based: `start_index` and `original_start_index` are both `0`
  and the validator subtracts nothing. The optional trailing element/point index column
  each row may carry is ignored, exactly as SU2 ignores it on read.
- **Boundary markers are metadata, not topology.** Each `MARKER_TAG` group is recorded as
  `su2_markers` on the global attributes (tag → boundary-element count); the marker edges
  are not built into a separate 1D network.
- **No CRS.** SU2 carries no coordinate-reference declaration, so `crs` is `None` and is
  never guessed from the coordinate magnitudes.

## API

::: pyramids_mesh.drivers.su2.Su2Driver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
