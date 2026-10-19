# GiD — `.msh` ASCII mesh (Iber)

The GiD / CIMNE reader. GiD writes a mesh as one or more plain-text blocks, each opening
with a `MESH … dimension … ElemType … Nnode …` header, then a `Coordinates` block and an
`Elements` block. This is the format the free **Iber** 2D hydraulic model exports, since
Iber is built on GiD/CIMNE (issue #68). The text is parsed directly — no declared
dependency.

| | |
|---|---|
| Driver name | `gid` |
| Suffixes | `.msh` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

# an Iber / GiD export dispatches by the .msh suffix
mesh = UgridDataset.read_file("tube.GiD.msh")

faces = mesh.mesh.face_node_connectivity   # triangles and quadrilaterals
```

## Documented limitations

- **Higher-order surface elements are reduced to their corners.** A six-node quadratic
  `Triangle` (`Nnode 6`) or an eight-node `Quadrilateral` lists its corners first and its
  mid-edge nodes after; only the corners (3 for a triangle, 4 for a quadrilateral) form
  the face polygon, so the mid-edge nodes are dropped from the connectivity — the same
  stance the CGNS and MED readers take. The mid-edge nodes remain in the node table,
  simply unreferenced.
- **`Line` and `Point` elements are dropped.** They are lower-dimensional scaffolding, not
  surface faces; folding them into the face table would mislabel them.
- **A volumetric `Tetrahedra`/`Hexahedra` block becomes faces verbatim.** Its node list is
  read as a polygon rather than expanded into bounding faces — a genuine 3D mesh is not
  modelled yet — exactly as the Exodus reader does. A file whose only elements are
  `Line`/`Point` (or an unrecognised type) is refused for having no surface.
- **Node ids may be gapped or out of order** — resolved through a label→index map, never
  by assuming `id == index + 1`; the 1-based GiD base is recorded for a writer. A node id
  re-listed across blocks keeps its first definition. Clockwise faces are reversed to
  counter-clockwise.
- **An unclosed `Coordinates`/`Elements` block is refused**, rather than reading a
  truncated mesh.
- **Only the mesh is read.** A trailing per-element material column is ignored; GiD result
  (`.res`) data is a separate format.
- **The `.msh` suffix is shared with the Gmsh reader.** `can_open` requires the GiD
  signature (a first significant line starting with `MESH` carrying `dimension` and
  `ElemType`), so a Gmsh `$MeshFormat` file is declined and the two never both claim a
  file.
- **No CRS.** GiD carries engineering coordinates with no datum, so `crs` is `None` rather
  than guessed. Node elevation becomes a `node_z` variable only when some node has a
  non-zero z.

## API

::: pyramids_mesh.drivers.gid.GidDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
