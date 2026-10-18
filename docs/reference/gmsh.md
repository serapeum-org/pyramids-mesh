# Gmsh — `.msh` ASCII mesh

The Gmsh reader. Gmsh is a general mesh generator whose `.msh` files store nodes and
elements in plain text; this driver reads both the legacy **v2.2** record layout and the
current **v4.1** entity-block layout. The text is parsed directly — no declared
dependency.

| | |
|---|---|
| Driver name | `gmsh` |
| Suffixes | `.msh` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

# either layout dispatches by the .msh suffix
mesh = UgridDataset.read_file("domain.msh")

faces = mesh.mesh.face_node_connectivity   # triangles and quadrilaterals
```

## Documented limitations

- **Binary `.msh` is refused.** The `$MeshFormat` header's `file-type` must be `0` (ASCII);
  a `1` (binary) file is declined rather than mis-read.
- **Only v2.2 and v4.1 are read.** The major version is taken from the header and must be
  `2` or `4`; a 1.x or 3.x file is refused by name.
- **Only surface elements become faces.** Triangles (type `2`) and quadrilaterals (type
  `3`) are kept; a mixed file pads its triangles to the quad width with `-1`.
- **Lines and points are dropped.** Type `1` line and type `15` point elements are the
  boundary scaffolding Gmsh writes beside the surface; folding them into the face table
  would mislabel one-dimensional entities as faces.
- **An element of an unaccounted type refuses the file.** Its node count is unknown, so the
  record cannot be sized and everything after it would misalign.
- **A file with no triangular or quadrilateral element is refused** — there is no surface
  to build, and an empty mesh would be a silent wrong answer.
- **Node tags may be gapped or out of order** — resolved through a tag→index map, so a
  renumbered file stays wired correctly; the 1-based Gmsh base is recorded for a writer.
  Clockwise faces are reversed to counter-clockwise.
- **No CRS.** Gmsh carries engineering coordinates, so `crs` is `None` rather than guessed.
  Physical-name and entity blocks are skipped. Node elevation becomes a `node_z` variable
  only when some node has a non-zero z.

## API

::: pyramids_mesh.drivers.gmsh.GmshDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
