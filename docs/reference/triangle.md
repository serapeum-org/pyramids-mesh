# Triangle / TetGen — `.node` + `.ele` pair

The Triangle / TetGen reader. Triangle (2-D) and TetGen (3-D) share one ASCII mesh split
across a `.node` vertex file and an `.ele` element file. Pure ASCII, parsed with the
standard library and numpy (no declared dependency).

| | |
|---|---|
| Driver name | `triangle` |
| Suffixes | `.node`, `.ele` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

`.node` and `.ele` are a pair sharing a stem. Point `read_file` at either file; the sibling
is read from beside it:

```python
from pyramids.netcdf.ugrid import UgridDataset

# point at either half of the pair
mesh = UgridDataset.read_file("path/to/mesh.node")
mesh = UgridDataset.read_file("path/to/mesh.ele")

elevation = mesh.get("node_z")  # present only for a 3-D (TetGen) mesh
```

The formats:

- `.node` header `<#verts> <dim> <#attrs> <#bndmarkers>`, then `id x y [z] [attrs...]
  [marker]` per vertex.
- `.ele` header `<#elts> <nodes-per-elt> <#attrs>`, then `id n1 n2 n3[ n4]` per element.

## Documented limitations

- **Both files are required.** `read_file` is pointed at either the `.node` or the `.ele`
  file and the other is read from beside it. A lone file with no sibling is declined.
- **The index base is detected, not assumed** — it is whatever the first vertex id is
  (`0` or `1`), resolved to 0-based and the base recorded for a writer. Vertices are read
  in file order, as both tools write them contiguously from the base.
- **TetGen tetrahedra are read as 4-node faces.** The element node list is taken verbatim
  as the face connectivity; no 3-D cell topology is derived here. Faces are normalised to
  counter-clockwise winding on the `xy` projection.
- **Per-vertex attributes and boundary markers are ignored**, and no CRS is declared — the
  format carries none, so `crs` is `None` rather than guessed. For a 3-D mesh the third
  coordinate becomes a `node_z` variable.

## API

::: pyramids_mesh.drivers.triangle.TriangleDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
