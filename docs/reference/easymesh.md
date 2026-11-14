# SWAN Easymesh — `.n` + `.e` pair

The SWAN Easymesh reader. SWAN defines no native unstructured grid: `READGRID UNSTRUCTURED`
delegates to ADCIRC (`fort.14`), Triangle (`.node`/`.ele`) or Easymesh (`.n`/`.e`). The
first two are covered by the [ADCIRC](adcirc.md) and [Triangle / TetGen](triangle.md)
readers; this driver adds the third, so all three SWAN unstructured routes are read. Pure
ASCII, parsed with the standard library and numpy (no declared dependency).

| | |
|---|---|
| Driver name | `easymesh` |
| Suffixes | `.n`, `.e` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `MeshDataset` |

`.n` and `.e` are a pair sharing a stem. Point `read_file` at either file; the sibling is
read from beside it:

```python
from pyramids.netcdf.ugrid import MeshDataset

# point at either half of the pair
mesh = MeshDataset.read_file("path/to/mesh.n")
mesh = MeshDataset.read_file("path/to/mesh.e")
```

The formats:

- `.n` — a node count on the first line, then `id:  x y marker` per node. Ids are 0-based
  and colon-suffixed; coordinates are written in scientific notation.
- `.e` — an element count on the first line, then
  `id:  i j k  ei ej ek  si sj sk  xV yV  sign` per triangle. Only the first three fields,
  the triangle's node ids `i j k`, are read.

Easymesh also writes a third `.s` sides file and closes each file with two trailer lines
(a rule of dashes and a column legend); neither is needed and both are ignored.

## Documented limitations

- **Both files are required.** `read_file` is pointed at either the `.n` or the `.e` file
  and the other is read from beside it. A lone file with no sibling is declined.
- **Only `.n` and `.e` are read.** The `.s` sides file is redundant — the sides, element
  neighbours and circumcentres are topology pyramids derives from the nodes and faces — so
  it is neither required nor read.
- **Triangles only.** Easymesh produces Delaunay triangulations; each element's first three
  ids are read as a triangular face, normalised to counter-clockwise winding on the `xy`
  projection.
- **The index base is detected, not assumed** — Easymesh is 0-based, but the base is read
  from the first node id, resolved to 0-based and recorded for a writer.
- **Node markers, element neighbours, side ids, the circumcentre and the sign are ignored**,
  and no CRS is declared — the format carries none, so `crs` is `None` rather than guessed.

## API

::: pyramids_mesh.drivers.easymesh.EasymeshDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
