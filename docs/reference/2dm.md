# 2DM — SMS / TUFLOW mesh frame

The 2DM reader. Plain text, one record per line, and the reference implementation every
other driver in this package follows.

| | |
|---|---|
| Driver name | `2dm` |
| Suffixes | `.2dm` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset`, or `Network1d` for a file of only `E2L` elements |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.2dm")
```

## What it reads

| Record | Meaning |
|---|---|
| `MESH2D` | the header; the file's first significant line must be this |
| `ND <id> <x> <y> <z>` | a node. `z` becomes a `node_z` variable, not a coordinate |
| `E3T <id> <n1..n3> <mat>` | a triangular face |
| `E4Q <id> <n1..n4> <mat>` | a quadrilateral face |
| `E6T <id> <n1..n6> <mat>` | a six-node face, read as such |
| `E2L <id> <n1> <n2> <mat>` | a 1D line element |
| `NUM_MATERIALS_PER_ELEM <k>` | switches the parser to `k` material columns |

Node and element ids are resolved through an id map rather than by assuming
`id == index + 1`: **2DM ids may have gaps and may be out of file order**, and a tool
that renumbers a mesh produces exactly that. The node count is the number of `ND`
records, never the highest id.

The material id becomes a face-located `material` variable. With
`NUM_MATERIALS_PER_ELEM k` the extra columns become `material_2` … `material_k`.

## Limitations

Every one of these is deliberate; none is a silent approximation.

- **`E3L`, `E8Q` and `E9Q` are refused**, and the whole file with them, naming the
  element. MDAL does the same (`Err_UnsupportedElement`). Treating a mid-side or centre
  node as a corner would distort the element without any error, so no mapping is
  guessed.
- **`E6T` is kept as a six-node face.** Its node order is corner, mid-side, corner,
  mid-side, corner, mid-side, so the hexagon through those points *is* the element
  outline; it is not demoted to its corner triangle.
- **A file holding both faces and `E2L` elements returns the faces only**, with a
  warning. A mixed-dimension file is reached through
  [`Network1d`][pyramids.netcdf.ugrid.Network1d]; `UgridDataset` and `Network1d` are
  separate types and one read returns one of them.
- **No elevation coordinate.** The pyramids mesh model is planar, so `z` survives as a
  node-located `node_z` variable. Anything consuming the mesh *geometry* sees the planar
  projection.
- **No CRS.** 2DM carries none, so `crs` is always `None`; it is never inferred from the
  coordinate magnitudes.
- **Read-only for now.** `write_mesh` arrives with the 2DM writer (task W1).
- Values are read eagerly: the format is text, so there is nothing to defer and
  `lazy_data` is not declared.
