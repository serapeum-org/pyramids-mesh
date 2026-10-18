# Medit — INRIA `.mesh`

The Medit reader. A plain-text keyword-block format: `MeshVersionFormatted`, a
`Dimension` of `2` or `3`, a `Vertices` block and the element sections. It is parsed with
the standard library alone, so the driver declares no dependency and imports no mesh
library.

| | |
|---|---|
| Driver name | `medit` |
| Suffixes | `.mesh` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.mesh")
```

## Layout

```text
MeshVersionFormatted 1          signature (first non-blank line)
Dimension 3                     2 or 3; value may fall on the next line
Vertices
39
<x> <y> <z> <ref>               one per vertex (x y ref for Dimension 2)
Triangles
72
<n1> <n2> <n3> <ref>            one per triangle, node ids 1-based
Tetrahedra
86
<n1> <n2> <n3> <n4> <ref>       dropped (volume cell)
End                             optional
```

Each section is a keyword, a count and then that many rows. Node ids are 1-based and
contiguous; the connectivity arrives 0-based with the base recorded for a writer to
restore. The trailing reference tag on every row is a Medit label and is dropped.

## Shared `.mesh` suffix

The `.mesh` suffix is also used by the DHI Mike21 frame format, so `can_open` requires the
`MeshVersionFormatted` signature on the first non-blank line. A Mike21 `.mesh`, which opens
with a numeric EUM header, is declined — and Mike21's own sniff declines a Medit file for
the same reason, so the two drivers never both claim one file.

## Which sections become a mesh

The reader turns the 2D surface sections into faces and drops the rest:

| Section | Row | Mapped to |
|---|---|---|
| `Vertices` | `x y [z] ref` | node coordinates (`node_z` for a 3D file) |
| `Triangles` | `n1 n2 n3 ref` | 3-node face |
| `Quadrilaterals` | `n1 n2 n3 n4 ref` | 4-node face |
| `Edges` | `n1 n2 ref` | read past, not kept (a 1D boundary marker) |
| `Tetrahedra` | `n1..n4 ref` | dropped (volume cell) |
| `Hexahedra` | `n1..n8 ref` | dropped (volume cell) |

## Documented limitations

- **Only the 2D surface sections become a mesh** — `Triangles` and `Quadrilaterals`.
  `Tetrahedra` and `Hexahedra` carry a vertical structure this package does not model yet;
  they are dropped, not boundary-extracted (that is pyramids' job). A file with no surface
  section is refused rather than returned as an empty mesh.
- **`Edges` are read past but not kept.** A Medit edge is a 1D boundary marker, not an edge
  of the surface mesh.
- **A section keyword this reader does not know is refused**, rather than skipped by
  guessing its row arity. `MeshVersionFormatted`, `Dimension`, `Vertices`, `Edges`,
  `Triangles`, `Quadrilaterals`, `Tetrahedra`, `Hexahedra` and `End` are the keywords read.
- **Only the ASCII format is read.** The binary `.meshb` variant is not handled and is not
  claimed by the suffix.
- **Node elevation becomes a `node_z` variable** for a 3D file, as it does for 2DM and
  Mike21: the pyramids mesh model is planar and carries no per-node elevation coordinate
  (design §6). A 2D file gets no `node_z`.
- There is no CRS in the format, so `crs` is `None`, never guessed from the coordinate
  magnitudes.

## API

::: pyramids_mesh.drivers.medit.MeditDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
