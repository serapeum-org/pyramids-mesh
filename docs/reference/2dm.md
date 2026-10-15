# 2DM — SMS / TUFLOW mesh frame

The 2DM reader. Plain text, one record per line, and the reference implementation every
other driver in this package follows.

| | |
|---|---|
| Driver name | `2dm` |
| Suffixes | `.2dm` |
| Capabilities | `read_mesh`, `read_data`, `multi_mesh` |
| Extra required | none |
| Returns | `UgridDataset` for the surface mesh, `Network1d` for the `E2L` network |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.2dm")
```

## Files holding both a surface and a network

A 2DM may carry a surface mesh (`E3T`/`E4Q`/`E6T`), a 1D network (`E2L`), or both. The two are
named separately, so `mesh_names` tells you what a file holds and `mesh_name=` picks one:

```python
from pyramids.netcdf.ugrid import Network1d, UgridDataset

UgridDataset.mesh_names("domain.2dm")            # ['mesh2d', 'network1d']
mesh = UgridDataset.read_file("domain.2dm")      # the surface mesh
channels = Network1d.read_file("domain.2dm", mesh_name="network1d")   # the 1D network
```

Reading the surface of a file that also holds `E2L` elements warns that the network is not in
that result and names this route. The `E2L` elements are a separate 1D topology, not the surface
mesh's edges, so they are never folded into `edge_node_connectivity`.

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
- **One read returns one topology.** `UgridDataset` and `Network1d` are separate types, so a
  file holding both is read twice, once per `mesh_name` (see above). Reading either half warns
  about what the other half leaves behind rather than omitting it silently.
- **No elevation coordinate.** The pyramids mesh model is planar, so on the **surface** path `z`
  survives as a node-located `node_z` variable. Anything consuming the mesh *geometry* sees the
  planar projection.
- **A 1D read cannot carry `z`, variables or global attributes at all.**
  [`Network1d`][pyramids.netcdf.ugrid.Network1d] holds a topology and a CRS and has no parameter
  for data, so reading the network half of a file reports each of those by name in a warning and
  returns without them. This is a `pyramids-gis` limitation, not a decision of this driver: it
  will be carried properly once `Network1d` has somewhere to put it.
- **No elevation is invented.** If no `ND` record carries a z column, `node_z` is omitted
  entirely rather than filled with zeros a consumer could not tell from a real zero datum.
- **No CRS.** 2DM carries none, so `crs` is always `None`; it is never inferred from the
  coordinate magnitudes.
- **Read-only for now.** `write_mesh` arrives with the 2DM writer (task W1).
- Values are read eagerly: the format is text, so there is nothing to defer and
  `lazy_data` is not declared.
