# 2DM — SMS / TUFLOW mesh frame

The 2DM reader. Plain text, one record per line, and the reference implementation every
other driver in this package follows.

| | |
|---|---|
| Driver name | `2dm` |
| Suffixes | `.2dm` |
| Capabilities | `read_mesh`, `read_data`, `multi_mesh`, `write_mesh`, `write_data` |
| | `read_data` and `multi_mesh` are **declarative** — they describe the driver in `info(drivers=True)` and require no method. There is no `read_data()` to call: `read_data` here means the file carries values (material, elevation) that `read` returns alongside the topology. A results-only format declares `needs_mesh` instead and *does* implement the method. |
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
`NUM_MATERIALS_PER_ELEM k` the extra columns become `material_2` … `material_k`, for `k` from 1
to 16 — each column costs an array and a variable, and the count comes from an untrusted file, so
a larger one is refused rather than allocated.

## Limitations

Every one of these is deliberate; none is a silent approximation.

- **`E3L`, `E8Q` and `E9Q` are refused**, and the whole file with them, naming the
  element. Treating a mid-side or centre
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
  returns without them. This is a `pyramids-core` limitation, not a decision of this driver: it
  will be carried properly once `Network1d` has somewhere to put it.
- **No elevation is invented.** If no `ND` record carries a z column, `node_z` is omitted
  entirely rather than filled with zeros a consumer could not tell from a real zero datum. A file
  whose `ND` records *disagree* about the z column is refused, since the ones without it would
  need a datum invented.
- **Records this driver does not model are skipped**: `MESHNAME`, the
  `BEGPARAMDEF` block and anything else beyond the nodes, elements and
  `NUM_MATERIALS_PER_ELEM`. One consequence is that a file naming its own mesh loses that name —
  the topology is always built as `mesh2d`.
- **A malformed record is refused, never guessed.** A record short of the fields its keyword
  requires, a non-numeric coordinate, id or material, and a `NUM_MATERIALS_PER_ELEM` below one all
  raise and name the file and the record. An element short of its *declared* material columns is
  the one exception: the missing columns read as `0.0`, because a writer omitting a trailing
  column is common and refusing the file would cost more than it buys.
- **Two nodes, faces or line elements sharing an id** are refused: the connectivity would be
  ambiguous.
- **No CRS.** 2DM carries none, so `crs` is always `None`; it is never inferred from the
  coordinate magnitudes.
- **Read-only for now.** `write_mesh` arrives with the 2DM writer (task W1).
- Values are read eagerly: the format is text, so there is nothing to defer and
  `lazy_data` is not declared.

## Writing

`UgridDataset.to_file("out.2dm", driver="2dm")` writes the mesh back. It emits `MESH2D`,
one `ND` per node, and one `E3T`/`E4Q`/`E6T` per face — the element keyword chosen from
the face's **real** vertex count, ignoring `-1` fill.

- **The recorded connectivity base is re-applied.** A file read 1-based is written back
  1-based (`original_start_index` on the connectivity), so a read/write round trip
  preserves the numbering.
- **Node elevation** comes from a node-located `node_z` variable when present, else `0.0`;
  the **material** column from a face-located `material` variable, else `0`.
- **Full precision by default.** Coordinates are written with Python's shortest exact
  `float64` representation, because a fixed number of decimals silently loses position for
  projected coordinates in the millions. Pass `precision=<n>` for `n` significant digits
  when a smaller file matters more than an exact round trip.
- A face whose real vertex count is not 3, 4 or 6 has no 2DM element keyword and is
  refused; a dataset with no face connectivity is not a 2DM mesh and is refused.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.2dm")
mesh.to_file("copy.2dm", driver="2dm")              # exact round trip
mesh.to_file("small.2dm", driver="2dm", precision=8)  # 8 significant digits
```
