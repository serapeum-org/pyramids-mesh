# pyramids-mesh

Reads and writes the file formats of hydrodynamic and hydraulic mesh models — 2DM, Selafin, DFSU,
XMDF, PLY and more — as [pyramids](https://github.com/serapeum-org/pyramids) UGRID objects.
`pyramids-core` already ships the mesh data model and every operation that follows from a topology
in hand; what it cannot do is open other people's mesh files. That is this package's only job.

## The layer boundary

The split is the point of the package, so it is worth stating precisely. `pyramids-core` owns the
mesh *data model* and everything that follows from a topology already in hand:

| Concern | Lives in |
|---|---|
| `Mesh2d`, `Network1d`, `UgridDataset`, `Connectivity` | `pyramids.netcdf.ugrid` |
| UGRID-compliant NetCDF read / write | `pyramids.netcdf.ugrid.io` |
| Connectivity derivation, graph algorithms, boundary extraction | `pyramids.netcdf.ugrid.graph` |
| Sampling, `rasterize_like`, mesh-to-mesh regridding, time interpolation | `pyramids.netcdf.ugrid.regrid` |
| Construction from a raster grid or a polygon layer | `pyramids.netcdf.ugrid.construct` |
| Mesh ↔ vector, gap filling, xarray / zarr hand-off | `pyramids.netcdf.ugrid.{vector_ops,fill,interop_xarray}` |
| Mesh rendering | `pyramids.netcdf.ugrid.plot` → cleopatra `MeshGlyph` |

**pyramids-mesh** adds the one thing that is missing from that list — **the ability to open other
people's mesh files**:

| Concern | Lives in |
|---|---|
| Reading 2DM, Selafin, DFSU, XMDF, XDMF, DAT, FLO-2D, 3Di, PLY, TIN, … | `pyramids_mesh.drivers` |
| Writing 2DM, Selafin, DAT, PLY, Mike21 | `pyramids_mesh.drivers` |
| Driver discovery via the `pyramids.mesh_drivers` entry points | `pyramids-core` reads, this package registers |

A survey of the mesh-format landscape found `pyramids-core` reads exactly one of the eighteen
driver families the established mesh I/O libraries cover (UGRID NetCDF).
The other seventeen are what this package is for. In the other direction those libraries do no
interpolation, regridding, reprojection or graph analysis, all of which `pyramids-core` already has.

Anything that belongs on the left-hand side of the first table is a change to `pyramids-core`, not a
feature of this package. Report it upstream rather than duplicating it here.

## Status

`0.1.0` is the repository scaffold — packaging, CI, docs, and the import contract that pins
`import pyramids` first so the vendored osgeo bindings are active before anything else loads. The
public surface is empty; `pyramids_mesh.__version__` is all it exports.

```python
>>> import pyramids_mesh
>>> pyramids_mesh.__version__
'0.1.0'
```

## Where to next

- [Installation](installation.md) — Pixi environments and what each one carries.
- [Reference](reference/index.md) — the API surface, as it lands.
- [Change log](change-log.md)
- [Architecture Decision Records](adr/index.md) — including the scope boundary above, as
  [ADR-0003](adr/0003-scope-boundary.md).

Source: <https://github.com/serapeum-org/pyramids-mesh>
