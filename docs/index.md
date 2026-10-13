# pyramids-mesh

The unstructured-mesh tier of the [pyramids](https://github.com/serapeum-org/pyramids) GIS stack.
It extends the UGRID support `pyramids-gis` already ships (`pyramids.netcdf.ugrid`) with the
operations that tier deliberately leaves out: **generation**, **quality**, and **refinement**.

## The layer boundary

The split is the point of the package, so it is worth stating precisely. `pyramids-gis` owns the
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

**pyramids-mesh** adds the three things that are missing from that list:

- **Generation** — building a mesh where no topology exists yet, rather than flattening a grid or a
  polygon layer that already carries one.
- **Quality** — per-element metrics (aspect ratio, skewness, orthogonality) and quality-driven
  smoothing.
- **Refinement** — local refinement and coarsening of an existing mesh.

Anything that belongs on the left-hand side of that table is a change to `pyramids-gis`, not a
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
