# STL — stereolithography

The STL reader. ASCII (`solid`/`facet`/`outer loop`/`vertex`/`endloop`/`endfacet`) and
binary (80-byte header, a `uint32` triangle count, then 50 bytes per triangle), hand-parsed
— a triangle-soup geometry format with elevation per vertex and no time axis.

> **No CRS, no georeferencing.** STL carries no coordinate reference system or spatial
> reference of any kind, so `crs` is always `None`. That makes it a low-value format for a
> GIS stack; it is provided for mesh interchange only.

| | |
|---|---|
| Driver name | `stl` |
| Suffixes | `.stl` |
| Capabilities | `read_mesh`, `read_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("solid.stl")
```

## Documented limitations

- **STL has no vertex indexing.** Every triangle stores its three corner coordinates
  inline, so the reader welds byte-identical vertices into a shared node list and rebuilds
  the triangle face connectivity against it.
- **ASCII and binary are both supported.** The two are told apart by the binary size
  identity (`80 + 4 + 50 × count`), not by the leading word — a binary STL's header very
  often begins with `solid` too, which naive detection mistakes for ASCII.
- **A round trip is not geometrically faithful.** `z` survives as a `node_z` variable, but
  pyramids' mesh model is planar, so a consumer reading the geometry sees the `(x, y)`
  projection — and a 3D solid's vertical faces collapse to zero-area (degenerate) faces in
  that projection (design §6).
- **STL carries no CRS and no georeferencing**, so `crs` is `None`.
- **The whole file is read eagerly**, so the driver does not declare `lazy_data`.
- The per-triangle normal and the 2-byte attribute word are read for width but not carried.
- Welded vertices are 0-based and stay so.

## API

::: pyramids_mesh.drivers.stl.StlDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
