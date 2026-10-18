# ESMF — unstructured-mesh NetCDF

The ESMF reader. ESMF (the Earth System Modeling Framework) stores its unstructured mesh as
NetCDF with an explicit node list and element connectivity: node coordinates in
`nodeCoords` (shape `nodeCount` × `coordDim`, column 0 longitude and column 1 latitude), the
per-element node table in `elementConn` (`elementCount` × `maxNodePElement`), and
`numElementConn` giving each element's real node count. The driver builds the surface mesh —
the node list becomes the nodes and `elementConn` the fill-padded, mixed-degree face
connectivity. Read through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `esmf` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

An ESMF mesh is not UGRID, so `read_file` reaches the driver through its NetCDF-suffix
fallback:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("ne4np4-esmf.nc")
mesh.mesh.n_node          # the ESMF node list
mesh.mesh.n_face          # the ESMF elements
```

## Documented limitations

- **The real node count comes from `numElementConn`, not from a fill value.** Each element
  keeps its first `numElementConn` nodes; the rest of the row is padding and becomes `-1`
  fill, so a mesh of triangles and quads yields a fill-padded, mixed-degree table without
  depending on the file's own pad value.
- **`elementConn` is 1-based by default.** A `start_index` attribute overrides the base;
  the connectivity is resolved to 0-based and the base recorded for a writer.
- **Coordinates are carried in the file's unit, degrees by default.** A `units` attribute on
  `nodeCoords` naming radians is honoured and the coordinates converted to degrees;
  otherwise they are left as the file gave them.
- **No CRS.** The mesh is lon/lat on a sphere with no EPSG authority, so `crs` is `None`
  rather than guessed.
- **Only the topology is read.** The element areas and other ESMF fields are not shipped as
  variables; this driver declares `read_mesh` only.

## API

::: pyramids_mesh.drivers.esmf.EsmfDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
