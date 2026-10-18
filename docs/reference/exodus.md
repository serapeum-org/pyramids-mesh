# Exodus II — finite-element mesh

The Exodus II reader. A `.exo`/`.e` file is NetCDF (CDF-2) underneath: node coordinates in
`coordx`/`coordy`/`coordz` and one or more element blocks `connect1`/`connect2`/… whose
rows are 1-based element-to-node lists carrying an `elem_type` attribute (`QUAD4`, `TRI3`,
`TETRA`, …), with the block ids in `eb_prop1`. The NetCDF substrate is read through the
vendored GDAL multidimensional API (no declared `netCDF4` dependency, no `exodus` library).

| | |
|---|---|
| Driver name | `exodus` |
| Suffixes | `.exo`, `.e` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.exo")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Documented limitations

- **Every element block's node list becomes a face row**, including a volume element type
  such as `TETRA`. A genuine volumetric (3D) mesh is not modelled yet (design §21.1), so a
  tetrahedron's four node ids are read verbatim as a face rather than expanded into its
  bounding triangles; winding is then judged in the `xy` plane like any other face.
- **Connectivity is 1-based with mixed widths padded by `0`** — a short row in a
  mixed-element table (triangles among quads) is padded with `0`, never a valid 1-based id,
  and the normaliser rewrites it to `-1` fill; the `1` base is recorded for a writer.
- **Results are not read** — only the mesh is built; the per-timestep nodal and element
  variables an Exodus file may carry are out of scope for this reader.
- `coordz` (a 3D file) contributes a `node_z` variable; a 2D file leaves it unset. There is
  no CRS in the format, so `crs` is `None`.

## API

::: pyramids_mesh.drivers.exodus.ExodusDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
