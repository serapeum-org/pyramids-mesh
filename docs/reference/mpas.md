# MPAS — spherical Voronoi NetCDF grid

The MPAS reader. MPAS (Model for Prediction Across Scales) writes its unstructured grid as
NetCDF with `Conventions = "MPAS"`. The grid is a spherical centroidal Voronoi
tessellation: a hex-dominant **primal** mesh of cells and a **dual** mesh whose vertices
are the Voronoi corners. This driver reads the dual — the corners become the mesh nodes
and each primal cell becomes a face listing the corners around it. The file is read
through the vendored GDAL multidimensional API (no declared dependency).

| | |
|---|---|
| Driver name | `mpas` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

An MPAS grid is not UGRID, so `read_file` reaches the driver through its NetCDF-suffix
fallback:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("mesh.QU.1920km.nc")
mesh.mesh.n_node          # the Voronoi dual vertices
mesh.mesh.n_face          # the primal (hex-dominant) cells
```

The corner table per cell is `verticesOnCell`, a `(nCells, maxEdges)` array that is
1-based and padded with `0`, with `nEdgesOnCell` giving each cell's valid count.
Subtracting one and dropping the padding yields fill-padded, mixed-degree face-node
connectivity: a coarse quasi-uniform mesh is 12 pentagons among hexagons, so the face
table carries both 5- and 6-node rows.

## Documented limitations

- **Longitude/latitude are exposed, in degrees, not the Cartesian `xVertex`/`yVertex`.**
  MPAS stores both: lon/lat (radians) and the 3D Cartesian coordinates on the sphere.
  Dropping `z` from the Cartesian triple collapses the two hemispheres onto one another
  and degenerates the cells at the limb, so the geographic coordinates are used instead
  and converted from radians to the conventional degrees. A cell straddling the
  antimeridian unwraps to a wide planar polygon — inherent to flattening a global mesh —
  but its winding is still well defined.
- **No CRS.** The grid lives on a unit sphere (`sphere_radius = 1`) with no EPSG
  authority, so `crs` is `None` rather than guessed at `4326` (a WGS84 ellipsoid the mesh
  is not on).
- **The primal (hex) cells are the faces; the dual vertices are the nodes.** Only the
  topology needed for that is read — coordinates and `verticesOnCell`. The many other MPAS
  fields (`cellsOnCell`, edge tables, areas, quality metrics) are not shipped as
  variables; this driver declares `read_mesh` only.
- **`verticesOnCell` is 1-based with `0` padding** — resolved to 0-based, the base `1`
  recorded for a writer.

## API

::: pyramids_mesh.drivers.mpas.MpasDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
