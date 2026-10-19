# Deltares 1D network — Delft3D FM / D-HYDRO UGRID NetCDF

The Deltares 1D reader, building a `Network1d` from the **`mesh1d` discretisation** of a
Delft3D FM / D-HYDRO UGRID-1.0 + Deltares-extension file: the computational nodes along
the branches and their edge connectivity. Read through the vendored GDAL multidimensional
API (no declared dependency).

| | |
|---|---|
| Driver name | `deltares1d` |
| Suffixes | `.nc` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `Network1d` |

A Deltares 1D file carries two one-dimensional topologies — a `network` of branches and a
`mesh1d` discretisation — plus `mesh1d_*` results over `time` on a map file. Both are
found by their `cf_role` attributes (`mesh_topology`, `edge_node_connectivity`), never by
a hard-coded name, so the two spellings in the wild (`network`/`mesh1d` and
`network1d`/`mesh1d`) read identically. The file is not plain 2D UGRID, so it is reached
through `read_file`'s NetCDF-suffix fallback (`pyramids-core` 0.69.0).

```python
from pyramids.netcdf.ugrid import Network1d

network = Network1d.read_file("FlowFM_map.nc")   # mesh1d discretisation
print(network.n_node, network.n_edge)
```

## Documented limitations

- **Only the `mesh1d` discretisation is built.** The coarser `network` of branch
  endpoints and its branch geometry are not returned; results live on the discretisation,
  which is what a `Network1d` represents. The discretisation is told from the bare network
  by its `coordinate_space` attribute.
- **A `Network1d` carries a topology and a CRS only.** The per-node branch index and
  offset and the `mesh1d_*` result variables are parsed and validated (node- or
  edge-located, over `time`), but `Network1d` has nowhere to attach them, so the built
  network warns that they are not in the result. Carrying them is upstream work in
  `pyramids-core` (design §10).
- **Edge indices honour the file's `start_index`** — a 1-based table is re-based to 0 and
  the base recorded for a writer.
- **A combined 1D2D file is declined by `can_open`.** A file that also carries a 2D
  `face_node_connectivity` is left to the UGRID reader (so plain 2D UGRID files are not
  stolen); its 1D half can still be read by calling `read` directly.
- Values are lazy; the loader reopens the file per read. The CRS comes from the file's
  EPSG code when it declares one, else `None`.

## API

::: pyramids_mesh.drivers.deltares.DeltaresNetworkDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
