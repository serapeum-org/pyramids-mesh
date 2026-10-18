# Kratos — `.mdpa` model part

The Kratos Multiphysics (`.mdpa`) reader. The model-part file is a sequence of `Begin/End`
blocks: `Begin Nodes` lists `id x y z`, and `Begin Elements <ElementType>` lists
`elem_id prop_id n1..nN` (the node count comes from the element-type suffix, e.g.
`Element3D4N` = 4 nodes); `//` begins a comment.

| | |
|---|---|
| Driver name | `kratos` |
| Suffixes | `.mdpa` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.mdpa")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Element handling

A 2-D surface element type (`Element2D3N`/`Element2D4N`) is preferred as a face when
present. Otherwise, following the Exodus reader's precedent, each 3-D element
(`Element3D4N` tet, `Element3D8N` hex, …) maps its node list to a fill-padded face row
rather than being boundary-extracted (derivation is pyramids' concern — ADR-0005). The
leading `elem_id` and `prop_id` columns are dropped; `Conditions`, `Properties` and
`SubModelPart` blocks are read past.

## Documented limitations

- Node ids are 1-based and mapped to contiguous indices; the base is recorded for a writer.
- There is no CRS in the format — engineering coordinates, no datum — so `crs` is `None`.
- A solid model part is mapped cell-by-cell to faces; it is not a true 2-D surface.

## API

::: pyramids_mesh.drivers.kratos.KratosDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
