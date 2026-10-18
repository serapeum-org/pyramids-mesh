# OpenFOAM polyMesh — directory of text components

The OpenFOAM reader. An OpenFOAM mesh is a `polyMesh` directory of plain-text,
extensionless component files (`points`, `faces`, `owner`, `neighbour`, `boundary`), not a
single file. Read through plain text parsing; no declared dependency.

| | |
|---|---|
| Driver name | `openfoam` |
| Suffixes | `.foam` (the ParaView case marker; a `polyMesh` directory is also accepted) |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

A directory has no suffix, and the component files are extensionless, so `read_file` cannot
dispatch the mesh by name. Pass the `polyMesh` directory with an explicit driver:

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("case/constant/polyMesh", driver="openfoam")

elevation = mesh.get("node_z")
```

`can_open` additionally accepts a path to the `points` component inside the directory, so a
dispatcher that probes candidate files still recognises the mesh.

## Documented limitations

- **The input is a directory.** `read_file` cannot dispatch a bare directory, and the
  component files are extensionless, so there is no suffix to claim. Reach the driver with
  `driver="openfoam"` on the `polyMesh` directory; `can_open` also accepts its `points`
  component.
- **Only the boundary surface is read.** OpenFOAM stores a 3D volume mesh; the
  `nInternalFaces` interior faces (and `owner`/`neighbour`) describe the volume and are
  dropped (ADR-0005). The mesh is the union of the faces the `boundary` patches enumerate
  through their `[startFace, startFace + nFaces)` ranges, with the referenced points
  compacted to a dense node list.
- **ASCII only.** A component whose `FoamFile` header declares `format binary` is refused
  rather than mis-parsed.
- **Point indices are already 0-based**, as OpenFOAM writes them; the base is recorded `0`
  for a writer. Faces may be triangles, quads or larger polygons, and the connectivity is
  fill-padded to the widest face. Faces not wound counter-clockwise in the XY projection are
  reversed; faces lying edge-on to that projection have no XY winding and are left unchanged.
- **No georeference.** Coordinates are CFD local coordinates, so `crs` is `None`.
- The third coordinate becomes a `node_z` variable.

## API

::: pyramids_mesh.drivers.openfoam.OpenFoamDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
