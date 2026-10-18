# Abaqus — input-deck mesh

The Abaqus `.inp` reader. A keyword deck lists nodes under `*NODE` and elements under
`*ELEMENT, TYPE=<code>`; lines beginning `**` are comments and data lines are
comma-separated. Node ids are arbitrary labels, resolved to contiguous indices through an
id map rather than by assuming `id == index + 1`.

| | |
|---|---|
| Driver name | `abaqus` |
| Suffixes | `.inp` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.inp", driver="abaqus")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Element handling

Surface element types (`S3`/`STRI3`/`CPS3`/`CPS4`/`S4`/`CAX3`/`CAX4`) map their node list to
a fill-padded face row. Following the Exodus reader's precedent, a volumetric element
(`C3D4`/`C3D8`) is mapped to a face row from its node list rather than boundary-extracted
(connectivity derivation is pyramids' concern, not a driver's — ADR-0005). Keywords are
matched case-insensitively.

## Documented limitations

- **Node ids are arbitrary labels**, mapped to contiguous 0-based indices with the 1-based
  base recorded for a writer.
- **Node elevation** — the `z` column becomes a `node_z` variable only when some value is
  non-zero, matching the Exodus and gmsh readers.
- There is no CRS in the format — the coordinates are engineering coordinates with no
  datum — so `crs` is `None`.
- **The `.inp` suffix is shared with AVS-UCD.** `can_open` requires the Abaqus signature
  (the first non-comment line begins with a `*` keyword), so an AVS-UCD `.inp` — whose
  first significant line is five integers — is declined rather than misread.

## API

::: pyramids_mesh.drivers.abaqus.AbaqusDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
