# Nastran — bulk-data mesh

The Nastran bulk-data reader. A `.bdf`/`.nas`/`.fem` deck is a plain-text finite-element
model: `GRID` cards give node coordinates and `CTRIA3`/`CQUAD4` cards give the surface
element-to-node lists. Node ids are arbitrary labels, resolved to contiguous indices
through an id map rather than by assuming `id == index + 1`, so a gapped or renumbered
deck wires correctly.

| | |
|---|---|
| Driver name | `nastran` |
| Suffixes | `.bdf`, `.nas`, `.fem` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("model.bdf")
mesh.mesh.n_node, mesh.mesh.n_face
```

## Field layouts

Three card layouts share one deck and each line declares its own:

- **small** fixed-field — ten 8-character columns;
- **large** fixed-field — a `*` on the card name, 16-character columns spilling onto a
  `*` continuation line;
- **free** field — comma-separated.

A fixed-field real may embed its exponent sign without an `E` — `-2.45-16` means
`-2.45e-16`, and `1.5D+3` (Fortran double) means `1500.0`. All three spellings are parsed.

## Documented limitations

- **Volumetric and line elements are skipped** — `CTETRA`/`CPENTA`/`CHEXA`/`CPYRA` solids
  and `CBAR`/`CBEAM` bars are dropped; only `CTRIA3` and `CQUAD4` surface faces are read. A
  genuine 3D mesh is out of scope (design §21.1), so a deck of only solids is refused for
  carrying no surface. `CTRIA3` rows are fill-padded to the four columns of `CQUAD4`.
- **Node ids are arbitrary labels**, mapped to contiguous 0-based indices with the 1-based
  base recorded for a writer. An element naming an id no `GRID` defines is refused, and a
  duplicate `GRID` id is ambiguous and refused.
- **Node elevation** — the `GRID` `X3` column becomes a `node_z` variable when any value is
  non-zero, matching the Exodus and gmsh readers; an all-zero column is dropped rather than
  kept as a meaningless `node_z`.
- There is no CRS in the format — the coordinates are engineering coordinates with no
  datum — so `crs` is `None`.
- **The `.fem` suffix is shared with DHI FEFLOW.** `can_open` requires the Nastran
  signature (a `GRID` and a `CTRIA3`/`CQUAD4` card, and no FEFLOW `PROBLEM:`/`CLASS`
  header), so a FEFLOW `.fem` is declined rather than misread.

## API

::: pyramids_mesh.drivers.nastran.NastranDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
