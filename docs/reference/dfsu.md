# DFSU — DHI MIKE unstructured mesh

The DHI MIKE DFSU reader. DFSU is a proprietary binary container read through DHI's own
`mikeio` library, isolated in the `dhi` extra.

| | |
|---|---|
| Driver name | `dfsu` |
| Suffixes | `.dfsu` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | `dhi` (`pip install pyramids-mesh[dhi]`) |
| Returns | `UgridDataset` |

Without the `dhi` extra the driver is not registered at all — nothing here is reachable.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("results.dfsu")
level = mesh.get("Surface elevation")   # face-located, over time
```

## Documented limitations

- **Requires the `dhi` extra.** Without `mikeio` installed the driver is skipped at
  discovery with a warning; it appears only when `[dhi]` is installed.
- **Elements are cell-centred, so every item is face-located.** Mixed triangles and quads
  are padded to the widest row with `-1`; the element table is already 0-based.
- **The projection is mapped to a CRS only when it is a real one.** DHI writes an internal
  projection name such as `UTM-33` or `LONG/LAT`, which carries no datum this package can
  resolve, so `crs` is left `None` rather than guessed.
- Values are lazy; the loader reopens the file and reads one item. Matplotlib's backend is
  pinned to `Agg` before `mikeio` is imported, so importing the driver never selects a GUI
  backend in a headless run.

## API

::: pyramids_mesh.drivers.dhi.DfsuDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
