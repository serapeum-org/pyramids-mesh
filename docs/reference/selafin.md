# Selafin / Serafin — TELEMAC

The TELEMAC Selafin reader, and the package's first binary driver. Fortran unformatted
records with an undeclared byte order and an optional double-precision variant. It proves
the lazy path: the topology and the one-float time records are read eagerly, and each
variable's per-timestep slab is left behind a loader that seeks straight to it.

| | |
|---|---|
| Driver name | `selafin` |
| Suffixes | `.slf`, `.srf`, `.ser`, `.sel` |
| Capabilities | `read_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

dataset = UgridDataset.read_file("model.slf")
depth = dataset.get("WATER DEPTH").data   # read now; deferred until this line
```

## How it is read

- **Byte order is detected, not declared.** The first record must announce an 80-byte
  title in one order or the other; a file that does neither is not Selafin.
- **Precision comes from the title marker.** `SERAFIND` is double precision, `SERAFIN`
  single, and the width applies to *every* float record — coordinates and data alike.
- **Variable names are 32 characters**, split by position into a 16-char name and a
  16-char unit. Splitting on whitespace instead would lose a multi-word name's tail.
- **`IKLE` connectivity is 1-based**; it arrives 0-based, with the base recorded for a
  writer to restore.

## Documented limitations

- **The coordinate origin is applied, and getting it wrong is undetectable.** `IPARAM[2]`
  and `IPARAM[3]` are the x and y origins the stored coordinates are offsets from; they
  are added to the node coordinates. Read without adding them, the mesh sits near zero
  instead of at its real location — plausible and completely wrong, with nothing in the
  data to catch it.
- **A 3D file ships its surface plane only.** A non-zero `IPARAM[6]` (plane count) means a
  layered 3D file; the surface is returned with a warning, because a full read needs the
  layered mesh model pyramids does not have yet (design §21.1).
- **A parallel sub-domain file is refused.** A non-zero `IPARAM[7]` or `IPARAM[8]` means
  local numbering within one partition, so the connectivity would index a sub-domain
  rather than the whole mesh. Merge the partitions before reading.
- **Variables are node-located, always** — never inferred from an array length.
- **The IKLE row/column order could not be checked against a vendor file.** It is read
  row-major (one element's `NDP` node ids contiguous), as MDAL reads it. A real file that
  stored it transposed would give a scrambled mesh; this is the detail a vendor sample
  should confirm, tracked for the `vendor_data` suite.
- **A partial final timestep is dropped with a warning** — a run killed mid-write leaves
  a dangling time record, and only whole steps are read.
- There is no CRS in the format, so `crs` is `None`; it is never inferred from the
  coordinate magnitudes.

## API

::: pyramids_mesh.drivers.telemac.SelafinDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
