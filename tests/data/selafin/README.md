# Selafin sample files — sources and licences

Four real TELEMAC Selafin files, committed so the driver's tests run offline like every other fixture in
`tests/data/`. They are not synthetic: `write_selafin` wrote every other Selafin fixture in this repository, so
the reader and the writer share one set of assumptions about the record layout and cannot contradict each other.
These files were written by other software, which is the only way to catch a layout understood wrongly in the
same way at both ends — the `IKLE` row-major order was settled this way.

`pyramids-mesh` is GPL-3.0, so redistributing the GPL-3.0 files below is licence-compatible, and the
BSD-3-Clause file is permissive. The wheel does not carry them (`[tool.setuptools.packages.find]` picks up
`src/` only).

| File | Bytes | sha256 | Source | Licence |
|---|---|---|---|---|
| `r2d_tidal_flats.slf` | 241,780 | `d915fef9dcadd785…` | [oceanmodeling/xarray-selafin](https://github.com/oceanmodeling/xarray-selafin) `tests/data/` | GPL-3.0 |
| `r3d_bump.slf` | 774,836 | `a1888c99c4cdab68…` | [oceanmodeling/xarray-selafin](https://github.com/oceanmodeling/xarray-selafin) `tests/data/` | GPL-3.0 |
| `r3d_bump_extracted_bottom_layer.slf` | 136,580 | `e28102245f84bdf4…` | [oceanmodeling/xarray-selafin](https://github.com/oceanmodeling/xarray-selafin) `tests/data/` | GPL-3.0 |
| `qgismesh.slf` | 505,376 | `7aa2edb127a89227…` | [hydro-informatics/telemac](https://github.com/hydro-informatics/telemac) `bk-slf/` | BSD-3-Clause |

The checksums are the first 8 bytes of the sha256 of each file as committed. The upstream copies are pinned by
commit in `docs/examples/real-data/selafin.ipynb` — `168855a4` for xarray-selafin and `d2cbc445` for
hydro-informatics/telemac — so the full digests can be re-derived from those exact URLs:

```bash
sha256sum tests/data/selafin/*.slf
```

```
d915fef9dcadd7850771872cd8767ed5c9e7755d2e9fe92e6ad2691a47af4330  r2d_tidal_flats.slf
a1888c99c4cdab683fc6ecc4168dfdfe8ee34889ef8ce3dfef3251cf2053509a  r3d_bump.slf
e28102245f84bdf4c63c5cf4132bc3614e529b1a4a6ca2329a6238b122bd9edc  r3d_bump_extracted_bottom_layer.slf
7aa2edb127a89227212a6c4db304c46743886207d9d9747e1a81524e0b61856b  qgismesh.slf
```

## What each file covers

- **`r2d_tidal_flats.slf`** — TELEMAC 2D output for the "Sloped flume Rouse profile test" validation case: 648
  nodes, 1030 triangles, 17 time steps, 5 node variables, big-endian, single precision, a date record and 264
  boundary nodes. The file the `IKLE` order, the header metadata and the round trip are checked against.
- **`qgismesh.slf`** — geometry written by **BlueKenue/QGIS** rather than by TELEMAC: 12,814 nodes, 25,007
  triangles, one `BOTTOM` field, projected UTM coordinates. A second, independent writer, and the file that shows
  `IPOBO` must be read from its own record — it carries 619 boundary nodes while leaving `IPARAM[7]` at zero.
- **`r3d_bump.slf`** — a genuine layered 3D file: `NDP=6` prisms, 10,480 elements over 5 planes. The refusal path.
- **`r3d_bump_extracted_bottom_layer.slf`** — one layer extracted from that same run by TELEMAC itself: `NDP=3`
  triangles that inherit the parent's `IPARAM[6] = 5`. The file that forced the 3D guard onto `NDP`, since the
  plane count refuses a mesh that reads perfectly well. Keeping both halves of one run is deliberate: the pair is
  what shows that the plane count cannot be the test.

The 3D file is the smaller `r3d_bump.slf` rather than the 5.0 MB `r3d_tidal_flats.slf` from the same source. Both
are `NDP=6` prisms and exercise exactly the same refusal, so the larger one bought nothing but bytes in the
history — and `r3d_bump.slf` has the advantage of being the parent of the extracted layer beside it.

## BSD-3-Clause notice for `qgismesh.slf`

```
Copyright (c) 2020, Sebastian Schwindt
All rights reserved.
```

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
conditions of the BSD 3-Clause licence are met; see
<https://github.com/hydro-informatics/telemac/blob/main/LICENSE> for the full text.
