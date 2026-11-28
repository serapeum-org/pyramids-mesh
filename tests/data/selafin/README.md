# Selafin sample files — sources and licences

Four real TELEMAC Selafin files, committed so the driver's tests run offline like every other fixture in
`tests/data/`. They are not synthetic: `write_selafin` wrote every other Selafin fixture in this repository, so
the reader and the writer share one set of assumptions about the record layout and cannot contradict each other.
These files were written by other software, which is the only way to catch a layout understood wrongly in the
same way at both ends — the `IKLE` row-major order was settled this way.

`pyramids-mesh` is GPL-3.0, so redistributing the GPL-3.0 files below is licence-compatible, and the
BSD-3-Clause file is permissive. The wheel does not carry them (`[tool.setuptools.packages.find]` picks up
`src/` only), and the default sdist does not carry `tests/data/`.

| File | Bytes | Source | Licence |
|---|---|---|---|
| `r2d_tidal_flats.slf` | 241,780 | [oceanmodeling/xarray-selafin](https://github.com/oceanmodeling/xarray-selafin) `tests/data/` | GPL-3.0 |
| `r3d_tidal_flats.slf` | 5,285,740 | [oceanmodeling/xarray-selafin](https://github.com/oceanmodeling/xarray-selafin) `tests/data/` | GPL-3.0 |
| `r3d_bump_extracted_bottom_layer.slf` | 136,580 | [oceanmodeling/xarray-selafin](https://github.com/oceanmodeling/xarray-selafin) `tests/data/` | GPL-3.0 |
| `qgismesh.slf` | 505,376 | [hydro-informatics/telemac](https://github.com/hydro-informatics/telemac) `bk-slf/` | BSD-3-Clause |

## What each file covers

- **`r2d_tidal_flats.slf`** — TELEMAC 2D output for the "Sloped flume Rouse profile test" validation case: 648
  nodes, 1030 triangles, 17 time steps, 5 node variables, big-endian, single precision, a date record and 264
  boundary nodes. The file the `IKLE` order, the header metadata and the round trip are checked against.
- **`qgismesh.slf`** — geometry written by **BlueKenue/QGIS** rather than by TELEMAC: 12,814 nodes, 25,007
  triangles, one `BOTTOM` field, projected UTM coordinates. A second, independent writer, and the file that shows
  `IPOBO` must be read from its own record — it carries 619 boundary nodes while leaving `IPARAM[7]` at zero.
- **`r3d_tidal_flats.slf`** — a genuine layered 3D file: `NDP=6` prisms over 21 planes. The refusal path.
- **`r3d_bump_extracted_bottom_layer.slf`** — one layer extracted from a 3D run by TELEMAC itself: `NDP=3`
  triangles that inherit the parent's `IPARAM[6] = 5`. The file that forced the 3D guard onto `NDP`, since the
  plane count refuses a mesh that reads perfectly well.

## BSD-3-Clause notice for `qgismesh.slf`

Copyright (c) hydro-informatics contributors. Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions of the BSD 3-Clause licence are met; see
<https://github.com/hydro-informatics/telemac/blob/main/LICENSE> for the full text.
