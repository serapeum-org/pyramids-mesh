# Mike21 — DHI `.mesh` frame

The Mike21 reader. Plain text despite the DHI name, so it needs no dependency: a header
line, node records, an element header, element records.

| | |
|---|---|
| Driver name | `mike21` |
| Suffixes | `.mesh` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.mesh")
```

## Layout

```text
<item> <unit> <node_count> <projection...>     header
<id> <x> <y> <z> <boundary_code>               one per node
<element_count> <max_nodes> <type_code>        element header
<id> <n1> <n2> <n3> [n4]                        one per element, node ids 1-based
```

A triangle in a quad mesh is padded with a trailing `0` node id. Since ids are 1-based,
`0` is read as fill and becomes `-1` padding — never node index zero.

## Documented limitations

- **The projection field and the node-record column order are read as MDAL reads them,
  but could not be checked against a vendor file.** If a real Mike21 mesh orders its
  node columns differently, the coordinates would be wrong in a plausible way. This is
  the part of the driver a vendor sample should confirm — tracked for the `vendor_data`
  suite.
- **The projection is mapped to a CRS only when it is real.** A WKT string or an EPSG
  code the vendored PROJ validates is carried; a DHI-internal name such as `UTM-32` or
  `LONG/LAT` carries no datum and leaves `crs` as `None` rather than being guessed at.
  Validation is what tells the two apart — the field is not trusted for being present.
- **Node elevation becomes a `node_z` variable**, as it does for 2DM: the pyramids mesh
  model is planar and carries no per-node elevation coordinate (design §6). The z column
  is read as written, never fabricated.
- **The boundary code is not carried.** It marks open and land boundaries, which this
  package has nowhere to put on a bare frame; it is dropped rather than turned into a
  variable the mesh model would not recognise.
- Node ids are resolved through an id map rather than assumed contiguous, so a renumbered
  file is wired correctly; the 1-based numbering is recorded for a writer to restore.
- **Frame only.** The format carries no results, so there are no time-varying variables
  and no write support yet (write arrives with W4).

## API

::: pyramids_mesh.drivers.misc.Mike21Driver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
