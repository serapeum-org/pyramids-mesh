# Reference

The API surface of `pyramids_mesh`, generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

What this package publishes is **drivers**, not an API you call directly. A driver is registered
with `pyramids-gis` through the `pyramids.mesh_drivers` entry-point group, so installing this
package is the whole integration: `pyramids` then dispatches a read of a format it does not handle
itself to the driver that claims it.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.2dm")   # resolved to this package's 2dm driver
```

So the entry points below are the surface that matters, and each driver has its own page listing
its suffixes, capabilities and documented limitations:

| Driver | Format | Page |
|---|---|---|
| `2dm` | SMS / TUFLOW 2DM mesh frame | [2DM](2dm.md) |
| `dat` | SMS ASCII results, attached to a mesh | [DAT](dat.md) |
| `mike21` | DHI Mike21 mesh frame | [Mike21](mike21.md) |
| `selafin` | TELEMAC Selafin / Serafin | [Selafin](selafin.md) |
| `sww` | ANUGA SWW (NetCDF) | [SWW](sww.md) |
| `ply` | Stanford PLY polygon | [PLY](ply.md) |
| `xmstin` | XMS / SMS TIN | [XMS TIN](xmstin.md) |

The package root deliberately re-exports nothing (`__all__` is empty): it carries `__version__` and
the import contract that activates the vendored osgeo bindings. Driver classes are reached by their
module path — `pyramids_mesh.drivers.sms.TwoDmDriver` — which is also how their entry points name
them, though normal use goes through `pyramids` rather than importing them at all.

For the mesh data model itself — `Mesh2d`, `Network1d`, `UgridDataset`, connectivity, graph
algorithms, regridding, mesh ↔ vector — see the
[pyramids UGRID reference](https://serapeum-org.github.io/pyramids/latest/). The split between the
two is [ADR-0005](../adr/0005-file-format-drivers.md), which supersedes ADR-0003 in full.

::: pyramids_mesh
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
