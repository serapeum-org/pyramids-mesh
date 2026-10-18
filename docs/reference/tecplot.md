# Tecplot — finite-element ASCII mesh

The Tecplot finite-element ASCII reader. A `.dat`/`.tec` file is plain text: a `TITLE`
line, a `VARIABLES = "X" "Y" "Z" …` list, a finite-element `ZONE` header, then the
variable values and the element connectivity. The zone header carries `N=<nodes>` and
`E=<elements>`, the element type as `ZONETYPE=FETRIANGLE|FEQUADRILATERAL|FETETRAHEDRON|FEBRICK`
(or the legacy `ET=TRIANGLE|QUADRILATERAL|TETRAHEDRON|BRICK`), the data layout as
`DATAPACKING=POINT|BLOCK` (or the legacy `F=FEPOINT|FEBLOCK`), and an optional
`VARLOCATION([k]=CELLCENTERED)` marking cell-centred variables. This is a geometry reader:
the `X`/`Y` (and `Z`) coordinates and the connectivity are read, everything else skipped.

| | |
|---|---|
| Driver name | `tecplot` |
| Suffixes | `.dat`, `.tec` |
| Capabilities | `read_mesh` |
| Extra required | none |
| Returns | `UgridDataset` |

```python
from pyramids_mesh.drivers.tecplot import TecplotDriver

dataset = TecplotDriver().read("mesh.tec")
dataset.mesh.n_node, dataset.mesh.n_face
```

## Documented limitations

- **It is a geometry reader** — only `X`/`Y` (and `Z`) are read; every other variable, and
  every cell-centred variable, is skipped. `VARLOCATION` is parsed only so a cell-centred
  variable's block length (`E` rather than `N`) is read correctly and the connectivity
  stays aligned.
- **A volumetric element becomes one face** — `FETETRAHEDRON` (4 nodes) and `FEBRICK`
  (8 nodes) are read as a single face of that many nodes rather than expanded into bounding
  faces (a 3D mesh is not modelled yet, design §21.1), so winding is judged in the `xy`
  plane like any other face, matching the Exodus reader's precedent.
- **The first finite-element zone is read** — a file carrying more than one zone yields
  only the first; the rest are ignored.
- **Connectivity is 1-based**, resolved to 0-based with the `1` base recorded for a writer.
  A single-element-type zone needs no fill padding.
- **An ordered (structured I/J/K) zone is refused** — this reader builds finite-element
  meshes only.
- Because `.dat` is a generic suffix the SMS DAT and FVCOM grid also use, `can_open`
  requires a Tecplot signature (a `TITLE`/`VARIABLES`/`ZONE` keyword near the top), so it
  never steals one of those files.
- There is no CRS in the format, so `crs` is `None`; it is never guessed from the
  coordinate magnitudes.

## API

::: pyramids_mesh.drivers.tecplot.TecplotDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
