# Reference

The API surface of `pyramids_mesh`, generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

`0.1.0` publishes no mesh API — the package root carries `__version__` and the import contract that
activates the vendored osgeo bindings. Each module added later gets its own page under this section,
and the names it exports are re-exported from the package root, so `from pyramids_mesh import ...`
stays the import worth using.

For the mesh data model itself — `Mesh2d`, `UgridDataset`, connectivity, graph algorithms,
regridding, mesh ↔ vector — see the
[pyramids UGRID reference](https://serapeum-org.github.io/pyramids/latest/). The split between the
two is [ADR-0003](../adr/0003-scope-boundary.md).

::: pyramids_mesh
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
