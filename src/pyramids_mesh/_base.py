"""Driver scaffolding: the template-method base class and the parsed-mesh intermediate.

Holds ``BaseDriver``, whose ``read`` is a template method running ``_parse`` ->
``_validate`` -> construct, so a format driver implements ``_parse`` and nothing
structural. Also holds ``ParsedMesh`` / ``ParsedVariable``, this package's internal
intermediate between a file and ``pyramids.netcdf.ugrid`` types.

``ParsedMesh`` is deliberately richer than pyramids' ``construct.MeshArrays`` tuple,
which cannot carry edge connectivity, per-variable time, per-variable nodata or units,
a WKT CRS (Selafin and DFSU have no EPSG), a 1D network, or node elevation.

See ``planning/design.md`` sections 3, 6, 17 and 19. Implemented by task S2.
"""
