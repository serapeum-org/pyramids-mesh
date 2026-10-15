"""XMDF driver.

An HDF5 results container: results-only, lazy by nature, and carrying the per-step
``Active`` wet/dry mask. Task D6; see ``planning/design.md`` section 20.3.

HDF5 is read through the GDAL multidimensional API, over the GDAL vendored in the
``pyramids-gis`` wheel and activated by ``import pyramids`` -- group traversal,
attributes and typed array reads, with no declared dependency and no ``h5py``.
"""
