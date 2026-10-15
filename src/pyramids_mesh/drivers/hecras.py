"""HEC-RAS 2D driver.

HDF5 with an undocumented internal schema that has changed across HEC-RAS releases.
Fail loudly on an unrecognised layout rather than reading what looks familiar.
Task D11.

HDF5 is read through the GDAL multidimensional API, over the GDAL vendored in the
``pyramids-gis`` wheel and activated by ``import pyramids`` -- group traversal,
attributes and typed array reads, with no declared dependency and no ``h5py``.
"""
