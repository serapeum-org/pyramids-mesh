"""XDMF driver.

An XML index over an HDF5 payload, so two parsers. The HDF5 path is relative to the
XML file, not the working directory. Task D7.

HDF5 is read through the GDAL multidimensional API, over the GDAL vendored in the
``pyramids-gis`` wheel and activated by ``import pyramids`` -- group traversal,
attributes and typed array reads, with no declared dependency and no ``h5py``.
"""
