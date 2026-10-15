"""HEC-RAS 2D driver. Requires the ``hdf`` extra.

HDF5 with an undocumented internal schema that has changed across HEC-RAS releases.
Fail loudly on an unrecognised layout rather than reading what looks familiar.
Task D11.
"""
