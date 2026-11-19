"""Drivers for computational-fluid-dynamics (CFD) meshes and data.

The SU2 (`.su2`) and OpenFOAM (`polyMesh`) solver meshes, the CGNS CFD standard, and
the XDMF data model (XML + HDF5, common in CFD/HPC output).

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
