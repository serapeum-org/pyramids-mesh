"""Drivers for computational-fluid-dynamics (CFD) solver meshes.

SU2 (native `.su2` mesh) and OpenFOAM (`polyMesh`).

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
