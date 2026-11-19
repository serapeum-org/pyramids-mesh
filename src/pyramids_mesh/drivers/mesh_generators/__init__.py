"""Drivers for mesh-generator formats.

Tools whose output is a generated mesh: Gmsh, Netgen, Triangle/TetGen, EasyMesh and
Medit (INRIA).

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
