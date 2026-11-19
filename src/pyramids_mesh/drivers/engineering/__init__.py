"""Drivers for engineering FEM/CAE solver meshes.

Structural, mechanical and multiphysics solver and mesh-generator formats: Abaqus,
Nastran, Gmsh, Kratos, FLAC3D, MED (SALOME), Netgen, Medit (INRIA), AVS-UCD, Exodus and
Tecplot. CFD solvers (SU2, OpenFOAM) live in the sibling ``cfd`` subpackage.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
