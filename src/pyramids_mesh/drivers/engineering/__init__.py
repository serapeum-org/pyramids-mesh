"""Drivers for engineering FEM/CAE and CFD solver meshes.

Structural, mechanical, multiphysics and CFD solver formats: Abaqus, Nastran, Gmsh,
Kratos, FLAC3D, MED (Salome), Netgen, Medit (INRIA), AVS-UCD, Exodus, SU2, OpenFOAM
and Tecplot.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
