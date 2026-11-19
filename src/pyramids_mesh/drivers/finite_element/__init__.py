"""Drivers for finite-element / CAE analysis formats.

Structural, mechanical and multiphysics analysis: Abaqus, Nastran, Kratos, FLAC3D,
SALOME (MED), SEACAS (Exodus) and AVS-UCD. Fluid solvers live in ``cfd``.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
