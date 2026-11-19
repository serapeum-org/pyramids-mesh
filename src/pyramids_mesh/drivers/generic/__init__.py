"""Drivers for format-neutral interchange, CAD geometry and mesh generators.

Formats not tied to one solver: VTK, XDMF, CGNS, the geometry formats OBJ/STL/OFF,
the two TIN terrain surfaces (XMS TIN via the interchange module, and Esri TIN) plus
PLY, and the triangulators Triangle and EasyMesh.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
