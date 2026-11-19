"""Drivers for CAD/surface geometry, TIN terrain and visualization containers.

VTK/VTU, the geometry formats OBJ/STL/OFF, the interchange pair PLY and XMS TIN, and the
Esri TIN terrain surface.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
