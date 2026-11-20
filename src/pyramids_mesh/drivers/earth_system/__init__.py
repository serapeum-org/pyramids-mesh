"""Drivers for earth-system model grids and regridding weights.

Global atmosphere/ocean unstructured grids and remapping descriptions: MPAS, ICON,
NASA GEOS (cubed-sphere), ESMF unstructured mesh and SCRIP remapping grids.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
