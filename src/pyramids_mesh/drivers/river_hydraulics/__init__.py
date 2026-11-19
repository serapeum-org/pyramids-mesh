"""Drivers for river and flood hydraulic models.

Surface-water and 1D/2D hydraulic formats: HEC-RAS, FLO-2D, RiverFlow2D, SRH-2D,
the SMS family (2DM/DAT) and its XMS siblings (Esri TIN, XMDF), Deltares 1D networks,
DHI MIKE 21 and GiD (the Iber model).

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
