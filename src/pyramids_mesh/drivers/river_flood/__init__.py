"""Drivers for river and flood hydraulic models.

Surface-water and 1D/2D hydraulic formats: HEC-RAS, FLO-2D, RiverFlow2D, SRH-2D, 3Di
(urban/rural flood), the SMS family (2DM/DAT) and the XMDF results format, Deltares 1D
networks (Delft3D FM / D-HYDRO) and GiD (the Iber model).

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
