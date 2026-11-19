"""Drivers for coastal-ocean and shallow-water circulation models.

Unstructured hydrodynamic meshes from estuary, coastal and ocean circulation codes:
ADCIRC and SCHISM (sharing the ``_gr3`` ASCII-grid parser), FVCOM, TELEMAC (Selafin),
ANUGA (SWW), TUFLOW FV, DHI MIKE (the MIKE 21 2D model and the DFSU result format)
and 3Di.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
