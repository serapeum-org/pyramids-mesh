"""Aquaveo XMS surface-water formats.

Two Aquaveo modules share this nested package inside the ``river_flood`` domain: ``sms``
(the SMS 2DM/DAT formats) and ``xms`` (the XMDF results container). They are grouped here
only because they come from the same vendor suite; the domain is still ``river_flood``.

Import-free by contract: driver discovery is entry-point based, so a missing optional
extra skips one driver with a warning instead of breaking the package at import.
"""
