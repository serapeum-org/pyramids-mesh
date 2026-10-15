"""Geometry interchange drivers: PLY, XMS TIN, Esri TIN. Requires ``interchange``.

Not solver formats -- no time axis, and elevation per node. pyramids' mesh model is
planar, so z becomes a node-located variable and a round trip is not geometrically
faithful. Task D5; see ``planning/design.md`` section 6.
"""
