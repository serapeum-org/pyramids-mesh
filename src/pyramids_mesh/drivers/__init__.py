"""Format drivers, one module per vendor toolchain.

Grouped by vendor rather than by format, because formats from one vendor share a parser:
2DM and DAT are both SMS, DFSU and DFS2 are both ``mikeio``.

This module must stay free of eager driver imports. A missing extra would otherwise break
the whole package at import, instead of letting pyramids' registry skip that one driver
with a warning.
"""
