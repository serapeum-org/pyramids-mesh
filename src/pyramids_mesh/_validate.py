"""Enforcement of the invariants pyramids assumes of a mesh.

One shared implementation, because every one of these failures is **silent**: a driver
that forgets one produces wrong numbers rather than an error.

1. Connectivity is 0-indexed -- ``original_start_index`` only records the file's base.
2. Faces are fill-padded to a rectangular array, never ragged.
3. The fill value does not collide with a valid index (``-1``).
4. Face winding is counter-clockwise -- area, centroid, ``polygonize`` and ``voronoi``
   all assume it.
5. Every variable's declared shape matches its element and time counts.

See ``planning/design.md`` section 17.3. Implemented by task S3.
"""
