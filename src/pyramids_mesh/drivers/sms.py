"""SMS-family drivers: 2DM (mesh frame) and DAT (results-only).

2DM is the reference driver every later one copies (task D1). DAT carries values but
no topology, so it declares ``needs_mesh`` and implements ``read_data`` rather than
``read`` (task D2). See ``planning/design.md`` section 20.1.
"""
