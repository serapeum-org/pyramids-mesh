"""DHI MIKE drivers: DFSU and DFS2. Requires the ``dhi`` extra.

``mikeio`` must be imported through ``require_optional`` at this module's import, not
at package import: it pulls matplotlib, which can select a GUI backend in a headless
run. Tasks D8 and D9; see ``planning/design.md`` section 20.3.
"""
