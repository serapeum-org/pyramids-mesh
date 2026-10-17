"""pytest collection guard for the optional-dependency driver modules.

``pytest --doctest-modules src/pyramids_mesh`` (the ``static`` job, and the local
``doctest`` pre-commit hook) imports every module it collects so it can run the examples
in their docstrings. The DHI drivers in ``drivers/dhi.py`` import ``matplotlib`` and
``mikeio`` -- the ``dhi`` extra -- at module top, so that the driver registry skips them
with a warning when the extra is absent rather than failing the whole package. Doctest
collection imports the module directly, bypassing that registry guard, so in an
environment without the extra the import raises and collection would error out.

This conftest sits in ``src/`` -- beside the package, not inside it, so it is never
shipped in the wheel -- and is loaded whenever collection targets ``src/pyramids_mesh``.
When the extra cannot be imported it drops ``drivers/dhi.py`` from collection; the
dedicated ``dhi`` environment has the extra, imports it cleanly, and still doctests it.
"""

collect_ignore = []

try:  # the dhi extra; present only in the `dhi` environment
    import matplotlib  # noqa: F401
    import mikeio  # noqa: F401
except Exception:  # pragma: no cover - taken only in an environment without the extra
    collect_ignore.append("pyramids_mesh/drivers/dhi.py")
