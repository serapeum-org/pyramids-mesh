"""Registration contract for the mesh drivers this package ships (S5).

The invariant harness in ``test_invariants.py`` parametrises over whatever
``mesh_drivers()`` discovers, so it shrinks silently if a driver stops being
registered -- a dependency-free driver dropped by a packaging change, or a name
collision. This module makes that visible: it names the drivers that must always be
present and the ones gated behind an extra, and checks every registered driver is
structurally valid.

A driver is added to ``DEPENDENCY_FREE_DRIVERS`` (or ``OPTIONAL_DRIVERS``) in the same
change that registers its entry point.
"""

import pyramids  # noqa: F401  # isort:skip

import sys
import types
import warnings
from importlib.util import find_spec

import pytest
from pyramids.netcdf.ugrid import (
    CAPABILITIES,
    MESH_DRIVER_API_VERSION,
    RESERVED_NAMES,
    _drivers,
    mesh_drivers,
)

# Drivers that need no optional extra, so discovery must always find them.
DEPENDENCY_FREE_DRIVERS: set[str] = {
    "2dm",
    "dat",
    "mike21",
    "selafin",
    "sww",
    "ply",
    "xms-tin",
    "xmdf",
    "3di",
    "tuflowfv",
    "arcgis-tin",
    "xdmf",
    "flo2d",
    "hecras",
    "adcirc",
    "delft3d-1d",
    "dhydro-1d",
    "schism",
    "gmsh",
    "vtk",
    "modflow6-disv",
    "mpas",
    "icon",
    "fvcom",
    "triangle",
    "easymesh",
    "exodus",
    "su2",
    "srh2d",
    "modflow6-disu",
    "feflow",
    "scrip",
    "esmf",
    "geos",
    "cgns",
    "med",
    "nastran",
    "abaqus",
    "avs-ucd",
    "medit",
    "netgen",
    "tecplot",
    "flac3d",
    "kratos",
    "openfoam",
    "obj",
    "stl",
    "off",
    "gid",
    "riverflow2d",
}

# Drivers gated behind an optional extra; present only when the extra is installed.
OPTIONAL_DRIVERS: set[str] = {
    "dfsu",  # the dhi extra (mikeio)
}


def test_dependency_free_drivers_are_registered():
    """Every always-available driver is discovered, or a packaging change broke it."""
    missing = DEPENDENCY_FREE_DRIVERS - set(mesh_drivers())
    assert not missing, f"dependency-free drivers missing from discovery: {missing}"


def test_no_unexpected_drivers_registered():
    """Nothing is discovered that this package did not declare and catalogue."""
    allowed = DEPENDENCY_FREE_DRIVERS | OPTIONAL_DRIVERS
    unexpected = set(mesh_drivers()) - allowed
    assert not unexpected, f"unexpected drivers registered: {unexpected}"


def test_every_registered_driver_is_structurally_valid():
    """Each registered driver satisfies the contract pyramids enforces at discovery."""
    for name, driver in mesh_drivers().items():
        assert name not in RESERVED_NAMES, name
        assert driver.api_version == MESH_DRIVER_API_VERSION, name
        capabilities = set(driver.capabilities)
        assert ("read_mesh" in capabilities) ^ ("needs_mesh" in capabilities), name
        # A driver inherits empty defaults for these from BaseDriver, so a subclass
        # that forgets one would otherwise register nameless or suffixless.
        assert driver.name, f"{name} registered with no name"
        assert driver.long_name, f"{name} registered with no long_name"
        assert driver.suffixes, f"{name} registered with no suffixes"
        for suffix in driver.suffixes:
            assert suffix == suffix.lower(), suffix
            assert suffix.startswith("."), suffix
        assert capabilities <= set(CAPABILITIES), name


def test_dfsu_driver_declares_its_mikeio_extra():
    """DfsuDriver names ``mikeio`` as its extra so the registry can probe it with find_spec.

    Importing the driver module must succeed even without ``mikeio`` installed -- the import
    itself is part of the silent-skip contract, because the registry loads the class before it
    probes the extra. A module that imports its backend eagerly would raise on load and be
    reported as a broken plugin instead of skipped.
    """
    from pyramids_mesh.drivers.coastal_ocean.mike import DfsuDriver

    assert DfsuDriver.extra == "mikeio"


def test_discovery_emits_no_dfsu_runtimewarning():
    """Driver discovery never warns about ``dfsu``, whether or not ``mikeio`` is installed.

    With ``mikeio`` present the class loads cleanly; without it the registry takes the silent
    ``extra_missing`` skip. Either way a ``dfsu`` ``RuntimeWarning`` is noise every non-``dhi``
    read would print, and it must not appear.
    """
    # _drivers._CACHE is an upstream private, used deliberately (as the repo's other driver
    # registration tests do) to save and restore the discovery cache around a forced re-discovery.
    original = _drivers._CACHE
    try:
        _drivers.clear_driver_cache()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh_drivers()
        dfsu_warnings = [
            str(w.message)
            for w in caught
            if issubclass(w.category, RuntimeWarning) and "dfsu" in str(w.message)
        ]
        assert not dfsu_warnings, dfsu_warnings
    finally:
        _drivers._CACHE = original


@pytest.mark.skipif(
    find_spec("mikeio") is not None,
    reason="dfsu is registered when the dhi extra (mikeio) is installed",
)
def test_dfsu_absent_from_registry_without_mikeio():
    """Without ``mikeio`` the driver is skipped (not broken) and stays out of the registry."""
    assert "dfsu" not in mesh_drivers()


def test_mikeio_does_not_repin_backend_when_already_imported(monkeypatch):
    """``_mikeio`` returns an already-imported ``mikeio`` without re-pinning the backend.

    ``mikeio`` selects a backend on import, so ``Agg`` is pinned before the *first* import; re-pinning
    on a later call would clobber a backend chosen afterwards (e.g. cleopatra's inline backend while a
    plot is being drawn). With a ``mikeio`` already in ``sys.modules`` the helper must return it and
    leave ``matplotlib.use`` untouched. ``matplotlib.use`` is spied, never called, so the real process
    backend is not mutated. Skipped where matplotlib is absent (a bare wheel env has no backend to pin).
    """
    matplotlib = pytest.importorskip("matplotlib")

    from pyramids_mesh.drivers.coastal_ocean.mike import _mikeio

    calls: list = []
    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: calls.append(args))
    fake = types.ModuleType("mikeio")
    monkeypatch.setitem(sys.modules, "mikeio", fake)

    assert _mikeio() is fake, "the helper must return the already-imported mikeio module"
    assert calls == [], f"matplotlib.use must not be called when mikeio is already imported, got {calls}"


@pytest.mark.skipif(
    find_spec("mikeio") is not None,
    reason="forces the first-import path, which only occurs when mikeio is absent",
)
def test_mikeio_pins_agg_before_importing_when_absent(monkeypatch):
    """``_mikeio`` pins ``Agg`` before importing ``mikeio`` the first time, even on the failing path.

    The pin must run before the import is attempted, so a headless run never selects a GUI backend --
    and it still runs when the import then raises ``ModuleNotFoundError`` because the backend is absent.
    ``matplotlib.use`` is spied, so the real process backend is not mutated. Skipped where matplotlib is
    absent (a bare wheel env has no backend to pin). This covers the Agg-before-import *ordering*; the
    re-pin *guard* is covered by ``test_mikeio_does_not_repin_backend_when_already_imported``.
    """
    matplotlib = pytest.importorskip("matplotlib")

    from pyramids_mesh.drivers.coastal_ocean.mike import _mikeio

    calls: list = []
    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: calls.append(args[0] if args else None))
    with pytest.raises(ModuleNotFoundError):
        _mikeio()
    assert calls == ["Agg"], f"Agg must be pinned once before the mikeio import, got {calls}"
