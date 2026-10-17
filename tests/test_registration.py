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

from pyramids.netcdf.ugrid import (
    CAPABILITIES,
    MESH_DRIVER_API_VERSION,
    RESERVED_NAMES,
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
    "xmstin",
    "xmdf",
    "3di",
    "tuflowfv",
    "esri_tin",
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
