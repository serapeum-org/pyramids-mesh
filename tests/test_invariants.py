"""Cross-driver invariant harness (S5).

Every test here is parametrised over **each registered driver**, so a new driver is
covered by registering it and exposing a fixture builder -- not by writing new invariant
tests. The invariants are the ones pyramids assumes and that fail *silently*: 0-based
connectivity, ``-1`` trailing padding, counter-clockwise winding, and a ``shape`` that is
right before any lazy load.

The builder is looked up as ``build_minimal_fixture_<driver name>`` first and
``build_minimal_fixture`` second, because one module may hold several drivers: ``sms``
holds both 2DM and DAT, and a single shared name would hand the DAT driver a 2DM file.

A driver declaring ``needs_mesh`` carries values and no topology, so it has no ``read``
and the topology invariants cannot apply to it. Those tests skip for it by name; the
``can_open`` invariants still run, and its data invariants belong in its own driver tests,
which know the mesh to attach it to.

The driver set is resolved at collection time, so it shrinks without a word if a driver
stops being discovered; ``test_registration.py`` is what makes that visible.
"""

import inspect

import pyramids  # noqa: F401  # isort:skip

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset, mesh_drivers

from pyramids_mesh._validate import _area_tolerance, _signed_area_2


@pytest.fixture(params=sorted(mesh_drivers()))
def driver(request):
    """Each registered driver in turn."""
    return mesh_drivers()[request.param]


@pytest.fixture
def minimal_file(driver, tmp_path):
    """A path to the smallest valid file for ``driver``, hand-built by its module."""
    module = inspect.getmodule(type(driver))
    # Per-driver name first: one module may hold several drivers, and the shared name
    # would otherwise hand every driver in it the first driver's format.
    builder = getattr(module, f"build_minimal_fixture_{driver.name}", None)
    if builder is None:
        builder = getattr(module, "build_minimal_fixture", None)
    if builder is None:
        pytest.skip(f"driver {driver.name!r} exposes no build_minimal_fixture")
    path = str(tmp_path / f"minimal{driver.suffixes[0]}")
    builder(path)
    return path


def _read_or_skip(driver, path, **kwargs):
    """Read ``path``, skipping the test when the driver has no ``read`` to call.

    A ``needs_mesh`` driver reads a results-only format: it implements ``read_data`` and
    its ``read`` refuses by contract, so every topology invariant here is inapplicable
    rather than failing. Skipping by capability keeps that explicit -- a driver that
    forgot to declare ``needs_mesh`` still fails these tests, which is the point.
    """
    if "needs_mesh" in set(driver.capabilities):
        pytest.skip(
            f"{driver.name!r} declares needs_mesh: it carries values only, so it has "
            f"no read() and no topology to check here"
        )
    return driver.read(path, **kwargs)


@pytest.fixture
def junk_files(tmp_path):
    """Files of every shape ``can_open`` must decline without raising."""
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    text = tmp_path / "other.txt"
    text.write_text("this is not a mesh\njust prose\n", encoding="utf-8")
    binary = tmp_path / "other.bin"
    binary.write_bytes(bytes(range(256)) * 4)
    directory = tmp_path / "a_directory"
    directory.mkdir()
    return {
        "empty file": str(empty),
        "foreign text": str(text),
        "foreign binary": str(binary),
        "directory": str(directory),
    }


def _connectivity(result):
    """The connectivity array and node count of a dataset or a network."""
    if isinstance(result, UgridDataset):
        return result.mesh.face_node_connectivity.data, result.mesh.n_node
    return result.edge_node_connectivity.data, result.n_node


def _winding(node_x, node_y, vertices):
    """Return the signed area of a face and the tolerance below which it is degenerate.

    Deliberately the package's own functions rather than a reimplementation. Summing
    over absolute coordinates cancels at projected magnitudes, and comparing against an
    exact zero never fires there, so a harness that rolled its own would assert on noise
    for the first driver whose fixture is in a real CRS -- and could disagree with the
    code it exists to check.
    """
    xs, ys = node_x[vertices], node_y[vertices]
    return _signed_area_2(xs, ys), _area_tolerance(xs, ys)


def test_connectivity_is_zero_based_integer(driver, minimal_file):
    """Connectivity is an integer array whose real entries start at 0."""
    data, _ = _connectivity(_read_or_skip(driver, minimal_file, lazy=True))

    assert np.issubdtype(data.dtype, np.integer)
    real = data[data != -1]
    assert real.size
    assert real.min() >= 0


def test_connectivity_indices_are_in_range(driver, minimal_file):
    """No connectivity entry points past the last node."""
    data, n_node = _connectivity(_read_or_skip(driver, minimal_file, lazy=True))

    real = data[data != -1]
    assert real.max() < n_node


def test_fill_is_minus_one_and_only_trailing(driver, minimal_file):
    """Padding is exactly ``-1`` and occupies a contiguous tail of each row."""
    data, _ = _connectivity(_read_or_skip(driver, minimal_file, lazy=True))

    for row in np.atleast_2d(data):
        padding = np.flatnonzero(row == -1)
        if padding.size:
            assert padding.min() == row.size - padding.size
            assert bool((row[padding.min() :] == -1).all())


def test_faces_are_counter_clockwise(driver, minimal_file):
    """Every non-degenerate face has a positive signed area."""
    result = _read_or_skip(driver, minimal_file, lazy=True)
    if not isinstance(result, UgridDataset):
        pytest.skip("a 1D network has no faces")
    mesh = result.mesh

    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        if vertices.size < 3:
            continue
        area2, tolerance = _winding(mesh.node_x, mesh.node_y, vertices)
        if abs(area2) <= tolerance:
            continue
        assert area2 > 0


def test_variable_shape_matches_its_loaded_array(driver, minimal_file):
    """``shape`` is the array's real shape, so the along-time API can trust it."""
    result = _read_or_skip(driver, minimal_file, lazy=True)
    if not isinstance(result, UgridDataset):
        pytest.skip("a 1D network carries no data variables")

    for name in result.data_variable_names:
        variable = result.get(name)
        assert tuple(variable.shape) == tuple(np.asarray(variable.data).shape), name


def test_lazy_driver_defers_its_values(driver, minimal_file):
    """A driver declaring ``lazy_data`` reports a source without loading values."""
    if "lazy_data" not in set(driver.capabilities):
        pytest.skip("driver does not declare lazy_data")
    result = _read_or_skip(driver, minimal_file, lazy=True)
    if not isinstance(result, UgridDataset):
        pytest.skip("a 1D network carries no data variables")

    for name in result.data_variable_names:
        variable = result.get(name)
        assert variable.has_data_source, name
        assert variable._data is None, name


def test_crs_is_a_valid_value_or_none(driver, minimal_file):
    """The CRS is a real CRS or ``None`` -- never a malformed string."""
    result = _read_or_skip(driver, minimal_file, lazy=True)
    crs = result.crs_wkt if isinstance(result, UgridDataset) else result.crs

    assert crs is None or isinstance(crs, (str, int))
    if isinstance(crs, str):
        assert crs.strip()


def test_can_open_declines_junk_without_raising(driver, junk_files):
    """``can_open`` answers False -- never raises -- for anything foreign."""
    for label, path in junk_files.items():
        assert driver.can_open(path) is False, label


def test_can_open_declines_right_suffix_with_garbage(driver, tmp_path):
    """A matching suffix is not enough: the content is sniffed too."""
    path = tmp_path / f"garbage{driver.suffixes[0]}"
    path.write_bytes(b"\x00\x01not a mesh at all\xff\xfe")

    assert driver.can_open(str(path)) is False
