"""Cross-driver invariant harness (S5).

Every test here is parametrised over **each registered driver**, so a new driver is
covered by registering it and exposing a ``build_minimal_fixture`` -- not by writing new
invariant tests. The invariants are the ones pyramids assumes and that fail *silently*:
0-based connectivity, ``-1`` trailing padding, counter-clockwise winding, and a
``shape`` that is right before any lazy load.

The driver set is resolved at collection time, so it shrinks without a word if a driver
stops being discovered; ``test_registration.py`` is what makes that visible.
"""

import inspect

import pyramids  # noqa: F401  # isort:skip

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset, mesh_drivers


@pytest.fixture(params=sorted(mesh_drivers()))
def driver(request):
    """Each registered driver in turn."""
    return mesh_drivers()[request.param]


@pytest.fixture
def minimal_file(driver, tmp_path):
    """A path to the smallest valid file for ``driver``, hand-built by its module."""
    module = inspect.getmodule(type(driver))
    builder = getattr(module, "build_minimal_fixture", None)
    if builder is None:
        pytest.skip(f"driver {driver.name!r} exposes no build_minimal_fixture")
    path = str(tmp_path / f"minimal{driver.suffixes[0]}")
    builder(path)
    return path


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


def _signed_area2(node_x, node_y, real):
    """Twice the signed area of the polygon through ``real`` node indices."""
    xs, ys = node_x[real], node_y[real]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def test_connectivity_is_zero_based_integer(driver, minimal_file):
    """Connectivity is an integer array whose real entries start at 0."""
    data, _ = _connectivity(driver.read(minimal_file, lazy=True))

    assert np.issubdtype(data.dtype, np.integer)
    real = data[data != -1]
    assert real.size
    assert real.min() >= 0


def test_connectivity_indices_are_in_range(driver, minimal_file):
    """No connectivity entry points past the last node."""
    data, n_node = _connectivity(driver.read(minimal_file, lazy=True))

    real = data[data != -1]
    assert real.max() < n_node


def test_fill_is_minus_one_and_only_trailing(driver, minimal_file):
    """Padding is exactly ``-1`` and occupies a contiguous tail of each row."""
    data, _ = _connectivity(driver.read(minimal_file, lazy=True))

    for row in np.atleast_2d(data):
        padding = np.flatnonzero(row == -1)
        if padding.size:
            assert padding.min() == row.size - padding.size
            assert bool((row[padding.min() :] == -1).all())


def test_faces_are_counter_clockwise(driver, minimal_file):
    """Every non-degenerate face has a positive signed area."""
    result = driver.read(minimal_file, lazy=True)
    if not isinstance(result, UgridDataset):
        pytest.skip("a 1D network has no faces")
    mesh = result.mesh

    for row in mesh.face_node_connectivity.data:
        real = row[row != -1]
        if real.size < 3:
            continue
        area2 = _signed_area2(mesh.node_x, mesh.node_y, real)
        if area2 == 0:
            continue
        assert area2 > 0


def test_variable_shape_matches_its_loaded_array(driver, minimal_file):
    """``shape`` is the array's real shape, so the along-time API can trust it."""
    result = driver.read(minimal_file, lazy=True)
    if not isinstance(result, UgridDataset):
        pytest.skip("a 1D network carries no data variables")

    for name in result.data_variable_names:
        variable = result.get(name)
        assert tuple(variable.shape) == tuple(np.asarray(variable.data).shape), name


def test_lazy_driver_defers_its_values(driver, minimal_file):
    """A driver declaring ``lazy_data`` reports a source without loading values."""
    if "lazy_data" not in set(driver.capabilities):
        pytest.skip("driver does not declare lazy_data")
    result = driver.read(minimal_file, lazy=True)
    if not isinstance(result, UgridDataset):
        pytest.skip("a 1D network carries no data variables")

    for name in result.data_variable_names:
        variable = result.get(name)
        assert variable.has_data_source, name
        assert variable._data is None, name


def test_crs_is_a_valid_value_or_none(driver, minimal_file):
    """The CRS is a real CRS or ``None`` -- never a malformed string."""
    result = driver.read(minimal_file, lazy=True)
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
