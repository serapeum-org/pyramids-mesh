"""Tests for the XMDF reader (D6), against a real committed sample.

XMDF is results-only (``needs_mesh``), read through the vendored GDAL multidimensional
API. The fixture is a genuine XMDF file and its paired 2DM mesh committed under
``tests/data/xmdf/`` -- 25 nodes, 16 faces, one node-located dataset ``Minimum dt`` over
three timesteps with a per-face active mask, plus a ``/Final`` summary that must not be
read as a timestep. The tests read the committed bytes; constructing an HDF5 file through
GDAL and reopening it in the same process is unreliable on this platform, and a real file
is the better oracle anyway.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.xmdf import XmdfDriver

_DATA = Path(__file__).parent.parent / "data" / "xmdf"
_MESH = _DATA / "minimal.2dm"
_XMDF = _DATA / "minimal.xmdf"


@pytest.fixture
def mesh():
    """The 25-node, 16-face 2DM mesh the XMDF results attach to."""
    return UgridDataset.read_file(str(_MESH))


def test_read_raises_pointing_at_attach_data():
    """``read`` refuses and names ``attach_data`` rather than failing obscurely."""
    with pytest.raises(ValueError, match="no read\\(\\)") as caught:
        XmdfDriver().read(str(_XMDF))
    assert "attach_data" in str(caught.value)


def test_attaches_node_variable_with_time_axis(mesh):
    """The temporal dataset attaches as a node-located variable over its three steps."""
    attached = mesh.attach_data(str(_XMDF))

    variable = attached.get("Minimum dt")
    assert variable.location == "node"
    assert variable.shape == (3, 25)
    assert variable.attributes["time_values"].tolist() == [0.0, 0.5, 1.0]
    assert variable.data[0][0] == pytest.approx(0.3192754, rel=1e-5)


def test_active_becomes_face_located_sibling(mesh):
    """The per-step wet/dry mask is a standalone face-located ``<name>_active``."""
    attached = mesh.attach_data(str(_XMDF))

    active = attached.get("Minimum dt_active")
    assert active.location == "face"
    assert active.shape == (3, 16)

    # the parent variable is node-located; the mask is face-located -- different count
    # and location, so it cannot ride along as ParsedVariable.active
    assert attached.get("Minimum dt").location == "node"
    assert active.data.shape == (3, 16)


def test_final_summary_is_not_read_as_a_timestep(mesh):
    """The ``/Final`` snapshot is skipped, so the temporal dataset is not doubled."""
    attached = mesh.attach_data(str(_XMDF))

    names = list(attached.data_variable_names)
    assert names.count("Minimum dt") == 1
    # three real timesteps, not four (the Final snapshot would have added one)
    assert attached.get("Minimum dt").shape[0] == 3


def test_mins_and_maxs_carried_as_range_attributes(mesh):
    """The precomputed Mins/Maxs ranges travel on the variable's attributes."""
    variable = mesh.attach_data(str(_XMDF)).get("Minimum dt")
    assert "minimum" in variable.attributes
    assert "maximum" in variable.attributes
    assert len(variable.attributes["maximum"]) == 3


def test_values_are_lazy(mesh):
    """A lazy read reports a source without loading the values."""
    variable = mesh.attach_data(str(_XMDF)).get("Minimum dt")
    assert variable.has_data_source
    assert variable._data is None
    assert variable.data.shape == (3, 25)


def test_loader_reopens_and_survives_repeated_reads(mesh):
    """The lazy loader reopens the file each call, so repeated reads all succeed."""
    variable = mesh.attach_data(str(_XMDF)).get("Minimum dt")
    first = variable.data
    second = variable.data
    assert np.array_equal(first, second)


def test_element_count_mismatch_is_refused(tmp_path):
    """Attaching to a mesh of the wrong size is refused -- the only safeguard there is."""
    wrong = tmp_path / "wrong.2dm"
    wrong.write_text(
        "MESH2D\nND 1 0.0 0.0 0.0\nND 2 1.0 0.0 0.0\nND 3 1.0 1.0 0.0\nE3T 1 1 2 3 1\n",
        encoding="utf-8",
    )
    dataset = UgridDataset.read_file(str(wrong))
    with pytest.raises(ValueError, match="do not belong to this mesh"):
        dataset.attach_data(str(_XMDF))


def test_sniff_requires_hdf5_magic_and_file_type(tmp_path):
    """``can_open`` needs the HDF5 magic and a ``File Type`` of ``Xmdf``."""
    driver = XmdfDriver()
    assert driver.can_open(str(_XMDF)) is True

    not_hdf5 = tmp_path / "text.xmdf"
    not_hdf5.write_text("not an hdf5 file\n", encoding="utf-8")
    assert driver.can_open(str(not_hdf5)) is False

    empty = tmp_path / "empty.xmdf"
    empty.write_bytes(b"")
    assert driver.can_open(str(empty)) is False

    directory = tmp_path / "folder.xmdf"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_plain_netcdf_under_h5_suffix_is_declined(tmp_path):
    """A non-XMDF HDF5-ish file under the shared ``.h5`` suffix is declined by File Type.

    A binary file with the right magic but no ``File Type`` of ``Xmdf`` is not XMDF.
    """
    fake = tmp_path / "other.h5"
    fake.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert XmdfDriver().can_open(str(fake)) is False


def test_declares_needs_mesh_and_lazy(mesh):
    """The driver advertises needs_mesh, read_data and lazy_data."""
    caps = set(XmdfDriver.capabilities)
    assert caps == {"needs_mesh", "read_data", "lazy_data"}
