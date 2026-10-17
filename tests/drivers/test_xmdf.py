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

from pyramids_mesh.drivers.xmdf import _GROUPTYPE, XmdfDriver, _XmdfReader

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


def test_active_sibling_is_lazy_like_its_parent(mesh):
    """The active-mask sibling is loader-backed under a lazy attach, not materialised.

    The parent values are lazy, so the ``<name>_active`` sibling must be too -- it reports
    a data source and holds no in-memory array until read.
    """
    active = mesh.attach_data(str(_XMDF)).get("Minimum dt_active")
    assert active.has_data_source
    assert active._data is None
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


class _FakeAttribute:
    """A stand-in for a GDAL attribute whose ``Read`` returns a scalar value."""

    def __init__(self, value) -> None:
        self._value = value

    def Read(self):
        return self._value


class _FakeDim:
    """A stand-in GDAL dimension that reports only its size."""

    def __init__(self, size: int) -> None:
        self._size = size

    def GetSize(self) -> int:
        return self._size


class _FakeArray:
    """A stand-in GDAL MD-array reporting its dimensions and, optionally, a value.

    Pass ``rank`` when only the number of axes matters (the guard checks), or ``shape``
    when the sizes matter (the active-mask width check).
    """

    def __init__(self, rank: int | None = None, value=None, shape=None) -> None:
        if shape is not None:
            self._dims = [_FakeDim(size) for size in shape]
        else:
            self._dims = [_FakeDim(0) for _ in range(rank or 0)]
        self._value = value

    def GetDimensions(self) -> list:
        return self._dims

    def ReadAsArray(self):
        return self._value


class _FakeNamed:
    """A stand-in for a GDAL attribute handle that only reports its name."""

    def __init__(self, name: str) -> None:
        self._name = name

    def GetName(self) -> str:
        return self._name


class _FakeGroup:
    """A stand-in GDAL group exposing named arrays and a ``Grouptype`` attribute."""

    def __init__(self, grouptype: str = "DATASET SCALAR", arrays=None) -> None:
        self._grouptype = grouptype
        self._arrays = arrays or {}

    def GetAttribute(self, name: str) -> _FakeAttribute:
        return _FakeAttribute(self._grouptype)

    def GetAttributes(self) -> list:
        return [_FakeNamed(_GROUPTYPE)]

    def GetMDArrayNames(self) -> list:
        return list(self._arrays)

    def OpenMDArray(self, name: str) -> _FakeArray:
        return self._arrays[name]


def _reader() -> _XmdfReader:
    """A reader bound to the committed mesh's 25-node, 16-face counts."""
    return _XmdfReader(str(_XMDF), n_node=25, n_face=16, lazy=True)


class TestScalar2dGuard:
    """``_guard_scalar_2d`` refuses the dataset kinds not shipped yet.

    A vector dataset or a layered-3D (rank > 2) dataset is refused rather than flattened.
    The committed sample is a 2D scalar, so these refusals are exercised with a fake group
    carrying the marking attribute -- no fabricated HDF5 bytes, just the guard's decision.
    """

    def test_vector_dataset_is_refused(self):
        """A ``VECTOR`` Grouptype is refused, naming scalar as the supported case."""
        group = _FakeGroup(grouptype="DATASET VECTOR", arrays={"Values": _FakeArray(2)})
        with pytest.raises(FileFormatNotSupportedError, match="vector dataset"):
            _reader()._guard_scalar_2d(group, "velocity")

    def test_layered_3d_dataset_is_refused(self):
        """A rank-3 ``Values`` array (a layered 3D result) is refused, not flattened."""
        group = _FakeGroup(grouptype="DATASET SCALAR", arrays={"Values": _FakeArray(3)})
        with pytest.raises(FileFormatNotSupportedError, match="layered 3D"):
            _reader()._guard_scalar_2d(group, "conc")


class TestCollectRootDataset:
    """``_collect`` handles a DATASET group sitting at the HDF5 root (empty path)."""

    def test_root_level_dataset_is_named_without_crashing(self):
        """A dataset at the root is recorded with a literal fallback name, not a crash.

        A well-formed XMDF nests datasets under containers, but a DATASET group found at
        the root (``path == ()``) must still be named rather than raising ``AttributeError``
        on a non-existent ``_XmdfReader.name``.
        """
        group = _FakeGroup(grouptype="DATASET SCALAR", arrays={"Values": _FakeArray(2)})
        found: list = []
        _reader()._collect(group, (), found)
        assert found == [("dataset", ())]


class TestDuplicateDatasetNames:
    """Two datasets sharing a leaf name are refused, not silently overwritten."""

    def test_duplicate_leaf_names_are_refused(self):
        """Same leaf name under different containers raises, naming both group paths.

        Keyed by leaf name, a plain-dict build would let the second dataset overwrite the
        first and silently lose results; the reader must refuse it (as the DAT driver does).
        """
        found = [("Depth", ("RunA", "Depth")), ("Depth", ("RunB", "Depth"))]
        with pytest.raises(FileFormatNotSupportedError, match="two datasets named"):
            _reader()._guard_unique_names(found)

    def test_unique_leaf_names_pass(self):
        """Distinct leaf names are accepted."""
        _reader()._guard_unique_names(
            [("Depth", ("RunA", "Depth")), ("Vel", ("RunA", "Vel"))]
        )


class TestLocation:
    """``_location`` places a dataset by its per-step element count against the mesh."""

    def test_face_located_when_count_matches_faces(self):
        """A per-step width equal to the face count is placed on faces."""
        assert _reader()._location((3, 16), "x") == "face"

    def test_single_axis_uses_the_sole_length(self):
        """A one-axis (static, single-step) dataset is placed by its only length."""
        assert _reader()._location((25,), "x") == "node"

    def test_mismatch_is_a_value_error(self):
        """A width matching neither nodes nor faces is refused."""
        with pytest.raises(ValueError, match="do not belong to this mesh"):
            _reader()._location((3, 99), "x")


class TestMaybeActive:
    """``_maybe_active`` attaches an active sibling only when it fits the faces."""

    def test_no_active_array_attaches_nothing(self):
        """A dataset without an ``Active`` array yields no sibling."""
        variables: dict = {}
        _reader()._maybe_active(
            _FakeGroup(arrays={}), "d", (), np.array([0.0]), variables
        )
        assert variables == {}

    def test_active_of_the_wrong_width_is_left_off(self):
        """An active mask whose width is not the face count is left off, not mislabelled."""
        group = _FakeGroup(arrays={"Active": _FakeArray(shape=(3, 99))})
        variables: dict = {}
        _reader()._maybe_active(
            group, "d", ("grp",), np.array([0.0, 0.5, 1.0]), variables
        )
        assert variables == {}
