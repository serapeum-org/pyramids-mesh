"""Tests for the XDMF reader (D7), against a real committed sample.

XDMF is an XML index over an HDF5 payload. The fixture under ``tests/data/xdmf/`` is a
genuine meshio ``XdmfTimeSeriesWriter`` output (``out.xdmf`` + ``out.h5``): 4 nodes, 2
triangles, 5 timesteps, with a node scalar ``phi``, a node vector ``u`` and a cell scalar
``a``. The tests read the committed bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

import xml.etree.ElementTree as ET

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.xdmf import (
    XdmfDriver,
    _XdmfReader,
    _read_md_array,
    _reshape_by_dimensions,
    _resolve_reference,
)

_XDMF = Path(__file__).parent.parent / "data" / "xdmf" / "out.xdmf"


def _read():
    """Read the committed XDMF sample, ignoring the expected winding-fix warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return XdmfDriver().read(str(_XDMF))


def test_reads_topology_from_geometry_and_topology():
    """The mesh Grid's Geometry and Topology build a 4-node, 2-triangle mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 4
    assert mesh.n_face == 2
    assert mesh.face_node_connectivity.data.shape == (2, 3)


def test_connectivity_is_zero_based():
    """XDMF connectivity is already 0-based; the base is recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.original_start_index == 0
    assert int(connectivity.data.min()) == 0
    assert int(connectivity.data.max()) == 3


def test_faces_are_counter_clockwise():
    """Every face comes back counter-clockwise after validation."""
    mesh = _read().mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 > 0


def test_node_scalar_is_node_located_over_time():
    """The node scalar ``phi`` is a node-located variable over the five timesteps."""
    phi = _read().get("phi")
    assert phi.location == "node"
    assert phi.shape == (5, 4)
    assert phi.attributes["time_values"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_cell_scalar_is_face_located_over_time():
    """The cell scalar ``a`` is a face-located variable over the five timesteps."""
    a = _read().get("a")
    assert a.location == "face"
    assert a.shape == (5, 2)
    # first step of the committed sample
    assert a.data[0].tolist() == pytest.approx([3.0, 4.2])


def test_vector_attribute_is_skipped():
    """The vector attribute ``u`` is left out (2D scalar ships first), not mislabelled."""
    names = list(_read().data_variable_names)
    assert "phi" in names
    assert "a" in names
    assert "u" not in names


def test_values_are_lazy():
    """A lazy read reports a source without loading the values."""
    phi = _read().get("phi")
    assert phi.has_data_source
    assert phi._data is None
    assert phi.data.shape == (5, 4)


def test_loader_survives_repeated_reads():
    """The lazy loader reopens the HDF5 payload each call, so repeated reads all succeed."""
    phi = _read().get("phi")
    first = phi.data
    second = phi.data
    assert np.array_equal(first, second)


def test_sniff_declines_non_xdmf(tmp_path):
    """can_open declines a non-XDMF file, junk and a directory without raising."""
    driver = XdmfDriver()
    assert driver.can_open(str(_XDMF)) is True

    not_xdmf = tmp_path / "other.xdmf"
    not_xdmf.write_text("<vtkfile>not xdmf</vtkfile>\n", encoding="utf-8")
    assert driver.can_open(str(not_xdmf)) is False

    empty = tmp_path / "empty.xdmf"
    empty.write_bytes(b"")
    assert driver.can_open(str(empty)) is False

    directory = tmp_path / "d.xdmf"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_unsupported_topology_is_refused(tmp_path):
    """A volumetric (3D) topology is refused rather than flattened."""
    xdmf = tmp_path / "vol.xdmf"
    xdmf.write_text(
        '<Xdmf Version="3.0"><Domain>'
        '<Grid Name="mesh" GridType="Uniform">'
        '<Geometry GeometryType="XYZ">'
        '<DataItem Dimensions="4 3" Format="HDF">out.h5:/data0</DataItem></Geometry>'
        '<Topology TopologyType="Tetrahedron" NumberOfElements="1">'
        '<DataItem Dimensions="1 4" Format="HDF">out.h5:/data1</DataItem></Topology>'
        "</Grid></Domain></Xdmf>",
        encoding="utf-8",
    )
    with pytest.raises(FileFormatNotSupportedError, match="TopologyType"):
        XdmfDriver().read(str(xdmf))


def test_resolve_reference_without_a_file_prefix():
    """A reference with no ``file:`` prefix resolves to a path with an empty dataset name."""
    path, dataset = _resolve_reference("sibling.h5", "/base")
    assert path.replace("\\", "/").endswith("/base/sibling.h5")
    assert dataset == ""


def test_read_md_array_walks_nested_group_names():
    """A group-nested dataset name is walked group by group before the array is read."""

    class _Array:
        def ReadAsArray(self):
            return np.arange(3)

    class _Group:
        def OpenGroup(self, name):
            return self

        def OpenMDArray(self, name):
            return _Array()

    out = _read_md_array(_Group(), "grp/sub/data")
    assert out.tolist() == [0, 1, 2]


def test_mesh_grid_without_geometry_and_topology_is_refused():
    """A document whose grids carry no Geometry+Topology pair is refused."""
    reader = _XdmfReader(str(_XDMF), lazy=True)
    with pytest.raises(FileFormatNotSupportedError, match="no grid carries both"):
        reader._mesh_grid([])


def test_geometry_xyz_sets_node_z():
    """An ``XYZ`` geometry contributes a ``node_z`` from the third coordinate column."""
    reader = _XdmfReader(str(_XDMF), lazy=True)
    reader._read = lambda data_item: np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 3.0], [0.0, 1.0, 4.0]]
    )
    grid = ET.fromstring(
        '<Grid><Geometry GeometryType="XYZ">'
        "<DataItem>x.h5:/d</DataItem></Geometry></Grid>"
    )
    node_x, node_y, node_z = reader._geometry(grid)
    assert node_z is not None
    assert node_z.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_variables_without_a_collection_is_empty():
    """With no temporal ``Collection`` grid, there are no result variables."""
    reader = _XdmfReader(str(_XDMF), lazy=True)
    assert reader._variables([], n_node=4, n_face=2) == {}


class TestReshapeByDimensions:
    """``_reshape_by_dimensions`` honours the XDMF ``Dimensions``, not the stored rank.

    The flat-1-D-stored branch is covered only by these synthetic unit tests: the committed
    ``out.h5`` stores geometry and topology already 2-D (matching their ``Dimensions``), so
    the full-``read`` integration path exercises the 2-D branch, not the flat one. No real
    flat-stored XDMF sample is available to commit (real-sample-only rule).
    """

    def test_reshapes_a_flat_array_to_the_declared_dimensions(self):
        """A flat HDF5 array is reshaped to the authoritative 2D ``Dimensions``."""
        out = _reshape_by_dimensions(np.arange(8), (4, 2))
        assert out.shape == (4, 2)
        assert out.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]

    def test_flat_connectivity_reshapes_by_dimensions(self):
        """A flat 1D connectivity array becomes (n_face, nodes_per_face) via ``Dimensions``."""
        out = _reshape_by_dimensions(np.arange(6), (2, 3))
        assert out.shape == (2, 3)

    def test_falls_back_to_native_2d_shape_without_dimensions(self):
        """With no usable ``Dimensions`` a 2D-stored array is returned unchanged."""
        arr = np.arange(8).reshape(4, 2)
        assert _reshape_by_dimensions(arr, ()).shape == (4, 2)
