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

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.xdmf import XdmfDriver

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
