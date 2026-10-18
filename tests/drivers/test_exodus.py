"""Tests for the Exodus II reader (C10), against real committed samples.

The fixtures are genuine Exodus II meshes committed under ``tests/data/exodus/`` (LGPL-2.1,
libMesh test meshes): ``nontrivial_node_num_map.exo`` (a 9-node, 4-element ``QUAD4`` mesh
whose node numbering map is non-trivial) and ``geom_2.exo`` (a 45-node, 96-element
``TETRA`` mesh). Exodus is NetCDF (CDF-2) underneath and is read through the vendored GDAL
multidimensional API, with no declared ``netCDF4`` dependency. The tests read the committed
bytes -- a real file is a better oracle than one fabricated through GDAL and reopened in the
same process, which is unreliable on this platform.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.exodus import ExodusDriver, _pad_blocks

_DATA = Path(__file__).parent.parent / "data" / "exodus"
_QUAD = _DATA / "nontrivial_node_num_map.exo"
_TETRA = _DATA / "geom_2.exo"
# A real NetCDF file that is not Exodus, reused to prove the content sniff declines it.
_NON_EXODUS_NC = _DATA.parent / "3di" / "results_3di.nc"


def _read(path):
    """Read a committed Exodus file, silencing the winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExodusDriver().read(str(path))


def _signed_area(face, node_x, node_y):
    """Twice the signed polygon area of a face's real (non-fill) vertices; CCW is > 0."""
    vertices = face[face != -1]
    xs = node_x[vertices]
    ys = node_y[vertices]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def test_reads_nodes_and_quad_faces():
    """The QUAD4 file builds a 9-node, 4-quad mesh with the expected coordinates."""
    mesh = _read(_QUAD).mesh
    assert mesh.n_node == 9
    assert mesh.n_face == 4
    assert mesh.face_node_connectivity.data.shape == (4, 4)
    assert float(mesh.node_x[0]) == pytest.approx(0.5)
    assert float(mesh.node_y[0]) == pytest.approx(-0.5)


def test_connectivity_is_one_based_resolved_to_zero_based():
    """The 1-based ``connect1`` table arrives 0-based, with the base recorded for a writer."""
    parsed = ExodusDriver()._parse(str(_QUAD), lazy=False)
    # As parsed, the table is still the file's own 1-based ids.
    assert int(parsed.faces.min()) == 1
    assert parsed.start_index == 1

    connectivity = _read(_QUAD).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert connectivity.data.tolist() == [
        [0, 1, 2, 3],
        [1, 4, 5, 2],
        [3, 2, 6, 7],
        [2, 5, 8, 6],
    ]


def test_quad_faces_are_counter_clockwise():
    """Every built face is wound counter-clockwise, the invariant the normaliser enforces."""
    mesh = _read(_QUAD).mesh
    node_x = np.asarray(mesh.node_x)
    node_y = np.asarray(mesh.node_y)
    for face in mesh.face_node_connectivity.data:
        assert _signed_area(face, node_x, node_y) > 0.0


def test_block_metadata_is_carried():
    """``eb_prop1`` and the per-element ``elem_type`` ride along as global attributes."""
    attributes = _read(_QUAD).global_attributes
    assert attributes["block_ids"] == [1]
    assert attributes["element_types"] == ["QUAD4"] * 4


def test_two_dimensional_file_has_no_node_z():
    """A 2D (``num_dim == 2``) file carries no ``coordz``, so there is no ``node_z``."""
    dataset = _read(_QUAD)
    assert ExodusDriver()._parse(str(_QUAD), lazy=False).node_z is None
    assert "node_z" not in set(dataset.data_variable_names)


def test_mixed_width_blocks_are_fill_padded():
    """A triangle block beside a quad block pads the triangles, 1-based ids preserved."""
    tris = np.array([[1, 2, 3], [3, 4, 5]])
    quads = np.array([[2, 3, 4, 5]])
    faces, element_types = _pad_blocks([(tris, "TRI3"), (quads, "QUAD4")])
    assert faces.tolist() == [[1, 2, 3, 0], [3, 4, 5, 0], [2, 3, 4, 5]]
    assert element_types == ["TRI3", "TRI3", "QUAD4"]


def test_tetra_file_element_node_lists():
    """The TETRA file reads its 45 nodes and 96 four-node element lists, with node_z."""
    parsed = ExodusDriver()._parse(str(_TETRA), lazy=False)
    assert len(parsed.node_x) == 45
    assert parsed.node_z is not None  # 3D file: coordz becomes node_z
    assert parsed.faces.shape == (96, 4)
    # The first element's 1-based node list, verbatim from connect1.
    assert parsed.faces[0].tolist() == [1, 25, 13, 24]
    assert parsed.global_attributes["element_types"] == ["TETRA"] * 96

    mesh = _read(_TETRA).mesh
    assert mesh.n_node == 45
    assert mesh.n_face == 96
    # 1-based [1, 25, 13, 24] drops to 0-based; this element is already CCW in xy.
    assert mesh.face_node_connectivity.data[0].tolist() == [0, 24, 12, 23]


def test_can_open_accepts_exodus_and_declines_others(tmp_path):
    """``can_open`` accepts the Exodus files and declines a plain NetCDF, junk and a dir."""
    driver = ExodusDriver()
    assert driver.can_open(str(_QUAD)) is True
    assert driver.can_open(str(_TETRA)) is True

    # A real NetCDF that is not Exodus: it opens through GDAL but lacks coordx/connect1,
    # so the content sniff declines it. Copied under a claimed suffix so the sniff runs.
    non_exodus = tmp_path / "plain.exo"
    non_exodus.write_bytes(_NON_EXODUS_NC.read_bytes())
    assert driver.can_open(str(non_exodus)) is False

    junk = tmp_path / "junk.exo"
    junk.write_text("not a netcdf file at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "folder.exo"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_read_dispatches_to_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a ``UgridDataset``."""
    assert isinstance(_read(_QUAD), UgridDataset)


def test_declares_read_mesh_only():
    """The driver advertises exactly the ``read_mesh`` capability."""
    assert set(ExodusDriver.capabilities) == {"read_mesh"}
