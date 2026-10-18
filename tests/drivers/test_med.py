"""Tests for the MED-file (Salome / code_aster) reader, against a real committed sample.

The fixture is ``cylinder.med`` committed under ``tests/data/med/`` (meshio test data, MIT
-- see ``SOURCE.txt``), a Salome 9.2.2 mesh. It is a plain HDF5 container read by forcing
the vendored GDAL HDF5 multidimensional driver, since ``.med`` is not auto-sniffed.

Its single mesh ``Cylinder`` has 33 nodes (``NOE/COO`` is coordinate-major and 3D). The
mesh carries surface groups ``MAI/TR3`` (4 triangles) and ``MAI/QU4`` (18 quads), segment
group ``MAI/SE2`` (skipped, 1D) and volumetric groups ``MAI/TE4``/``MAI/PY5`` (skipped,
surface is preferred). The reader returns the 22-face surface. The tests read the committed
HDF5 bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.med import (
    MedDriver,
    _array_count,
    _MedReader,
    _med_corner_count,
    _pad_groups,
)

_DATA = Path(__file__).parent.parent / "data" / "med"
_MED = _DATA / "cylinder.med"
_CGNS = _DATA.parent / "cgns" / "cgnslib_vers-4400.cgns"


def _read():
    """Read the committed MED sample, ignoring the degenerate-face winding warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MedDriver().read(str(_MED))


def test_reads_surface_of_the_first_mesh():
    """The mesh's surface (TR3 + QU4) groups build a 33-node, 22-face mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 33
    assert mesh.n_face == 22  # 4 triangles + 18 quads


def test_faces_are_zero_based_rebased_and_padded():
    """Faces come back 0-based with the triangles -1-padded to the quad width."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.data.shape == (22, 4)
    assert connectivity.original_start_index == 1
    real = connectivity.data[connectivity.data != -1]
    assert int(real.min()) == 0
    assert int(real.max()) <= 32  # 33 nodes, 0-based
    # the triangle rows are shorter than the quads, so some -1 padding is present
    assert bool((connectivity.data == -1).any())


def test_coordinate_z_is_carried_as_node_z():
    """The 3D cylinder's z column becomes a node_z variable."""
    dataset = _read()
    assert "node_z" in set(dataset.data_variable_names)
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (33,)


def test_node_coordinates_lie_on_the_unit_cylinder():
    """COO is coordinate-major: the leading columns recover x,y on the unit circle."""
    mesh = _read().mesh
    # the first ring of nodes sits on the unit circle x^2 + y^2 == 1
    radii = np.hypot(mesh.node_x[:16], mesh.node_y[:16])
    assert np.allclose(radii, 1.0, atol=1e-6)


def test_element_types_metadata_covers_both_surface_codes():
    """The per-element metadata names TR3 and QU4, one entry per face."""
    element_types = _read().global_attributes["element_types"]
    assert len(element_types) == 22
    assert set(element_types) == {"TR3", "QU4"}


def test_sniff_accepts_med_and_declines_cgns_and_junk(tmp_path):
    """can_open accepts the MED sample and declines CGNS bytes, junk and a directory."""
    driver = MedDriver()
    assert driver.can_open(str(_MED)) is True

    # a genuine HDF5 that is not MED (the CGNS sample), under the .med suffix
    foreign = tmp_path / "foreign.med"
    foreign.write_bytes(_CGNS.read_bytes())
    assert driver.can_open(str(foreign)) is False

    junk = tmp_path / "plain.med"
    junk.write_bytes(b"\x00 not hdf5 \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "d.med"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_sniff_declines_hdf5_magic_but_unopenable(tmp_path):
    """A file with the HDF5 magic but an unreadable body is declined, not raised."""
    bad = tmp_path / "bad.med"
    bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert MedDriver().can_open(str(bad)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else."""
    assert set(MedDriver.capabilities) == {"read_mesh"}


# --- white-box coverage of the reader's defensive branches, driven by stand-in GDAL
# group/array objects (no fabricated MED HDF5; just the decision each branch makes) ---


class _Attr:
    """A stand-in GDAL attribute reading back a fixed name/value."""

    def __init__(self, name, value):
        self._name = name
        self._value = value

    def GetName(self):
        return self._name

    def Read(self):
        return self._value


class _Arr:
    """A stand-in GDAL MD-array with an NBR count attribute and a flat value."""

    def __init__(self, value, nbr):
        self._value = np.asarray(value)
        self._nbr = nbr

    def GetAttributes(self):
        return [_Attr("NBR", self._nbr)]

    def ReadAsArray(self):
        return self._value


class _Grp:
    """A stand-in GDAL group exposing named subgroups and named arrays."""

    def __init__(self, groups=None, arrays=None):
        self._groups = dict(groups or {})
        self._arrays = dict(arrays or {})

    def GetGroupNames(self):
        return list(self._groups)

    def OpenGroup(self, name):
        return self._groups[name]

    def GetMDArrayNames(self):
        return list(self._arrays)

    def OpenMDArray(self, name):
        return self._arrays[name]


def _reader():
    """A reader bound to a dummy path (the stand-ins carry the data)."""
    return _MedReader("fake.med")


class TestReaderDefensiveBranches:
    """Each refusal / fallback branch of the reader, via stand-in GDAL objects."""

    def test_no_meshes_group_is_refused(self):
        """A root with no ENS_MAA group is refused."""
        with pytest.raises(FileFormatNotSupportedError, match="no 'ENS_MAA'"):
            _reader()._first_step(_Grp(groups={}))

    def test_empty_meshes_group_is_refused(self):
        """An ENS_MAA group with no mesh under it is refused."""
        root = _Grp(groups={"ENS_MAA": _Grp(groups={})})
        with pytest.raises(FileFormatNotSupportedError, match="carries no mesh"):
            _reader()._first_step(root)

    def test_mesh_without_a_node_bearing_step_is_refused(self):
        """A mesh whose steps carry no NOE node group is refused."""
        mesh = _Grp(groups={"step": _Grp(groups={"MAI": _Grp()})})
        root = _Grp(groups={"ENS_MAA": _Grp(groups={"Cyl": mesh})})
        with pytest.raises(FileFormatNotSupportedError, match="no step with a 'NOE'"):
            _reader()._first_step(root)

    def test_step_without_mai_group_is_refused(self):
        """A step carrying no MAI element group is refused."""
        with pytest.raises(FileFormatNotSupportedError, match="no 'MAI'"):
            _reader()._faces(_Grp(groups={}))

    def test_segment_only_mesh_is_refused(self):
        """A mesh whose only MAI group is segments (SE2) yields no faces and is refused."""
        mai = _Grp(groups={"SE2": _Grp(arrays={"NOD": _Arr([1, 2, 2, 3], 2)})})
        step = _Grp(groups={"MAI": mai})
        with pytest.raises(
            FileFormatNotSupportedError, match="no surface or volumetric"
        ):
            _reader()._faces(step)

    def test_volumetric_only_mesh_falls_back_to_cells_as_faces(self):
        """With no surface group, a volumetric (TE4) group maps as faces."""
        # F-order flat for two tetrahedra, 4 nodes each
        mai = _Grp(
            groups={"TE4": _Grp(arrays={"NOD": _Arr([1, 5, 2, 6, 3, 7, 4, 8], 2)})}
        )
        step = _Grp(groups={"MAI": mai})
        faces, codes = _reader()._faces(step)
        assert faces.shape == (2, 4)
        assert set(codes) == {"TE4"}

    def test_surface_groups_are_preferred_over_volumetric(self):
        """A mesh with both a TE4 and a QU4 group returns the quads only."""
        mai = _Grp(
            groups={
                "TE4": _Grp(arrays={"NOD": _Arr([1, 2, 3, 4], 1)}),
                "QU4": _Grp(arrays={"NOD": _Arr([1, 2, 3, 4], 1)}),
            }
        )
        step = _Grp(groups={"MAI": mai})
        faces, codes = _reader()._faces(step)
        assert faces.shape == (1, 4)
        assert codes == ["QU4"]

    def test_coordinates_of_a_2d_mesh_have_no_node_z(self):
        """A 2D COO block (dim < 3) yields node_z None."""
        # coordinate-major: x = [0, 1, 2], y = [3, 4, 5]
        coo = _Arr([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3)
        step = _Grp(groups={"NOE": _Grp(arrays={"COO": coo})})
        node_x, node_y, node_z = _reader()._coordinates(step)
        assert node_z is None
        assert node_x.tolist() == [0.0, 1.0, 2.0]
        assert node_y.tolist() == [3.0, 4.0, 5.0]

    def test_coordinates_with_all_zero_z_is_dropped(self):
        """A 3D COO block whose z column is all zero drops node_z."""
        # coordinate-major: x = [0, 1], y = [3, 4], z = [0, 0]
        coo = _Arr([0.0, 1.0, 3.0, 4.0, 0.0, 0.0], 2)
        step = _Grp(groups={"NOE": _Grp(arrays={"COO": coo})})
        _, _, node_z = _reader()._coordinates(step)
        assert node_z is None

    def test_sort_group_skips_an_empty_group(self):
        """A recognised group whose NOD carries no elements is skipped."""
        surface: list = []
        volume: list = []
        mai = _Grp(groups={"QU4": _Grp(arrays={"NOD": _Arr([], 0)})})
        _reader()._sort_group(mai, "QU4", surface, volume)
        assert surface == []
        assert volume == []


def test_array_count_without_an_nbr_attribute_is_refused():
    """An array carrying no NBR attribute cannot report a count and is refused."""

    class _NoCount:
        """A stand-in array exposing only a non-NBR attribute."""

        def GetAttributes(self):
            return [_Attr("FOO", 1)]

    with pytest.raises(FileFormatNotSupportedError, match="no 'NBR'"):
        _array_count(_NoCount())


def test_pad_groups_pads_triangles_to_the_widest_block():
    """A triangle block beside a quad block pads the triangle rows with the 0 pad."""
    tris = np.array([[1, 2, 3]])
    quads = np.array([[2, 3, 4, 5]])
    faces, codes = _pad_groups([(tris, "TR3"), (quads, "QU4")])
    assert faces.tolist() == [[1, 2, 3, 0], [2, 3, 4, 5]]
    assert codes == ["TR3", "QU4"]


def test_higher_order_surface_codes_keep_only_corner_nodes():
    """A higher-order MED surface element contributes only its corner nodes to a face."""
    assert _med_corner_count("TR3") == 3
    assert _med_corner_count("TR6") == 3
    assert _med_corner_count("TR7") == 3
    assert _med_corner_count("QU4") == 4
    assert _med_corner_count("QU8") == 4
    assert _med_corner_count("QU9") == 4
