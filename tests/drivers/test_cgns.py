"""Tests for the CGNS/HDF5 reader, against a real committed sample.

The fixture is the official CGNS library test grid ``cgnslib_vers-4400.cgns`` committed
under ``tests/data/cgns/`` (zlib licence -- see ``SOURCE.txt``). It is a plain HDF5
container (SIDS tree) read by forcing the vendored GDAL HDF5 multidimensional driver, since
``.cgns`` is not auto-sniffed.

Its single ``Base``/``Zone 1`` is an unstructured zone of 3213 vertices. The zone carries
one volumetric section ``Elem`` (2560 ``HEXA_8`` cells) and three surface sections
``InflowElem``/``OutflowElem``/``SidewallElem`` (128 + 128 + 960 = 1216 ``QUAD_4`` boundary
faces). Surface sections are preferred, so the reader returns the 1216-quad surface skin;
the interior volume nodes it leaves unreferenced are dropped, so the skin keeps 1218 of the
3213 vertices (issue #97). The tests read the committed HDF5 bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest
from pyramids.base._errors import FileFormatNotSupportedError

from pyramids_mesh.drivers.cgns import (
    _PAD,
    CgnsDriver,
    _cgns_corner_count,
    _CgnsReader,
    _drop_orphan_nodes,
    _group_label,
    _pad_sections,
)

_DATA = Path(__file__).parent.parent / "data" / "cgns"
_CGNS = _DATA / "cgnslib_vers-4400.cgns"
_ORPHAN = _DATA / "minimal_quadtri_orphans.cgns"
_MED = _DATA.parent / "med" / "cylinder.med"


def _read():
    """Read the committed CGNS sample, ignoring the degenerate-face winding warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CgnsDriver().read(str(_CGNS))


def _read_orphan():
    """Read the crafted minimal mixed quad+tri fixture (issue #97)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CgnsDriver().read(str(_ORPHAN))


def test_reads_surface_skin_of_the_first_zone():
    """The zone's surface (QUAD_4) sections build a 1216-face mesh of just the skin nodes.

    The zone carries 3213 volume nodes; only the 1218 the surface quads reference are kept
    (issue #97 -- the interior volume nodes are dropped rather than orphaned).
    """
    mesh = _read().mesh
    assert mesh.n_node == 1218
    assert mesh.n_face == 1216


def test_faces_are_quads_zero_based_and_rebased():
    """Faces come back 0-based with the 1-based CGNS numbering recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.data.shape == (1216, 4)
    assert connectivity.original_start_index == 1
    real = connectivity.data[connectivity.data != -1]
    assert int(real.min()) == 0
    assert int(real.max()) == 1217  # 1218 skin nodes kept, 0-based


def test_coordinate_z_is_carried_as_node_z():
    """The 3D grid's CoordinateZ becomes a node_z variable."""
    dataset = _read()
    assert "node_z" in set(dataset.data_variable_names)
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (1218,)


def test_element_types_metadata_is_quad4():
    """Every chosen surface element is a QUAD_4, one metadata entry per face."""
    dataset = _read()
    element_types = dataset.global_attributes["element_types"]
    assert len(element_types) == 1216
    assert set(element_types) == {"QUAD_4"}


def test_minimal_quadtri_reads_mixed_surface_skin():
    """The crafted fixture returns 4 surface faces: 2 quads + 2 tris padded to width 4.

    This is the mixed quad+tri surface path the quad-only cgnslib fixture does not cover.
    """
    dataset = _read_orphan()
    mesh = dataset.mesh
    assert mesh.n_face == 4
    connectivity = np.asarray(mesh.face_node_connectivity.data)
    assert connectivity.shape == (4, 4)
    # each triangle occupies a quad-width row, padded with one -1 fill
    assert int((connectivity == -1).sum()) == 2
    element_types = dataset.global_attributes["element_types"]
    assert sorted(element_types) == ["QUAD_4", "QUAD_4", "TRI_3", "TRI_3"]
    assert "node_z" in set(dataset.data_variable_names)


def test_minimal_quadtri_drops_orphan_interior_nodes_issue_97():
    """Issue #97 fixed: the 4 interior nodes are dropped; the skin carries no orphans.

    The fixture's 12 nodes compact to the 8 cube corners the preferred surface sections
    reference, so every kept node is referenced and the connectivity is re-indexed.
    """
    mesh = _read_orphan().mesh
    connectivity = np.asarray(mesh.face_node_connectivity.data)
    referenced = np.unique(connectivity[connectivity >= 0])
    assert mesh.n_node == 8
    assert referenced.size == mesh.n_node  # zero orphans


def test_sniff_accepts_cgns_and_declines_med_and_junk(tmp_path):
    """can_open accepts the CGNS sample and declines MED bytes, junk and a directory."""
    driver = CgnsDriver()
    assert driver.can_open(str(_CGNS)) is True

    # a genuine HDF5 that is not CGNS (the MED sample), under the .cgns suffix
    foreign = tmp_path / "foreign.cgns"
    foreign.write_bytes(_MED.read_bytes())
    assert driver.can_open(str(foreign)) is False

    junk = tmp_path / "plain.cgns"
    junk.write_bytes(b"\x00 not hdf5 \xff")
    assert driver.can_open(str(junk)) is False

    directory = tmp_path / "d.cgns"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_sniff_declines_hdf5_magic_but_unopenable(tmp_path):
    """A file with the HDF5 magic but an unreadable body is declined, not raised."""
    bad = tmp_path / "bad.cgns"
    bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert CgnsDriver().can_open(str(bad)) is False


def test_declares_read_mesh_only():
    """The driver advertises read_mesh and nothing else."""
    assert set(CgnsDriver.capabilities) == {"read_mesh"}


# --- white-box coverage of the reader's defensive branches, driven by stand-in GDAL
# group/array objects (no fabricated CGNS HDF5; just the decision each branch makes) ---


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
    """A stand-in GDAL MD-array that reads back a fixed value."""

    def __init__(self, value):
        self._value = np.asarray(value)

    def ReadAsArray(self):
        return self._value


class _Grp:
    """A stand-in GDAL group with a SIDS label, named subgroups and a ' data' array."""

    def __init__(self, label="", groups=None, data=None):
        self._label = label
        self._groups = dict(groups or {})
        self._data = data

    def GetAttributes(self):
        return [_Attr("label", self._label)] if self._label else []

    def GetGroupNames(self):
        return list(self._groups)

    def OpenGroup(self, name):
        return self._groups[name]

    def GetMDArrayNames(self):
        return [" data"] if self._data is not None else []

    def OpenMDArray(self, name):
        return _Arr(self._data)


def _reader():
    """A reader bound to a dummy path (the stand-ins carry the data)."""
    return _CgnsReader("fake.cgns")


class TestReaderDefensiveBranches:
    """Each refusal branch of the reader, via stand-in GDAL objects."""

    def test_no_base_is_refused(self):
        """A root with no CGNSBase_t group is refused."""
        root = _Grp(groups={"NotABase": _Grp(label="Descriptor_t")})
        with pytest.raises(FileFormatNotSupportedError, match="no 'CGNSBase_t'"):
            _reader()._first_zone(root)

    def test_base_without_zone_is_refused(self):
        """A base carrying no Zone_t group is refused."""
        base = _Grp(label="CGNSBase_t", groups={})
        root = _Grp(groups={"Base": base})
        with pytest.raises(FileFormatNotSupportedError, match="no 'Zone_t'"):
            _reader()._first_zone(root)

    def test_zone_without_grid_is_refused(self):
        """A zone carrying no GridCoordinates group is refused."""
        zone = _Grp(label="Zone_t", groups={})
        with pytest.raises(FileFormatNotSupportedError, match="no 'GridCoordinates'"):
            _reader()._coordinates(zone)

    def test_zone_without_any_usable_section_is_refused(self):
        """A zone whose only section is MIXED/1D yields no faces and is refused."""
        # a MIXED (code 20) section is skipped, leaving nothing to build faces from
        mixed = _Grp(
            label="Elements_t",
            groups={
                "ElementRange": _Grp(data=[1, 1]),
                "ElementConnectivity": _Grp(data=[20, 0, 1, 2]),
            },
            data=[20, 0],
        )
        zone = _Grp(label="Zone_t", groups={"MixedElem": mixed})
        with pytest.raises(
            FileFormatNotSupportedError, match="no surface or volumetric"
        ):
            _reader()._faces(zone)

    def test_volumetric_only_zone_falls_back_to_cells_as_faces(self):
        """With no surface section, a volumetric (HEXA_8) section maps as faces."""
        hexa = _Grp(
            label="Elements_t",
            groups={
                "ElementRange": _Grp(data=[1, 2]),
                "ElementConnectivity": _Grp(
                    data=[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
                ),
            },
            data=[17, 0],
        )
        zone = _Grp(label="Zone_t", groups={"Cells": hexa})
        faces, element_types = _reader()._faces(zone)
        assert faces.shape == (2, 8)
        assert set(element_types) == {"HEXA_8"}

    def test_surface_sections_are_preferred_over_volumetric(self):
        """A zone with both a hex and a quad section returns the quads only."""
        hexa = _Grp(
            label="Elements_t",
            groups={
                "ElementRange": _Grp(data=[1, 1]),
                "ElementConnectivity": _Grp(data=[1, 2, 3, 4, 5, 6, 7, 8]),
            },
            data=[17, 0],
        )
        quad = _Grp(
            label="Elements_t",
            groups={
                "ElementRange": _Grp(data=[1, 1]),
                "ElementConnectivity": _Grp(data=[1, 2, 3, 4]),
            },
            data=[7, 0],
        )
        zone = _Grp(label="Zone_t", groups={"Cells": hexa, "Skin": quad})
        faces, element_types = _reader()._faces(zone)
        assert faces.shape == (1, 4)
        assert element_types == ["QUAD_4"]

    def test_coordinates_without_coordinate_z_has_no_node_z(self):
        """A 2D grid carrying no CoordinateZ group yields node_z None."""
        grid = _Grp(
            groups={
                "CoordinateX": _Grp(data=[0.0, 1.0, 2.0]),
                "CoordinateY": _Grp(data=[0.0, 1.0, 2.0]),
            }
        )
        zone = _Grp(label="Zone_t", groups={"GridCoordinates": grid})
        node_x, node_y, node_z = _reader()._coordinates(zone)
        assert node_z is None
        assert node_x.tolist() == [0.0, 1.0, 2.0]
        assert node_y.tolist() == [0.0, 1.0, 2.0]

    def test_coordinates_with_all_zero_coordinate_z_is_dropped(self):
        """An all-zero CoordinateZ is a planar mesh, so node_z is dropped."""
        grid = _Grp(
            groups={
                "CoordinateX": _Grp(data=[0.0, 1.0, 2.0]),
                "CoordinateY": _Grp(data=[0.0, 1.0, 2.0]),
                "CoordinateZ": _Grp(data=[0.0, 0.0, 0.0]),
            }
        )
        zone = _Grp(label="Zone_t", groups={"GridCoordinates": grid})
        _, _, node_z = _reader()._coordinates(zone)
        assert node_z is None

    def test_sort_section_skips_an_unrecognised_code(self):
        """A non-MIXED code that is neither a surface nor a volume type is skipped."""
        surface: list = []
        volume: list = []
        section = _Grp(
            label="Elements_t",
            groups={
                "ElementRange": _Grp(data=[1, 1]),
                "ElementConnectivity": _Grp(data=[1, 2]),
            },
            data=[3, 0],  # code 3 (BAR_2): a 1D type, neither surface nor volume
        )
        _reader()._sort_section(section, surface, volume)
        assert surface == []
        assert volume == []

    def test_sort_section_skips_an_empty_range(self):
        """A section whose ElementRange yields no elements is skipped."""
        surface: list = []
        volume: list = []
        section = _Grp(
            label="Elements_t",
            groups={
                "ElementRange": _Grp(data=[2, 1]),  # 1 - 2 + 1 == 0 elements
                "ElementConnectivity": _Grp(data=[1, 2, 3, 4]),
            },
            data=[7, 0],  # QUAD_4
        )
        _reader()._sort_section(section, surface, volume)
        assert surface == []
        assert volume == []


def test_group_label_swallows_a_read_error():
    """A SIDS label attribute whose Read() raises is treated as an absent label."""

    class _RaisingAttr:
        """A stand-in GDAL attribute whose value read fails."""

        def GetName(self):
            return "label"

        def Read(self):
            raise RuntimeError("unreadable attribute")

    class _Group:
        """A stand-in group carrying a single raising label attribute."""

        def GetAttributes(self):
            return [_RaisingAttr()]

    assert _group_label(_Group()) == ""


def test_pad_sections_pads_triangles_to_the_widest_block():
    """A triangle block beside a quad block pads the triangle rows with the 0 pad."""
    tris = np.array([[1, 2, 3]])
    quads = np.array([[2, 3, 4, 5]])
    faces, types = _pad_sections([(tris, "TRI_3"), (quads, "QUAD_4")])
    assert faces.tolist() == [[1, 2, 3, 0], [2, 3, 4, 5]]
    assert types == ["TRI_3", "QUAD_4"]


def test_higher_order_surface_codes_keep_only_corner_nodes():
    """A higher-order CGNS surface element contributes only its corner nodes to a face."""
    assert _cgns_corner_count("TRI_3") == 3
    assert _cgns_corner_count("TRI_6") == 3
    assert _cgns_corner_count("QUAD_4") == 4
    assert _cgns_corner_count("QUAD_8") == 4
    assert _cgns_corner_count("QUAD_9") == 4


def test_cgns_example_notebook_commits_no_machine_specific_path():
    """The committed cgns example notebook must not embed an absolute local path.

    Re-executing the notebook bakes the executing checkout's absolute path into the warning
    stderr (e.g. a worktree ``src`` path or a pixi ``site-packages`` path); such a path is
    machine-specific, meaningless on the rendered docs site, and leaks the local layout. It
    must be scrubbed to a relative module path before commit (PR review SF1).
    """
    notebook = Path(__file__).parents[2] / "docs" / "examples" / "cgns.ipynb"
    text = notebook.read_text(encoding="utf-8")
    for needle in ("worktrees", "python-environments", "site-packages"):
        assert needle not in text, f"notebook leaks a machine-specific path token: {needle!r}"


class TestDropOrphanNodes:
    """Unit tests for _drop_orphan_nodes, the issue #97 node compaction."""

    def test_drops_unreferenced_nodes_and_keeps_contiguous_ids(self):
        """Interior nodes no face references are removed; a leading id block is unchanged.

        Test scenario:
            6 nodes with one quad referencing 1-based ids {1,2,3,4}; nodes 5-6 are interior
            volume-only nodes. They are dropped, coordinates subset to the four corners, and
            the connectivity is unchanged because the kept ids are already 1..4.
        """
        node_x = np.array([0.0, 1.0, 1.0, 0.0, 9.0, 9.0])
        node_y = np.array([0.0, 0.0, 1.0, 1.0, 9.0, 9.0])
        node_z = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
        faces = np.array([[1, 2, 3, 4]], dtype=np.int64)
        out_x, out_y, out_z, out_faces = _drop_orphan_nodes(node_x, node_y, node_z, faces)
        assert out_x.tolist() == [0.0, 1.0, 1.0, 0.0], f"x not subset: {out_x}"
        assert out_y.tolist() == [0.0, 0.0, 1.0, 1.0], f"y not subset: {out_y}"
        assert out_z.tolist() == [0.0, 0.0, 0.0, 0.0], f"z not subset: {out_z}"
        assert out_faces.tolist() == [[1, 2, 3, 4]], f"faces changed: {out_faces}"

    def test_reindexes_non_contiguous_referenced_ids(self):
        """Gaps in the referenced ids compact to a dense 1-based range, with node_z None.

        Test scenario:
            4 nodes, a padded triangle referencing ids {1, 3, 4}; node 2 is orphaned. The
            kept ids 1,3,4 remap to 1,2,3 and the connectivity follows; ``node_z`` stays None.
        """
        node_x = np.array([10.0, 20.0, 30.0, 40.0])
        node_y = np.zeros(4)
        faces = np.array([[1, 3, 4, _PAD]], dtype=np.int64)
        out_x, out_y, out_z, out_faces = _drop_orphan_nodes(node_x, node_y, None, faces)
        assert out_x.tolist() == [10.0, 30.0, 40.0], f"node 2 not dropped: {out_x}"
        assert out_z is None, f"node_z should stay None, got {out_z!r}"
        assert out_faces.tolist() == [[1, 2, 3, _PAD]], f"ids not remapped: {out_faces}"

    def test_noop_when_every_node_is_referenced(self):
        """A mesh whose faces span all nodes is returned unchanged, as the same objects.

        Test scenario:
            4 nodes, one quad referencing all of them -> no orphans -> identity return, so a
            2D surface or the volumetric fallback pays no copy cost.
        """
        node_x = np.arange(4.0)
        node_y = np.arange(4.0)
        node_z = np.arange(4.0)
        faces = np.array([[1, 2, 3, 4]], dtype=np.int64)
        out_x, out_y, out_z, out_faces = _drop_orphan_nodes(node_x, node_y, node_z, faces)
        assert out_x is node_x, "x array should be returned unchanged"
        assert out_y is node_y, "y array should be returned unchanged"
        assert out_z is node_z, "z array should be returned unchanged"
        assert out_faces is faces, "faces should be returned unchanged"

    def test_pad_fill_is_left_untouched(self):
        """The _PAD fill in short rows is preserved while real ids are remapped.

        Test scenario:
            A quad and a padded triangle; node 5 is orphaned. After compaction the pad slot
            stays _PAD rather than being rewritten as a remapped id.
        """
        node_x = np.array([0.0, 1.0, 2.0, 3.0, 99.0])
        node_y = np.zeros(5)
        faces = np.array([[1, 2, 3, 4], [1, 2, 3, _PAD]], dtype=np.int64)
        out_x, _, _, out_faces = _drop_orphan_nodes(node_x, node_y, None, faces)
        assert out_x.tolist() == [0.0, 1.0, 2.0, 3.0], f"orphan not dropped: {out_x}"
        assert out_faces.tolist() == [[1, 2, 3, 4], [1, 2, 3, _PAD]], (
            f"pad fill not preserved: {out_faces}"
        )
