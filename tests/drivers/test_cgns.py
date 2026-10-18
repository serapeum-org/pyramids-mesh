"""Tests for the CGNS/HDF5 reader, against a real committed sample.

The fixture is the official CGNS library test grid ``cgnslib_vers-4400.cgns`` committed
under ``tests/data/cgns/`` (zlib licence -- see ``SOURCE.txt``). It is a plain HDF5
container (SIDS tree) read by forcing the vendored GDAL HDF5 multidimensional driver, since
``.cgns`` is not auto-sniffed.

Its single ``Base``/``Zone 1`` is an unstructured zone of 3213 vertices. The zone carries
one volumetric section ``Elem`` (2560 ``HEXA_8`` cells) and three surface sections
``InflowElem``/``OutflowElem``/``SidewallElem`` (128 + 128 + 960 = 1216 ``QUAD_4`` boundary
faces). Surface sections are preferred, so the reader returns the 1216-quad surface skin.
The tests read the committed HDF5 bytes.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.cgns import (
    CgnsDriver,
    _CgnsReader,
    _cgns_corner_count,
    _pad_sections,
)

_DATA = Path(__file__).parent.parent / "data" / "cgns"
_CGNS = _DATA / "cgnslib_vers-4400.cgns"
_MED = _DATA.parent / "med" / "cylinder.med"


def _read():
    """Read the committed CGNS sample, ignoring the degenerate-face winding warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CgnsDriver().read(str(_CGNS))


def test_reads_surface_skin_of_the_first_zone():
    """The zone's surface (QUAD_4) sections build a 3213-node, 1216-face mesh."""
    mesh = _read().mesh
    assert mesh.n_node == 3213
    assert mesh.n_face == 1216


def test_faces_are_quads_zero_based_and_rebased():
    """Faces come back 0-based with the 1-based CGNS numbering recorded for a writer."""
    connectivity = _read().mesh.face_node_connectivity
    assert connectivity.data.shape == (1216, 4)
    assert connectivity.original_start_index == 1
    real = connectivity.data[connectivity.data != -1]
    assert int(real.min()) == 0
    assert int(real.max()) == 3212  # 3213 nodes, 0-based


def test_coordinate_z_is_carried_as_node_z():
    """The 3D grid's CoordinateZ becomes a node_z variable."""
    dataset = _read()
    assert "node_z" in set(dataset.data_variable_names)
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (3213,)


def test_element_types_metadata_is_quad4():
    """Every chosen surface element is a QUAD_4, one metadata entry per face."""
    dataset = _read()
    element_types = dataset.global_attributes["element_types"]
    assert len(element_types) == 1216
    assert set(element_types) == {"QUAD_4"}


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
