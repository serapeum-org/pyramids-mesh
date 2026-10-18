"""Tests for the Tecplot finite-element ASCII reader, against a real committed sample.

The committed fixture ``tests/data/tecplot/quad_zone_space.tec`` (meshio test data, MIT)
is a ``FEBLOCK`` quad zone: ``ET=QUADRILATERAL``, ``N=4``/``E=1``, with a cell-centred
fourth variable and whitespace between the zone keys. The tiny inline ``.tec`` snippets
cover the defensive branches no single committed sample can reach at once -- ``POINT``
versus ``BLOCK`` packing, the legacy ``ET=``/``F=`` keywords, a ``Z`` elevation column and
the volumetric element types read as faces.
"""

import pyramids  # noqa: F401  # isort:skip

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import UgridDataset

from pyramids_mesh.drivers.tecplot import TecplotDriver

_DATA = Path(__file__).parent.parent / "data"
_SAMPLE = _DATA / "tecplot" / "quad_zone_space.tec"
# Real samples of the two formats that share the generic ``.dat`` suffix; can_open must
# decline both so the Tecplot driver never steals them.
_FVCOM_DAT = _DATA / "fvcom" / "chn_grd.dat"
_SMS_DAT = _DATA / "minimal.dat"


def _read(path):
    """Read a Tecplot file, silencing any winding-normalisation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TecplotDriver().read(str(path))


def _signed_area(face, node_x, node_y):
    """Twice the signed polygon area of a face's real (non-fill) vertices; CCW is > 0."""
    vertices = face[face != -1]
    xs = node_x[vertices]
    ys = node_y[vertices]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _write(tmp_path, name, text):
    """Write a tiny inline ``.tec`` snippet and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- the committed FEBLOCK quad sample ---------------------------------------------


def test_reads_nodes_and_quad_face():
    """The sample builds a 4-node, 1-quad mesh with the unit-square coordinates."""
    mesh = _read(_SAMPLE).mesh
    assert mesh.n_node == 4
    assert mesh.n_face == 1
    assert mesh.face_node_connectivity.data.shape == (1, 4)
    assert np.asarray(mesh.node_x).tolist() == [0.0, 1.0, 1.0, 0.0]
    assert np.asarray(mesh.node_y).tolist() == [0.0, 0.0, 1.0, 1.0]


def test_connectivity_is_one_based_resolved_to_zero_based():
    """The 1-based quad arrives 0-based, with the base recorded for a writer."""
    parsed = TecplotDriver()._parse(str(_SAMPLE), lazy=False)
    assert int(parsed.faces.min()) == 1
    assert parsed.start_index == 1

    connectivity = _read(_SAMPLE).mesh.face_node_connectivity
    assert connectivity.original_start_index == 1
    assert connectivity.data.tolist() == [[0, 1, 2, 3]]


def test_face_is_counter_clockwise():
    """The built quad is wound counter-clockwise, the invariant the normaliser enforces."""
    mesh = _read(_SAMPLE).mesh
    node_x = np.asarray(mesh.node_x)
    node_y = np.asarray(mesh.node_y)
    for face in mesh.face_node_connectivity.data:
        assert _signed_area(face, node_x, node_y) > 0.0


def test_cell_centered_and_extra_vars_are_skipped():
    """Only geometry is read: the sample's node and cell data variables are not carried."""
    dataset = _read(_SAMPLE)
    assert list(dataset.data_variable_names) == []
    assert dataset.mesh.crs is None


def test_metadata_and_element_type_are_carried():
    """The legacy ``ET=QUADRILATERAL`` is reported under its canonical modern name."""
    attributes = _read(_SAMPLE).global_attributes
    assert attributes["source"] == "Tecplot FE ASCII"
    assert attributes["element_type"] == "FEQUADRILATERAL"
    assert attributes["data_packing"] == "BLOCK"


def test_read_dispatches_to_a_ugrid_dataset():
    """The read path (parse -> validate -> construct) yields a ``UgridDataset``."""
    assert isinstance(_read(_SAMPLE), UgridDataset)


def test_declares_read_mesh_only():
    """The driver advertises exactly the ``read_mesh`` capability."""
    assert set(TecplotDriver.capabilities) == {"read_mesh"}
    assert TecplotDriver.name == "tecplot"
    assert TecplotDriver.suffixes == (".dat", ".tec")


# -- can_open: signature required, shared ``.dat`` declined -------------------------


def test_can_open_accepts_the_tecplot_sample():
    """can_open accepts the committed ``.tec`` sample."""
    assert TecplotDriver().can_open(str(_SAMPLE)) is True


def test_can_open_declines_fvcom_and_sms_dat():
    """A ``.dat`` suffix is not enough: the FVCOM grid and SMS DAT are both declined."""
    driver = TecplotDriver()
    assert _FVCOM_DAT.is_file() and _SMS_DAT.is_file()
    assert driver.can_open(str(_FVCOM_DAT)) is False
    assert driver.can_open(str(_SMS_DAT)) is False


def test_can_open_accepts_a_tecplot_dat(tmp_path):
    """A Tecplot file under the generic ``.dat`` suffix is accepted on its signature."""
    path = _write(
        tmp_path,
        "mesh.dat",
        'TITLE = "t"\nVARIABLES = X Y\n'
        "ZONE N=3 E=1 ZONETYPE=FETRIANGLE DATAPACKING=POINT\n"
        "0.0 0.0\n1.0 0.0\n0.0 1.0\n1 2 3\n",
    )
    assert TecplotDriver().can_open(str(path)) is True


def test_can_open_declines_junk_and_directory(tmp_path):
    """A non-Tecplot ``.tec`` and a directory are both declined without raising."""
    junk = _write(tmp_path, "junk.tec", "not a tecplot file at all\n1 2 3\n")
    assert TecplotDriver().can_open(str(junk)) is False
    directory = tmp_path / "folder.tec"
    directory.mkdir()
    assert TecplotDriver().can_open(str(directory)) is False


def test_sniff_never_raises_on_binary(tmp_path):
    """``_sniff`` decodes leniently, so binary content answers ``False`` rather than raising."""
    binary = tmp_path / "blob.tec"
    binary.write_bytes(bytes(range(256)) * 4)
    assert TecplotDriver()._sniff(str(binary)) is False


# -- packings and keyword generations ----------------------------------------------


def test_point_packing_triangle(tmp_path):
    """POINT packing reads one row of all variables per node."""
    path = _write(
        tmp_path,
        "point.tec",
        'TITLE = "p"\nVARIABLES = "X" "Y"\n'
        "ZONE N=3 E=1 ZONETYPE=FETRIANGLE DATAPACKING=POINT\n"
        "0.0 0.0\n1.0 0.0\n0.0 1.0\n1 2 3\n",
    )
    mesh = _read(path).mesh
    assert mesh.n_node == 3
    assert mesh.n_face == 1
    assert mesh.face_node_connectivity.data.tolist() == [[0, 1, 2]]
    assert np.asarray(mesh.node_x).tolist() == [0.0, 1.0, 0.0]
    assert np.asarray(mesh.node_y).tolist() == [0.0, 0.0, 1.0]


def test_legacy_et_and_feblock_triangle(tmp_path):
    """The legacy ``ET=TRIANGLE`` / ``F=FEBLOCK`` spelling parses as a block triangle."""
    path = _write(
        tmp_path,
        "legacy.tec",
        'TITLE = "l"\nVARIABLES = X Y\n'
        "ZONE N=3 E=1 ET=TRIANGLE F=FEBLOCK\n"
        "0.0 1.0 0.0\n0.0 0.0 1.0\n1 2 3\n",
    )
    dataset = _read(path)
    mesh = dataset.mesh
    assert mesh.n_node == 3
    assert mesh.face_node_connectivity.data.tolist() == [[0, 1, 2]]
    assert np.asarray(mesh.node_x).tolist() == [0.0, 1.0, 0.0]
    assert dataset.global_attributes["element_type"] == "FETRIANGLE"


def test_z_variable_becomes_node_z(tmp_path):
    """A non-zero ``Z`` column is carried as a node-located ``node_z`` variable."""
    path = _write(
        tmp_path,
        "withz.tec",
        'TITLE = "z"\nVARIABLES = X Y Z\n'
        "ZONE N=3 E=1 ZONETYPE=FETRIANGLE DATAPACKING=POINT\n"
        "0.0 0.0 5.0\n1.0 0.0 6.0\n0.0 1.0 7.0\n1 2 3\n",
    )
    node_z = _read(path).get("node_z")
    assert node_z is not None
    assert node_z.location == "node"
    assert node_z.shape == (3,)
    assert node_z.data.tolist() == [5.0, 6.0, 7.0]


def test_all_zero_z_is_not_carried(tmp_path):
    """An all-zero ``Z`` column is dropped rather than kept as a meaningless elevation."""
    path = _write(
        tmp_path,
        "flatz.tec",
        "VARIABLES = X Y Z\n"
        "ZONE N=3 E=1 ZONETYPE=FETRIANGLE DATAPACKING=POINT\n"
        "0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1 2 3\n",
    )
    assert _read(path).get("node_z") is None


# -- volumetric element types map as faces -----------------------------------------


def test_fetetrahedron_maps_as_a_four_node_face(tmp_path):
    """A ``FETETRAHEDRON`` zone's node list becomes a single 4-node face."""
    path = _write(
        tmp_path,
        "tet.tec",
        "VARIABLES = X Y\n"
        "ZONE N=4 E=1 ZONETYPE=FETETRAHEDRON DATAPACKING=POINT\n"
        "0.0 0.0\n1.0 0.0\n1.0 1.0\n0.0 1.0\n1 2 3 4\n",
    )
    mesh = _read(path).mesh
    assert mesh.n_face == 1
    assert mesh.face_node_connectivity.data.shape == (1, 4)
    assert _read(path).global_attributes["element_type"] == "FETETRAHEDRON"


def test_febrick_maps_as_an_eight_node_face(tmp_path):
    """A ``FEBRICK`` zone's node list becomes a single 8-node face."""
    # Eight nodes placed on a convex octagon in connectivity order, so the 8-node face
    # reads as a simple counter-clockwise polygon rather than a degenerate one.
    import math

    points = [
        (math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)
    ]
    rows = "".join(f"{x} {y}\n" for x, y in points)
    path = _write(
        tmp_path,
        "brick.tec",
        "VARIABLES = X Y\n"
        "ZONE N=8 E=1 ZONETYPE=FEBRICK DATAPACKING=POINT\n"
        f"{rows}1 2 3 4 5 6 7 8\n",
    )
    mesh = _read(path).mesh
    assert mesh.n_face == 1
    assert mesh.face_node_connectivity.data.shape == (1, 8)


# -- malformed inputs ---------------------------------------------------------------


def test_truncated_data_is_a_format_error(tmp_path):
    """A file short of the values its header promises is a clean format error."""
    path = _write(
        tmp_path,
        "short.tec",
        "VARIABLES = X Y\n"
        "ZONE N=4 E=1 ZONETYPE=FEQUADRILATERAL DATAPACKING=POINT\n"
        "0.0 0.0\n1.0 0.0\n1 2 3 4\n",  # only two of four node rows
    )
    with pytest.raises(FileFormatNotSupportedError, match="truncated"):
        TecplotDriver().read(str(path))


def test_unsupported_element_type_is_refused(tmp_path):
    """An element type with no planar face mapping is refused by name."""
    path = _write(
        tmp_path,
        "line.tec",
        "VARIABLES = X Y\n"
        "ZONE N=2 E=1 ZONETYPE=FELINESEG DATAPACKING=POINT\n"
        "0.0 0.0\n1.0 0.0\n1 2\n",
    )
    with pytest.raises(FileFormatNotSupportedError, match="FELINESEG"):
        TecplotDriver().read(str(path))


def test_missing_zone_is_refused(tmp_path):
    """A file with no ZONE header is not a readable Tecplot mesh."""
    path = _write(tmp_path, "nozone.tec", 'TITLE = "t"\nVARIABLES = X Y\n')
    with pytest.raises(FileFormatNotSupportedError, match="no ZONE header"):
        TecplotDriver().read(str(path))


def test_missing_xy_is_refused(tmp_path):
    """A zone naming neither X nor Y has nothing to build a mesh from."""
    path = _write(
        tmp_path,
        "noxy.tec",
        "VARIABLES = A B\n"
        "ZONE N=3 E=1 ZONETYPE=FETRIANGLE DATAPACKING=POINT\n"
        "0.0 0.0\n1.0 0.0\n0.0 1.0\n1 2 3\n",
    )
    with pytest.raises(FileFormatNotSupportedError, match="no X/Y"):
        TecplotDriver().read(str(path))
