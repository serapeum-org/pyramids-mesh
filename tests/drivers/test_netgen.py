"""Tests for the Netgen ``.vol`` reader (C12), against the real committed sample.

One fixture lives under ``tests/data/netgen/`` (see ``SOURCE.txt`` for provenance):
``periodic_3d.vol`` is a NETGEN v6.2 mesh of 65 points and 108 surface triangles (with a
dropped block of 166 tetrahedra). The committed bytes drive the end-to-end reads; the
gzip round trip compresses those same bytes in ``tmp_path``, and the defensive branches
(a short row, a bad count, a truncated section, a non-Netgen file, a sniff that must not
raise) are written as tiny inline ``.vol`` snippets the real sample has no shape for --
the same inline-fixture pattern the VTK and SU2 tests use.
"""

import pyramids  # noqa: F401  # isort:skip

import gzip
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids_mesh.drivers.netgen import NetgenDriver, _iter_records

_DATA = Path(__file__).parent.parent / "data" / "netgen"
_SAMPLE = _DATA / "periodic_3d.vol"


def _read(path):
    """Read a sample, ignoring the expected winding/degeneracy warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NetgenDriver().read(str(path))


def _write(tmp_path, body, name="mesh.vol"):
    """Write an inline ``.vol`` body and return its path as a string."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


_MINIMAL = (
    "mesh3d\n"
    "dimension 3\n"
    "geomtype 0\n"
    "surfaceelements\n"
    "1\n"
    " 1 1 1 0 3 1 2 3\n"
    "points\n"
    "3\n"
    "0.0 0.0 0.0\n"
    "1.0 0.0 0.0\n"
    "0.0 1.0 0.0\n"
)


# -- the committed NETGEN sample -----------------------------------------------------


def test_sample_reads_surface_mesh():
    """The sample reads a 65-node, 108-triangle surface mesh."""
    mesh = _read(_SAMPLE).mesh
    assert mesh.n_node == 65
    assert mesh.n_face == 108
    assert mesh.face_node_connectivity.data.shape == (108, 3)


def test_sample_connectivity_is_rebased_from_one():
    """Netgen ids are 1-based, so the connectivity arrives 0-based with base 1 recorded."""
    connectivity = _read(_SAMPLE).mesh.face_node_connectivity
    assert int(connectivity.data.min()) == 0
    # Only the 56 surface nodes are referenced here; the interior nodes (up to 65) are
    # volume-only, and the volume elements are dropped.
    assert int(connectivity.data.max()) == 55
    assert connectivity.original_start_index == 1


def test_sample_volumeelements_are_dropped():
    """The 166 tetrahedra are not boundary-extracted; only the 108 surface faces remain."""
    assert _read(_SAMPLE).mesh.n_face == 108


def test_sample_faces_are_counter_clockwise():
    """Every built face comes back counter-clockwise after validation."""
    mesh = _read(_SAMPLE).mesh
    for row in mesh.face_node_connectivity.data:
        vertices = row[row != -1]
        xs = mesh.node_x[vertices]
        ys = mesh.node_y[vertices]
        signed_area2 = float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
        assert signed_area2 >= 0


def test_sample_node_z_is_node_located():
    """The node elevation comes back as a node-located ``node_z`` variable over 65 nodes."""
    dataset = _read(_SAMPLE)
    assert list(dataset.data_variable_names) == ["node_z"]
    node_z = dataset.get("node_z")
    assert node_z.location == "node"
    assert node_z.shape == (65,)


def test_sample_records_preamble_metadata():
    """The ``dimension`` and ``geomtype`` preamble settings reach the global attributes."""
    attributes = _read(_SAMPLE).global_attributes
    assert attributes["dimension"] == 3
    assert attributes["geomtype"] == 0
    assert attributes["source"] == "Netgen"


def test_sample_has_no_crs():
    """Netgen carries no CRS, so it is never guessed from the coordinates."""
    assert _read(_SAMPLE).mesh.crs is None


# -- gzip ----------------------------------------------------------------------------


def test_gzipped_sample_reads_the_same(tmp_path):
    """A ``.vol.gz`` file is decompressed and reads to the identical surface mesh."""
    gz_path = tmp_path / "periodic_3d.vol.gz"
    with gzip.open(gz_path, "wb") as handle:
        handle.write(_SAMPLE.read_bytes())
    mesh = _read(gz_path).mesh
    assert mesh.n_node == 65
    assert mesh.n_face == 108


def test_can_open_accepts_the_gzip_suffix(tmp_path):
    """can_open claims a ``.vol.gz`` file and sniffs its decompressed signature."""
    gz_path = tmp_path / "periodic_3d.vol.gz"
    with gzip.open(gz_path, "wb") as handle:
        handle.write(_SAMPLE.read_bytes())
    assert NetgenDriver().can_open(str(gz_path)) is True


# -- dispatch ------------------------------------------------------------------------


def test_can_open_accepts_sample_and_declines_foreign(tmp_path):
    """can_open accepts the real sample and declines junk, a BOM-less binary and a dir."""
    driver = NetgenDriver()
    assert driver.can_open(str(_SAMPLE)) is True

    junk = tmp_path / "junk.vol"
    junk.write_text("not a netgen file at all\n", encoding="utf-8")
    assert driver.can_open(str(junk)) is False

    empty = tmp_path / "empty.vol"
    empty.write_bytes(b"")
    assert driver.can_open(str(empty)) is False

    directory = tmp_path / "d.vol"
    directory.mkdir()
    assert driver.can_open(str(directory)) is False


def test_sniff_never_raises_on_a_fake_gzip(tmp_path):
    """A plain-text file under a ``.vol.gz`` name is declined, not raised, by the sniff."""
    fake = tmp_path / "fake.vol.gz"
    fake.write_bytes(b"mesh3d\nsurfaceelements\n")
    assert NetgenDriver().can_open(str(fake)) is False


def test_sniff_never_raises_on_binary(tmp_path):
    """A binary ``.vol`` is declined rather than raising out of the sniff."""
    binary = tmp_path / "bin.vol"
    binary.write_bytes(bytes(range(256)) * 4)
    assert NetgenDriver().can_open(str(binary)) is False


# -- a minimal valid mesh and the defensive branches ---------------------------------


def test_minimal_inline_mesh_reads(tmp_path):
    """A minimal inline file with the single-line preamble reads one triangle."""
    dataset = _read(_write(tmp_path, _MINIMAL))
    assert dataset.mesh.n_node == 3
    assert dataset.mesh.n_face == 1
    assert dataset.mesh.face_node_connectivity.data.tolist() == [[0, 1, 2]]
    assert dataset.global_attributes["dimension"] == 3


def test_non_netgen_file_is_refused(tmp_path):
    """A file with neither the preamble nor a surfaceelements section is refused."""
    path = _write(tmp_path, "hello world\n1 2 3\n")
    with pytest.raises(FileFormatNotSupportedError, match="not a Netgen file"):
        NetgenDriver().read(path)


def test_file_without_points_is_refused(tmp_path):
    """A file declaring surfaceelements but no points is refused, not built empty."""
    body = "mesh3d\nsurfaceelements\n1\n1 1 1 0 3 1 2 3\n"
    with pytest.raises(FileFormatNotSupportedError, match="no points nodes"):
        NetgenDriver().read(_write(tmp_path, body))


def test_file_without_surfaceelements_is_refused(tmp_path):
    """A file declaring points but no surfaceelements is refused."""
    body = "mesh3d\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
    with pytest.raises(FileFormatNotSupportedError, match="no surfaceelements"):
        NetgenDriver().read(_write(tmp_path, body))


def test_short_surfaceelements_row_is_refused(tmp_path):
    """A surfaceelements row carrying fewer node ids than its ``np`` is refused."""
    body = "mesh3d\nsurfaceelements\n1\n1 1 1 0 3 1 2\n"
    with pytest.raises(FileFormatNotSupportedError, match="short surfaceelements row"):
        NetgenDriver().read(_write(tmp_path, body))


def test_short_points_row_is_refused(tmp_path):
    """A points row short of three coordinates is refused."""
    body = "mesh3d\npoints\n1\n0.0 0.0\n"
    with pytest.raises(FileFormatNotSupportedError, match="short points row"):
        NetgenDriver().read(_write(tmp_path, body))


def test_non_integer_count_is_refused(tmp_path):
    """A section count line that is not an integer is refused, naming the section."""
    body = "mesh3d\nsurfaceelements\nabc\n"
    with pytest.raises(FileFormatNotSupportedError, match="expected an integer"):
        NetgenDriver().read(_write(tmp_path, body))


def test_truncated_section_is_refused(tmp_path):
    """A section declaring more rows than the file holds is refused as truncated."""
    body = "mesh3d\nsurfaceelements\n5\n1 1 1 0 3 1 2 3\n"
    with pytest.raises(FileFormatNotSupportedError, match="ends inside the"):
        NetgenDriver().read(_write(tmp_path, body))


def test_out_of_range_node_id_is_refused(tmp_path):
    """A face referencing a node id beyond the point count is caught by validation."""
    body = (
        "mesh3d\npoints\n3\n0 0 0\n1 0 0\n0 1 0\n"
        "surfaceelements\n1\n1 1 1 0 3 1 2 9\n"
    )
    with pytest.raises(ValueError, match="out of range"):
        NetgenDriver().read(_write(tmp_path, body))


# -- the gzip-aware record iterator --------------------------------------------------


def test_iter_records_skips_blank_and_comment_lines(tmp_path):
    """The record walk drops blank and ``#`` comment lines and splits the rest."""
    path = _write(tmp_path, "# a comment\n\nmesh3d\n 1 2 3 \n")
    assert list(_iter_records(path)) == [("mesh3d", []), ("1", ["2", "3"])]


# -- more defensive branches ---------------------------------------------------------


def test_sniff_stops_at_the_size_limit(tmp_path):
    """A head past the sniff limit with no signature is declined, not scanned whole."""
    filler = "filler data line\n" * 5000
    path = _write(tmp_path, filler, name="big.vol")
    assert NetgenDriver().can_open(path) is False


def test_preamble_setting_at_end_of_file_is_dropped(tmp_path):
    """A dimension keyword with no value and no following line is dropped, not a crash."""
    body = "mesh3d\ndimension\n"
    with pytest.raises(FileFormatNotSupportedError, match="no points"):
        NetgenDriver().read(_write(tmp_path, body))


def test_non_integer_preamble_setting_is_dropped(tmp_path):
    """A non-integer dimension is metadata, so it is dropped rather than failing."""
    body = (
        "mesh3d\n"
        "dimension abc\n"
        "surfaceelements\n"
        "1\n"
        " 1 1 1 0 3 1 2 3\n"
        "points\n"
        "3\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
    )
    dataset = _read(_write(tmp_path, body))
    assert "dimension" not in dataset.global_attributes


def test_surfaceelements_row_missing_leading_columns_is_refused(tmp_path):
    """A surfaceelements row short of even the four leading columns is refused."""
    body = "mesh3d\nsurfaceelements\n1\n1 1 1 0\n"
    with pytest.raises(FileFormatNotSupportedError, match="at least"):
        NetgenDriver().read(_write(tmp_path, body))


def test_non_numeric_point_coordinate_is_refused(tmp_path):
    """A non-numeric points coordinate is a format error naming the section."""
    body = "mesh3d\npoints\n1\nx 0.0 0.0\n"
    with pytest.raises(FileFormatNotSupportedError, match="expected a number"):
        NetgenDriver().read(_write(tmp_path, body))
