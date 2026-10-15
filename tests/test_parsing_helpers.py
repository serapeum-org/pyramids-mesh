"""Tests for the shared text and binary parsing helpers (S4).

Both modules are deliberately free of any pyramids import -- they are pure parsing --
so nothing here needs the mesh model. The Fortran-record fixtures are built byte by
byte, which is also the only way to produce the corrupt cases on purpose.
"""

import numpy as np
import pytest

from pyramids_mesh._binary import (
    detect_endianness,
    iter_fortran_records,
    read_record_floats,
    read_record_ints,
    record_offsets,
)
from pyramids_mesh._text import iter_records, native_id_map


def _record(payload: bytes, endian: str) -> bytes:
    """One Fortran unformatted record: length word, payload, length word."""
    word = np.array([len(payload)], dtype=f"{endian}i4").tobytes()
    return word + payload + word


@pytest.mark.parametrize("endian", ["<", ">"])
def test_detect_endianness_big_and_little(tmp_path, endian):
    """The 80-byte title record identifies the byte order in either orientation."""
    path = tmp_path / "model.slf"
    path.write_bytes(_record(b"T" * 80, endian))

    assert detect_endianness(str(path)) == endian


def test_detect_endianness_rejects_non_selafin(tmp_path):
    """A first record that is not 80 bytes in either order is not this format."""
    path = tmp_path / "other.slf"
    path.write_bytes(_record(b"x" * 12, "<"))

    with pytest.raises(ValueError, match="not this format"):
        detect_endianness(str(path))


def test_fortran_record_length_mismatch_raises(tmp_path):
    """A trailing length word disagreeing with the leading one is always an error."""
    path = tmp_path / "corrupt.slf"
    leading = np.array([8], dtype="<i4").tobytes()
    trailing = np.array([9], dtype="<i4").tobytes()
    path.write_bytes(leading + b"12345678" + trailing)

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="length mismatch"):
            next(iter_fortran_records(handle, "<"))


def test_record_offsets_match_sequential_read(tmp_path):
    """The offset index addresses exactly the payloads a sequential read yields."""
    payloads = [b"T" * 80, b"abcd", b"0123456789"]
    path = tmp_path / "model.slf"
    path.write_bytes(b"".join(_record(p, "<") for p in payloads))

    offsets = record_offsets(str(path), "<")

    assert len(offsets) == len(payloads)
    with open(path, "rb") as handle:
        sequential = list(iter_fortran_records(handle, "<"))
    assert sequential == payloads
    with open(path, "rb") as handle:
        for (offset, length), expected in zip(offsets, payloads):
            handle.seek(offset)
            assert handle.read(length) == expected


def test_record_offsets_handles_big_endian(tmp_path):
    """The index verifies trailing length words in the file's own byte order."""
    payloads = [b"T" * 80, b"wxyz"]
    path = tmp_path / "big.slf"
    path.write_bytes(b"".join(_record(p, ">") for p in payloads))

    offsets = record_offsets(str(path), ">")

    assert [length for _, length in offsets] == [80, 4]


def test_read_record_ints_and_floats(tmp_path):
    """Typed record reads decode the payload and leave the array writable."""
    ints = np.array([3, 4, 1], dtype="<i4")
    floats = np.array([1.5, 2.5], dtype="<f8")
    path = tmp_path / "typed.slf"
    path.write_bytes(_record(ints.tobytes(), "<") + _record(floats.tobytes(), "<"))

    with open(path, "rb") as handle:
        got_ints = read_record_ints(handle, "<", 3)
        got_floats = read_record_floats(handle, "<", 2, double=True)

    np.testing.assert_array_equal(got_ints, [3, 4, 1])
    np.testing.assert_allclose(got_floats, [1.5, 2.5])
    got_ints[0] = 99  # writable, not a read-only buffer view
    assert got_ints[0] == 99


def test_native_id_map_handles_gaps_and_reordering():
    """Native ids map to contiguous indices by order of appearance, gaps and all."""
    assert native_id_map([10, 5, 7]) == {10: 0, 5: 1, 7: 2}
    assert native_id_map([1, 2, 3]) == {1: 0, 2: 1, 3: 2}

    with pytest.raises(ValueError, match="duplicate native id"):
        native_id_map([4, 4])


def test_iter_records_skips_blank_and_comment_lines(tmp_path):
    """Blank and comment lines are dropped; the rest split into keyword and fields."""
    path = tmp_path / "mesh.2dm"
    path.write_text(
        "\n".join(
            [
                "MESH2D",
                "",
                "# a comment",
                "   ",
                "ND 1 0.0 0.0 1.0",
                "  E3T 1 1 2 3 0  ",
                "# trailing comment",
            ]
        ),
        encoding="utf-8",
    )

    records = list(iter_records(str(path)))

    assert records == [
        ("MESH2D", []),
        ("ND", ["1", "0.0", "0.0", "1.0"]),
        ("E3T", ["1", "1", "2", "3", "0"]),
    ]
