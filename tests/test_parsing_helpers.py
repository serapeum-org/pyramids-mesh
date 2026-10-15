"""Tests for the shared text and binary parsing helpers (S4).

Both modules are deliberately free of any pyramids import -- they are pure parsing. That
property is checked statically, by reading their import blocks, because it cannot be
checked by importing them: ``pyramids_mesh/__init__.py`` imports pyramids, so any import
of a submodule pulls it in regardless.

The Fortran-record fixtures are built byte by byte, which is also the only way to
produce the corrupt cases on purpose.
"""

import ast
from pathlib import Path

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


@pytest.mark.parametrize("module", ["_text.py", "_binary.py"])
def test_parsing_helpers_import_nothing_from_pyramids(module):
    """The pure-parsing modules name pyramids in no import, now or later.

    A static read of the import block is the only honest check: importing either module
    executes the package __init__, which imports pyramids, so a runtime assertion would
    pass no matter what these modules did.
    """
    source = (
        Path(__file__).parent.parent / "src" / "pyramids_mesh" / module
    ).read_text(encoding="utf-8")
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    offenders = [name for name in imported if name.split(".")[0] == "pyramids"]
    assert not offenders, f"{module} must stay pyramids-free, but imports {offenders}"


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


def test_detect_endianness_refuses_a_byte_palindromic_probe(tmp_path):
    """A header reading the same both ways cannot identify the order, so it raises.

    Returning the first match would be a guess presented as a detection. The default
    probe (80) is asymmetric, so this only reaches a caller passing its own value.
    """
    path = tmp_path / "palindrome.slf"
    path.write_bytes(b"\x01\x00\x00\x01" + b"x" * 8)

    with pytest.raises(ValueError, match="byte-palindrome"):
        detect_endianness(str(path), expect_first_record=0x01000001)


def test_detect_endianness_rejects_a_file_too_short_for_a_header(tmp_path):
    """A file shorter than one length word cannot carry a Fortran record."""
    path = tmp_path / "stub.slf"
    path.write_bytes(b"\x50\x00")

    with pytest.raises(ValueError, match="too short"):
        detect_endianness(str(path))


def test_truncated_payload_raises(tmp_path):
    """A record declaring more bytes than the file holds is an error, not a short read."""
    path = tmp_path / "short.slf"
    path.write_bytes(np.array([32], dtype="<i4").tobytes() + b"only eight")

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="truncated Fortran record"):
            next(iter_fortran_records(handle, "<"))


def test_missing_trailing_length_word_raises(tmp_path):
    """A record whose trailing length word was never written is rejected."""
    path = tmp_path / "notail.slf"
    path.write_bytes(np.array([4], dtype="<i4").tobytes() + b"abcd")

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="trailing length word"):
            next(iter_fortran_records(handle, "<"))


def test_truncated_header_raises(tmp_path):
    """A trailing fragment too small to be a length word is an error, not an end."""
    path = tmp_path / "fragment.slf"
    path.write_bytes(_record(b"abcd", "<") + b"\x01\x02")

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="ends inside a Fortran record header"):
            list(iter_fortran_records(handle, "<"))


def test_negative_record_length_raises(tmp_path):
    """A negative declared length is nonsense and is refused."""
    path = tmp_path / "negative.slf"
    path.write_bytes(np.array([-8], dtype="<i4").tobytes() + b"12345678")

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="negative Fortran record length"):
            next(iter_fortran_records(handle, "<"))


def test_typed_read_at_end_of_file_raises(tmp_path):
    """Asking for a record when the file is exhausted is an error, not an empty array."""
    path = tmp_path / "empty.slf"
    path.write_bytes(b"")

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="found end of file"):
            read_record_ints(handle, "<", 1)


def test_read_record_floats_single_precision(tmp_path):
    """Single-precision records decode when ``double`` is False (plain SERAFIN)."""
    values = np.array([1.5, -2.25], dtype="<f4")
    path = tmp_path / "single.slf"
    path.write_bytes(_record(values.tobytes(), "<"))

    with open(path, "rb") as handle:
        got = read_record_floats(handle, "<", 2, double=False)

    np.testing.assert_allclose(got, [1.5, -2.25])


def test_record_offsets_rejects_a_truncated_header(tmp_path):
    """The offset index refuses a file ending inside a length word."""
    path = tmp_path / "bad_index.slf"
    path.write_bytes(_record(b"abcd", "<") + b"\x07")

    with pytest.raises(ValueError, match="ends inside a Fortran record header"):
        record_offsets(str(path), "<")


def test_record_offsets_rejects_a_negative_length(tmp_path):
    """The offset index refuses a negative declared record length."""
    path = tmp_path / "negative_index.slf"
    path.write_bytes(np.array([-4], dtype="<i4").tobytes() + b"abcd")

    with pytest.raises(ValueError, match="negative Fortran record length"):
        record_offsets(str(path), "<")


def test_record_offsets_diagnoses_a_truncated_payload(tmp_path):
    """A payload shorter than declared is reported as truncation, not a missing word.

    Seeking past the end of a file succeeds, so without an explicit size check the
    truncation surfaces one step later as a missing trailing length word -- caught, but
    blamed on the wrong thing.
    """
    path = tmp_path / "truncated_index.slf"
    path.write_bytes(np.array([40], dtype="<i4").tobytes() + b"short")

    with pytest.raises(ValueError, match="truncated Fortran record: declared 40 bytes"):
        record_offsets(str(path), "<")


@pytest.mark.parametrize("count", [3, 7])
def test_typed_read_rejects_a_count_the_record_cannot_hold(tmp_path, count):
    """A count disagreeing with the record length raises instead of taking a slab.

    Reading fewer values than the record holds used to succeed silently and return the
    wrong part of the file.
    """
    payload = np.array([1, 2, 3, 4, 5], dtype="<i4").tobytes()
    word = np.array([len(payload)], dtype="<i4").tobytes()
    path = tmp_path / "count.slf"
    path.write_bytes(word + payload + word)

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="the count is wrong"):
            read_record_ints(handle, "<", count)


def test_detect_endianness_honours_a_custom_first_record_length(tmp_path):
    """``expect_first_record`` is the probe value, not a fixed 80.

    A relative of Selafin with a different leading record is detected by passing its
    own length; the parameter existed and nothing ever supplied it, so a probe hard-wired
    to 80 would have read as working.
    """
    path = tmp_path / "custom.slf"
    path.write_bytes(_record(b"x" * 12, ">"))

    assert detect_endianness(str(path), expect_first_record=12) == ">"
    with pytest.raises(ValueError, match="80-byte first record"):
        detect_endianness(str(path))


def test_float_read_rejects_a_count_the_record_cannot_hold(tmp_path):
    """The float path checks the record length too, and names the type it needed.

    The integer path was covered; a float read that silently returned the first two of
    three values would be a plausible-looking array taken from the wrong slab.
    """
    payload = np.array([1.0, 2.0, 3.0], dtype="<f8")
    path = tmp_path / "floats.slf"
    path.write_bytes(_record(payload.tobytes(), "<"))

    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match=r"2 values of <f8 need 16"):
            read_record_floats(handle, "<", 2, double=True)


def test_record_offsets_of_an_empty_file_is_empty(tmp_path):
    """A zero-byte file indexes to no records rather than raising."""
    path = tmp_path / "empty.slf"
    path.write_bytes(b"")

    assert record_offsets(str(path), "<") == []


def test_iter_records_honours_a_custom_comment_marker(tmp_path):
    """The comment marker is configurable, for formats that use something else."""
    path = tmp_path / "custom.txt"
    path.write_text("! skipped\nKEY value\n", encoding="utf-8")

    assert list(iter_records(str(path), comment="!")) == [("KEY", ["value"])]


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
