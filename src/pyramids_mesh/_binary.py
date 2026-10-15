"""Fortran-record and endianness helpers for the binary formats.

Selafin and its relatives store Fortran unformatted records -- ``[int32 length]``,
payload, ``[int32 length]`` -- and **do not declare their endianness**: it is detected by
reading the first record length and checking it is 80, swapping byte order and retrying
if not.

Also builds the ``(offset, length)`` record index a lazy variable loader seeks into, so a
driver does not rescan the file per variable per timestep.

These helpers are pure parsing: they import nothing from ``pyramids`` and are testable
without it installed.

See ``planning/design.md`` section 19. Implemented by task S4.
"""

from __future__ import annotations

from typing import BinaryIO, Iterator

import numpy as np

# The Selafin title record is 80 characters, which is what makes it usable as the
# endianness probe: no other value is expected in the file's first length word.
_TITLE_RECORD_LENGTH = 80

_LENGTH_WORD = 4


def detect_endianness(
    path: str, *, expect_first_record: int = _TITLE_RECORD_LENGTH
) -> str:
    """Detect a Fortran-record file's byte order from its first record length.

    Reads the leading ``int32``; if it is not the expected record length the byte order
    is swapped and it is read again. If neither matches, the file is not of this format.
    This is exactly how MDAL probes a Selafin file.

    Args:
        path: The binary file to probe.
        expect_first_record: The length the first record must declare. Defaults to 80,
            the Selafin title record.

    Returns:
        str: ``"<"`` for little-endian or ``">"`` for big-endian, as a numpy dtype
            prefix.

    Raises:
        ValueError: If the file is shorter than one length word, or declares the
            expected length in neither byte order.

    Examples:
        - A little-endian file announces its 80-byte title record in the low byte:

            >>> import os
            >>> import tempfile
            >>> import numpy as np
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     word = np.array([80], dtype="<i4").tobytes()
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(word + b"T" * 80 + word)
            ...     detect_endianness(path)
            '<'

        - The same file written big-endian is detected by retrying the swapped
          order, which is the only way to tell: Selafin never declares it:

            >>> import os
            >>> import tempfile
            >>> import numpy as np
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     word = np.array([80], dtype=">i4").tobytes()
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(word + b"T" * 80 + word)
            ...     detect_endianness(path)
            '>'
    """
    with open(path, "rb") as handle:
        head = handle.read(_LENGTH_WORD)
    if len(head) < _LENGTH_WORD:
        raise ValueError(f"{path!r} is too short to hold a Fortran record header")
    for endian in ("<", ">"):
        if _as_int32(head, endian) == expect_first_record:
            return endian
    raise ValueError(
        f"{path!r} does not declare a {expect_first_record}-byte first record in "
        f"either byte order; it is not this format"
    )


def iter_fortran_records(handle: BinaryIO, endian: str) -> Iterator[bytes]:
    """Yield each Fortran unformatted record's payload in file order.

    Args:
        handle: A binary file object positioned at a record boundary.
        endian: ``"<"`` or ``">"``, as returned by :func:`detect_endianness`.

    Yields:
        bytes: Each record's payload, without its length words.

    Raises:
        ValueError: On a truncated record, or when the trailing length word does not
            match the leading one.

    Examples:
        - Walk two records, getting the payloads without their length words:

            >>> import os
            >>> import tempfile
            >>> import numpy as np
            >>> def record(payload):
            ...     word = np.array([len(payload)], dtype="<i4").tobytes()
            ...     return word + payload + word
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(record(b"abcd") + record(b"wxyz"))
            ...     with open(path, "rb") as handle:
            ...         payloads = list(iter_fortran_records(handle, "<"))
            >>> payloads
            [b'abcd', b'wxyz']

        - A trailing length word disagreeing with the leading one means a corrupt
          file or the wrong byte order, and is never read past:

            >>> import os
            >>> import tempfile
            >>> import numpy as np
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "corrupt.slf")
            ...     leading = np.array([4], dtype="<i4").tobytes()
            ...     trailing = np.array([5], dtype="<i4").tobytes()
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(leading + b"abcd" + trailing)
            ...     with open(path, "rb") as handle:
            ...         try:
            ...             next(iter_fortran_records(handle, "<"))
            ...         except ValueError as error:
            ...             print(str(error).split(";")[0])
            Fortran record length mismatch: leading 4, trailing 5
    """
    while True:
        payload = _read_record(handle, endian)
        if payload is None:
            return
        yield payload


def read_record_ints(handle: BinaryIO, endian: str, count: int) -> np.ndarray:
    """Read one record and interpret its payload as ``int32`` values.

    Args:
        handle: A binary file object positioned at a record boundary.
        endian: ``"<"`` or ``">"``.
        count: How many integers the record holds.

    Returns:
        np.ndarray: A writable array of ``count`` integers.

    Raises:
        ValueError: If the record is missing, truncated, or length-mismatched.
    """
    return _read_typed(handle, endian, count, "i4")


def read_record_floats(
    handle: BinaryIO, endian: str, count: int, *, double: bool
) -> np.ndarray:
    """Read one record and interpret its payload as floating-point values.

    Args:
        handle: A binary file object positioned at a record boundary.
        endian: ``"<"`` or ``">"``.
        count: How many floats the record holds.
        double: ``True`` for ``float64`` (Selafin's ``SERAFIND``), ``False`` for
            ``float32``.

    Returns:
        np.ndarray: A writable array of ``count`` floats.

    Raises:
        ValueError: If the record is missing, truncated, or length-mismatched.
    """
    return _read_typed(handle, endian, count, "f8" if double else "f4")


def record_offsets(path: str, endian: str) -> list[tuple[int, int]]:
    """Index every record's payload position, for a lazy loader to seek into.

    Built once while parsing, so a lazy read seeks straight to a variable's slab
    instead of rescanning the file per variable per timestep.

    Args:
        path: The binary file to index.
        endian: ``"<"`` or ``">"``.

    Returns:
        list[tuple[int, int]]: One ``(offset, length)`` pair per record, in file order,
            where ``offset`` is the payload's byte position.

    Raises:
        ValueError: On a truncated or length-mismatched record.

    Examples:
        - Index a two-record file. The first payload starts after its 4-byte length
          word, and the second after the first record's trailing word:

            >>> import os
            >>> import tempfile
            >>> import numpy as np
            >>> def record(payload):
            ...     word = np.array([len(payload)], dtype="<i4").tobytes()
            ...     return word + payload + word
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(record(b"T" * 80) + record(b"abcd"))
            ...     offsets = record_offsets(path, "<")
            >>> offsets
            [(4, 80), (92, 4)]

        - The index is what a lazy loader seeks into, so a variable is read without
          rescanning the file:

            >>> import os
            >>> import tempfile
            >>> import numpy as np
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "model.slf")
            ...     word = np.array([4], dtype="<i4").tobytes()
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(word + b"wxyz" + word)
            ...     offset, length = record_offsets(path, "<")[0]
            ...     with open(path, "rb") as handle:
            ...         _ = handle.seek(offset)
            ...         handle.read(length)
            b'wxyz'
    """
    offsets: list[tuple[int, int]] = []
    with open(path, "rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(0)
        while True:
            head = handle.read(_LENGTH_WORD)
            if not head:
                return offsets
            length = _decode_length(head, endian, path)
            offset = handle.tell()
            # Seeking past the end succeeds silently, so the truncation would otherwise
            # surface one step later as a missing trailing word -- caught, but
            # misdiagnosed.
            if offset + length > size:
                raise ValueError(
                    f"truncated Fortran record: declared {length} bytes, "
                    f"{size - offset} remain in the file"
                )
            handle.seek(length, 1)
            _check_trailer(handle, length, endian=endian)
            offsets.append((offset, length))


def _as_int32(raw: bytes, endian: str) -> int:
    """One ``int32`` decoded from four bytes in the given byte order."""
    return int(np.frombuffer(raw, dtype=f"{endian}i4", count=1)[0])


def _decode_length(head: bytes, endian: str, path: str | None = None) -> int:
    """Return the payload length a record header declares, validated as usable.

    One implementation for both the sequential read and the offset index, so the two
    cannot drift into reporting the same corruption differently.

    Args:
        head: The four bytes of the length word.
        endian: ``"<"`` or ``">"``.
        path: The file being read, named in the message when known.

    Returns:
        int: The declared payload length.

    Raises:
        ValueError: If the header is short, or declares a negative length.
    """
    where = f"{path!r} " if path is not None else ""
    if len(head) < _LENGTH_WORD:
        raise ValueError(f"{where}ends inside a Fortran record header")
    length = _as_int32(head, endian)
    if length < 0:
        raise ValueError(f"{where}declares a negative Fortran record length {length}")
    return length


def _read_record(handle: BinaryIO, endian: str) -> bytes | None:
    """One record's payload, or ``None`` at a clean end of file."""
    head = handle.read(_LENGTH_WORD)
    if not head:
        return None
    length = _decode_length(head, endian)
    payload = handle.read(length)
    if len(payload) != length:
        raise ValueError(
            f"truncated Fortran record: declared {length} bytes, read {len(payload)}"
        )
    _check_trailer(handle, length, endian=endian)
    return payload


def _check_trailer(handle: BinaryIO, length: int, *, endian: str) -> None:
    """Verify a record's trailing length word against its leading one.

    A mismatch means a corrupt file or the wrong byte order. Ignoring it yields arrays
    that look plausible and are wrong, so it always raises.

    Args:
        handle: The file object, positioned at the trailing length word.
        length: The length the leading word declared.
        endian: ``"<"`` or ``">"``.

    Raises:
        ValueError: If the trailing word is missing or disagrees with ``length``.
    """
    tail = handle.read(_LENGTH_WORD)
    if len(tail) < _LENGTH_WORD:
        raise ValueError("Fortran record is missing its trailing length word")
    trailing = _as_int32(tail, endian)
    if trailing != length:
        raise ValueError(
            f"Fortran record length mismatch: leading {length}, trailing {trailing}; "
            f"the file is corrupt or the byte order is wrong"
        )


def _read_typed(handle: BinaryIO, endian: str, count: int, code: str) -> np.ndarray:
    """One record decoded as ``count`` values of the given numpy type code.

    The record's own length is checked against ``count`` first. A wrong ``count`` is
    exactly the mistake a driver makes when it misreads a header field, and without the
    check numpy happily returns the requested number of values taken from the wrong part
    of the file -- a plausible-looking array that is simply wrong, which is the failure
    class the trailing-length-word verification exists to prevent.

    Raises:
        ValueError: At end of file, or when the record does not hold exactly ``count``
            values of this type.
    """
    payload = _read_record(handle, endian)
    if payload is None:
        raise ValueError("expected a Fortran record, found end of file")
    dtype = np.dtype(f"{endian}{code}")
    expected = count * dtype.itemsize
    if len(payload) != expected:
        raise ValueError(
            f"Fortran record holds {len(payload)} bytes but {count} values of "
            f"{dtype.str} need {expected}; the count is wrong"
        )
    return np.frombuffer(payload, dtype=dtype, count=count).copy()
