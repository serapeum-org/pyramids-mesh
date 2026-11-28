"""One reader for a Selafin file's header, for the tests that assert on what was written.

Not a test module -- a helper the Selafin test modules share. Both of them need to look at
raw header records (the title, ``IPARAM``, the date record, ``IPOBO``, the variable names),
and walking those records by hand in each test is the drift the driver's own
``_peek_header`` exists to remove. Reproducing it five times in the tests would be the same
mistake one level down: a test that walks the records slightly differently from the driver
proves something slightly different from what it claims.

Deliberately independent of the driver's own walk: these tests check what the writer
*emitted*, so reading it back through the same code that produced it would make the check
circular. This is a plain, literal re-implementation of the record order from the format
specification.
"""

import pyramids  # noqa: F401  # isort:skip

from pyramids_mesh._binary import _read_record, detect_endianness, read_record_ints

_TITLE_LENGTH = 80
_TITLE_TEXT = 72
_IPARAM_COUNT = 10
_DATE_FIELDS = 6


def read_header(path) -> dict:
    """Read a Selafin file's header records into a dictionary.

    Args:
        path: The Selafin file to read; anything :func:`open` accepts.

    Returns:
        dict: ``title`` (stripped free text), ``marker`` (``SERAFIN``/``SERAFIND``),
            ``nbv`` (the two variable counts), ``names`` (16-character names, stripped),
            ``units`` (the 16-character unit halves, stripped), ``iparam`` (ten ints),
            ``date`` (six ints, or ``None`` when ``IPARAM[9]`` is not 1), ``nelem``,
            ``npoin``, ``ndp`` and ``ipobo``.

    Examples:
        - The minimal fixture declares one variable, no date record and no boundary nodes:

            >>> import os
            >>> import tempfile
            >>> from pyramids_mesh.drivers.coastal_ocean.telemac import (
            ...     build_minimal_fixture_selafin,
            ... )
            >>> from tests.drivers.coastal_ocean.selafin_header import read_header
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "minimal.slf")
            ...     build_minimal_fixture_selafin(path)
            ...     header = read_header(path)
            >>> header["title"], header["names"], header["date"]
            ('pyramids-mesh fixture', ['WATER DEPTH'], None)
            >>> header["nelem"], header["npoin"], header["ndp"]
            (2, 4, 3)
            >>> set(header["ipobo"])
            {0}
    """
    endian = detect_endianness(str(path), expect_first_record=_TITLE_LENGTH)
    with open(path, "rb") as handle:
        title_record = _read_record(handle, endian).decode("latin-1")
        nbv = read_record_ints(handle, endian, 2).tolist()
        raw_names = [
            _read_record(handle, endian).decode("latin-1")
            for _ in range(int(nbv[0] + nbv[1]))
        ]
        iparam = read_record_ints(handle, endian, _IPARAM_COUNT).tolist()
        # The flag decides whether the record is there at all, so it is read first -- a
        # read before the check would consume the dimensions record on an undated file.
        date = (
            read_record_ints(handle, endian, _DATE_FIELDS).tolist()
            if iparam[9] == 1
            else None
        )
        dims = read_record_ints(handle, endian, 4).tolist()
        nelem, npoin, ndp = int(dims[0]), int(dims[1]), int(dims[2])
        read_record_ints(handle, endian, nelem * ndp)
        ipobo = read_record_ints(handle, endian, npoin).tolist()
    return {
        "endian": endian,
        "title": title_record[:_TITLE_TEXT].strip(),
        "marker": title_record[_TITLE_TEXT:].strip(),
        "nbv": nbv,
        "names": [name[:16].strip() for name in raw_names],
        "units": [name[16:32].strip() for name in raw_names],
        "iparam": iparam,
        "date": date,
        "nelem": nelem,
        "npoin": npoin,
        "ndp": ndp,
        "ipobo": ipobo,
    }
