"""Fortran-record and endianness helpers for the binary formats.

Selafin and its relatives store Fortran unformatted records -- ``[int32 length]``,
payload, ``[int32 length]`` -- and **do not declare their endianness**: it is detected by
reading the first record length and checking it is 80, swapping byte order and retrying
if not.

Also builds the ``(offset, length)`` record index a lazy variable loader seeks into, so a
driver does not rescan the file per variable per timestep.

See ``planning/design.md`` section 19. Implemented by task S4.
"""
