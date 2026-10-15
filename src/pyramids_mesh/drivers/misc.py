"""One-off drivers with no shared parser: H2i, Mike21, ADCIRC.

Mike21 ``.mesh`` is plain text despite the DHI name, so it needs no extra. ADCIRC and
SAGA FLOW appear only in MDAL's README prose with no driver-table row, so their
support there is unstated -- verify against a real file before implementing.
Tasks D4 and D12.
"""
