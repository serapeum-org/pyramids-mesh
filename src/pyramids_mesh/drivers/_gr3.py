"""Shared parser for the ADCIRC ``fort.14`` / SCHISM ``.gr3`` ASCII grid layout.

ADCIRC's ``fort.14`` and SCHISM's ``hgrid.gr3`` share one ASCII layout: a free-text
title line, an ``NE NP`` count pair (elements, nodes), ``NP`` node lines ``id x y depth``
and ``NE`` element lines ``id n_type n1 n2 n3[ n4]``. This module holds the one node- and
element-block parser the two drivers share, so a fix to it reaches both.

The only real difference is the element-type column: ADCIRC fixes it at ``3``
(triangles only), while SCHISM allows ``i34`` in ``{3, 4}`` -- a triangle or a quad --
followed by open/land boundary sections the caller reads on from the same handle.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

from typing import TextIO

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError


def sniff_gr3(path: str, *, encoding: str = "latin-1") -> bool:
    """Report whether ``path`` opens as a ``fort.14`` / ``.gr3`` grid.

    Cheap and discriminating: line 2 is the ``NE NP`` count pair (two positive ints) and
    the first node line is ``id x y depth`` (one int then three floats). A trailing
    inline comment (SCHISM writes ``! # of elements and nodes``) is ignored because only
    the leading fields are read.

    Args:
        path: The file to inspect.
        encoding: The text encoding to read with; ``latin-1`` never raises on a stray
            byte, which suits a sniff that must not throw on a foreign file.

    Returns:
        bool: ``True`` when the header and first node line are grid-shaped.
    """
    with open(path, "r", encoding=encoding) as handle:
        handle.readline()  # title
        counts = handle.readline().split()
        node = handle.readline().split()
    if len(counts) < 2:
        return False
    try:
        n_elem, n_node = int(counts[0]), int(counts[1])
    except ValueError:
        return False
    if n_elem <= 0 or n_node <= 0 or len(node) < 4:
        return False
    try:
        int(node[0])
        float(node[1])
        float(node[2])
        float(node[3])
    except ValueError:
        return False
    return True


def read_gr3_grid(handle: TextIO, *, allow_quads: bool, source: str):
    """Read the title, counts, node block and element block from an open handle.

    The handle is left positioned at the first line after the element block, so a caller
    with more to read -- SCHISM's boundary sections -- continues straight from there.

    Node ids are not consumed: the node block is assumed contiguous and 1-based (true of
    both formats), so a node's row is its id. Connectivity is returned with the file's
    own 1-based ids; ``validate_and_normalise`` re-bases it. A mixed triangle/quad grid
    comes back as an ``(n_elem, 4)`` table whose triangle rows are padded in the trailing
    column with ``-1``; a triangle-only grid stays ``(n_elem, 3)``.

    Args:
        handle: An open text handle positioned at the start of the file.
        allow_quads: When ``True`` a ``type 4`` element is read as a quad; when ``False``
            any non-triangle element is refused (ADCIRC grids are triangulations).
        source: The file path, used only in the error message.

    Returns:
        tuple: ``(node_x, node_y, node_z, faces)`` -- three float64 coordinate/elevation
            arrays of length ``n_node`` and the int64 face-node table, 1-based and
            ``-1``-padded.

    Raises:
        FileFormatNotSupportedError: If an element's type column is one this reader does
            not accept.
    """
    handle.readline()  # title
    n_elem, n_node = (int(value) for value in handle.readline().split()[:2])
    nodes = np.empty((n_node, 3), dtype=np.float64)
    for row in range(n_node):
        fields = handle.readline().split()
        nodes[row] = (fields[1], fields[2], fields[3])
    element_nodes: list[tuple[int, ...]] = []
    width = 3
    for _ in range(n_elem):
        fields = handle.readline().split()
        i34 = fields[1]
        if i34 == "3":
            element_nodes.append((int(fields[2]), int(fields[3]), int(fields[4])))
        elif i34 == "4" and allow_quads:
            element_nodes.append(
                (int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5]))
            )
            width = 4
        else:
            allowed = "triangular (type 3)"
            if allow_quads:
                allowed = "triangular (type 3) and quadrilateral (type 4)"
            raise FileFormatNotSupportedError(
                f"{source!r} element {fields[0]} has type {fields[1]!r}; this reader "
                f"only reads {allowed} grids"
            )
    faces = np.full((n_elem, width), -1, dtype=np.int64)
    for index, vertices in enumerate(element_nodes):
        faces[index, : len(vertices)] = vertices
    return nodes[:, 0], nodes[:, 1], nodes[:, 2], faces
