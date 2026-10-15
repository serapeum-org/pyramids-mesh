"""Enforcement of the invariants pyramids assumes of a mesh.

One shared implementation, because every one of these failures is **silent**: a driver
that forgets one produces wrong numbers rather than an error.

1. Connectivity is 0-indexed -- ``original_start_index`` only records the file's base.
2. Faces are fill-padded to a rectangular array, never ragged.
3. The fill value does not collide with a valid index (``-1``).
4. Face winding is counter-clockwise -- area, centroid, ``polygonize`` and ``voronoi``
   all assume it.
5. Every variable's declared shape matches its element and time counts.

Checks 1-3 raise on the unrecoverable (an out-of-range index, a ragged array, a fill
value indistinguishable from a real index); check 4 corrects what it can and leaves a
genuinely degenerate face alone with a warning; check 5 reads ``shape`` only, never the
values, so it never defeats a lazy load.

Normalisation also settles the two index bases: ``start_index`` becomes ``0`` (the
re-basing is done) and ``original_start_index`` becomes the base a writer should emit,
so the returned object never contradicts itself and re-validating it changes nothing.

See ``planning/design.md`` section 17.3. Implemented by task S3.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids a circular import
    from pyramids_mesh._base import ParsedMesh

_ELEMENT_COUNT_SOURCE = {"node": "node", "face": "face", "edge": "edge"}


def validate_and_normalise(parsed: "ParsedMesh") -> "ParsedMesh":
    """Enforce every pyramids invariant, returning a corrected copy.

    The copy is 0-based, ``-1``-padded and counter-clockwise, and it describes its own
    state: ``start_index`` is ``0`` because there is nothing left to subtract, while
    ``original_start_index`` carries the base a **writer** should emit. Because the copy
    is self-consistent, validating it again is a no-op rather than a second re-basing.
    The input is left untouched.

    Args:
        parsed: The mesh as a driver's ``_parse`` produced it.

    Returns:
        ParsedMesh: A normalised copy safe to construct pyramids objects from.

    Raises:
        ValueError: On an out-of-range index, a ragged connectivity array, a fill value
            that collides with a valid index, or a variable whose shape does not match
            its element and time counts.

    Examples:
        - A 1-based file becomes 0-based. The copy then says so -- ``start_index`` is
          ``0`` because the re-basing is done -- while ``original_start_index``
          carries the base a writer should emit:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh
            >>> parsed = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     faces=np.array([[1, 2, 3]]),
            ...     start_index=1,
            ... )
            >>> result = validate_and_normalise(parsed)
            >>> result.faces.tolist()
            [[0, 1, 2]]
            >>> result.start_index, result.original_start_index
            (0, 1)

        - A clockwise face is reversed, because area, centroid and ``polygonize``
          all assume counter-clockwise winding:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh
            >>> clockwise = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 1.0, 0.0]),
            ...     faces=np.array([[0, 1, 2]]),
            ... )
            >>> validate_and_normalise(clockwise).faces.tolist()
            [[2, 1, 0]]

        - A fill value that a real index could also take is unrecoverable, so it
          raises rather than guessing which entries are padding:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh
            >>> colliding = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     faces=np.array([[0, 1, 2]]),
            ...     fill_value=0,
            ... )
            >>> try:
            ...     validate_and_normalise(colliding)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            faces fill_value 0 collides with a valid index (base 0, 3 nodes)
    """
    check_shapes(parsed)
    n_node = int(np.asarray(parsed.node_x).shape[0])
    faces = _normalise_connectivity(
        parsed.faces, parsed.start_index, parsed.fill_value, n_node, "faces"
    )
    edges = _normalise_connectivity(
        parsed.edges, parsed.start_index, parsed.fill_value, n_node, "edges"
    )
    network_edges = _normalise_connectivity(
        parsed.network_edges,
        parsed.start_index,
        parsed.fill_value,
        n_node,
        "network_edges",
    )
    if faces is not None:
        faces, reversed_count, degenerate_count = fix_winding(
            faces, np.asarray(parsed.node_x), np.asarray(parsed.node_y), -1
        )
        if degenerate_count:
            warnings.warn(
                f"{degenerate_count} degenerate face(s) have no winding and were "
                f"left unchanged",
                stacklevel=2,
            )
    writer_base = parsed.start_index
    if parsed.original_start_index is not None:
        writer_base = parsed.original_start_index
    return replace(
        parsed,
        faces=faces,
        edges=edges,
        network_edges=network_edges,
        fill_value=-1,
        start_index=0,
        original_start_index=writer_base,
    )


def _normalise_connectivity(
    conn: np.ndarray | None,
    start_index: int,
    fill_value: int,
    n_node: int,
    label: str,
) -> np.ndarray | None:
    """Range-check and rewrite one connectivity array to 0-based, ``-1``-padded form.

    Args:
        conn: The connectivity array as the file gave it, or ``None``.
        start_index: The file's connectivity base.
        fill_value: The pad value the file used.
        n_node: The node count, for the range check.
        label: ``"faces"`` or ``"edges"``, used in error messages.

    Returns:
        np.ndarray | None: The 0-based, ``-1``-padded array, or ``None`` if ``conn``
            was ``None``.

    Raises:
        ValueError: On a ragged array, a fill/index collision, or an out-of-range index.
    """
    if conn is None:
        return None
    array = np.asarray(conn)
    if array.dtype == object or array.ndim != 2:
        raise ValueError(
            f"{label} connectivity must be a rectangular 2-D array; a ragged one is a "
            f"driver bug"
        )
    if start_index <= fill_value < start_index + n_node:
        raise ValueError(
            f"{label} fill_value {fill_value} collides with a valid index "
            f"(base {start_index}, {n_node} nodes); real index and fill cannot be told "
            f"apart"
        )
    real = array != fill_value
    shifted = to_zero_based(array, start_index, fill_value)
    valid = shifted[real]
    if valid.size and (valid.min() < 0 or valid.max() >= n_node):
        raise ValueError(
            f"{label} index out of range after re-basing: expected 0..{n_node - 1}, "
            f"got {int(valid.min())}..{int(valid.max())}; the start index was wrong"
        )
    return shifted


def to_zero_based(conn: np.ndarray, start_index: int, fill_value: int) -> np.ndarray:
    """Re-base real connectivity entries to 0, setting every pad entry to ``-1``.

    The fill value is masked out before the subtraction, so padding is never shifted
    into ``-2`` (design's trap: mask first, subtract second).

    Args:
        conn: The connectivity array.
        start_index: The base to subtract from real entries.
        fill_value: The value marking padding.

    Returns:
        np.ndarray: An integer array with real entries re-based to 0 and padding ``-1``.

    Examples:
        - Re-base a 1-based array whose short rows are padded with ``0``. The pad
          becomes ``-1`` rather than being shifted to ``-1`` by the subtraction:

            >>> import numpy as np
            >>> conn = np.array([[1, 2, 3, 0], [1, 3, 4, 0]])
            >>> to_zero_based(conn, start_index=1, fill_value=0).tolist()
            [[0, 1, 2, -1], [0, 2, 3, -1]]

        - An array that is already 0-based is unchanged apart from its padding:

            >>> import numpy as np
            >>> to_zero_based(np.array([[0, 1, 2]]), 0, -1).tolist()
            [[0, 1, 2]]
    """
    array = np.asarray(conn)
    out = np.full(array.shape, -1, dtype=np.int64)
    real = array != fill_value
    out[real] = array[real] - start_index
    return out


def repad(conn: np.ndarray, old_fill: int, new_fill: int = -1) -> np.ndarray:
    """Replace one pad value with another, leaving real entries untouched.

    Args:
        conn: The connectivity array.
        old_fill: The pad value to replace.
        new_fill: The pad value to use. Defaults to ``-1``.

    Returns:
        np.ndarray: A copy with ``old_fill`` entries rewritten to ``new_fill``.

    Examples:
        - Rewrite a format's own pad value to the ``-1`` pyramids expects:

            >>> import numpy as np
            >>> repad(np.array([[0, 1, 9], [2, 3, 9]]), old_fill=9).tolist()
            [[0, 1, -1], [2, 3, -1]]

        - Real entries are left alone, so a row with no padding is untouched:

            >>> import numpy as np
            >>> repad(np.array([[0, 1, 2]]), old_fill=9).tolist()
            [[0, 1, 2]]
    """
    out = np.asarray(conn).copy()
    out[out == old_fill] = new_fill
    return out


def fix_winding(
    faces: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    fill_value: int,
) -> tuple[np.ndarray, int, int]:
    """Reverse clockwise faces to counter-clockwise, counting what was changed.

    The signed area is the shoelace sum over a face's **real** vertices only; padding
    is sliced off first, or a ``-1`` entry poisons the sign and reverses a correct face.
    A degenerate face (zero signed area: collinear or a repeated node) has no winding,
    so it is neither reversed nor rejected, matching pyramids' own handling.

    Args:
        faces: The 0-based, fill-padded face-node connectivity.
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        fill_value: The pad value marking short rows.

    Returns:
        tuple[np.ndarray, int, int]: The corrected faces, the number of faces reversed,
            and the number of degenerate faces found.

    Examples:
        - A clockwise triangle is reversed and counted:

            >>> import numpy as np
            >>> faces = np.array([[0, 1, 2]])
            >>> node_x = np.array([0.0, 1.0, 1.0])
            >>> node_y = np.array([0.0, 1.0, 0.0])
            >>> fixed, reversed_count, degenerate = fix_winding(
            ...     faces, node_x, node_y, -1
            ... )
            >>> fixed.tolist()
            [[2, 1, 0]]
            >>> reversed_count
            1

        - Padding is sliced off before the shoelace sum, so a padded
          counter-clockwise face is left alone instead of being flipped:

            >>> import numpy as np
            >>> padded = np.array([[0, 1, 2, -1]])
            >>> node_x = np.array([0.0, 1.0, 1.0])
            >>> node_y = np.array([0.0, 0.0, 1.0])
            >>> fixed, reversed_count, degenerate = fix_winding(
            ...     padded, node_x, node_y, -1
            ... )
            >>> fixed.tolist()
            [[0, 1, 2, -1]]
            >>> reversed_count
            0

        - A collinear face has no winding, so it is counted rather than reversed:

            >>> import numpy as np
            >>> flat = np.array([[0, 1, 2]])
            >>> node_x = np.array([0.0, 1.0, 2.0])
            >>> node_y = np.array([0.0, 0.0, 0.0])
            >>> _, reversed_count, degenerate = fix_winding(
            ...     flat, node_x, node_y, -1
            ... )
            >>> reversed_count, degenerate
            (0, 1)
    """
    out = np.asarray(faces).copy()
    reversed_count = 0
    degenerate_count = 0
    for row in out:
        real = row[row != fill_value]
        if real.size < 3:
            continue
        area2 = _signed_area_2(node_x[real], node_y[real])
        if area2 < 0:
            row[: real.size] = real[::-1]
            reversed_count += 1
        elif area2 == 0:
            degenerate_count += 1
    return out, reversed_count, degenerate_count


def _signed_area_2(xs: np.ndarray, ys: np.ndarray) -> float:
    """Twice the signed polygon area via the shoelace formula (positive is CCW)."""
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def check_shapes(parsed: "ParsedMesh") -> None:
    """Check every variable's declared shape against its element and time counts.

    Reads ``shape`` only -- never ``values`` or ``loader`` -- so a lazy variable is
    validated without forcing a read.

    Args:
        parsed: The mesh carrying the variables.

    Raises:
        ValueError: If a variable's location is unknown, or its shape does not equal
            ``(element_count,)`` (static) or ``(len(time_values), element_count)``
            (temporal).
    """
    counts = {
        "node": int(np.asarray(parsed.node_x).shape[0]),
        "face": 0 if parsed.faces is None else int(np.asarray(parsed.faces).shape[0]),
        "edge": 0 if parsed.edges is None else int(np.asarray(parsed.edges).shape[0]),
    }
    for var_name, var in parsed.variables.items():
        if var.location not in counts:
            raise ValueError(
                f"variable {var_name!r} has unknown location {var.location!r}"
            )
        element_count = counts[var.location]
        expected: tuple[int, ...]
        if var.time_values is not None:
            expected = (int(np.asarray(var.time_values).shape[0]), element_count)
        else:
            expected = (element_count,)
        if tuple(var.shape) != expected:
            raise ValueError(
                f"variable {var_name!r} shape {tuple(var.shape)} does not match its "
                f"{var.location} count: expected {expected}"
            )
