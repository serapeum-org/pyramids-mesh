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

    The copy is 0-based, ``-1``-padded and counter-clockwise; ``start_index`` is
    preserved so a writer can still restore the file's base. The input is left
    untouched.

    Args:
        parsed: The mesh as a driver's ``_parse`` produced it.

    Returns:
        ParsedMesh: A normalised copy safe to construct pyramids objects from.

    Raises:
        ValueError: On an out-of-range index, a ragged connectivity array, a fill value
            that collides with a valid index, or a variable whose shape does not match
            its element and time counts.
    """
    check_shapes(parsed)
    n_node = int(np.asarray(parsed.node_x).shape[0])
    faces = _normalise_connectivity(
        parsed.faces, parsed.start_index, parsed.fill_value, n_node, "faces"
    )
    edges = _normalise_connectivity(
        parsed.edges, parsed.start_index, parsed.fill_value, n_node, "edges"
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
    return replace(parsed, faces=faces, edges=edges, fill_value=-1)


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
