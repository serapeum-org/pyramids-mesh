"""Enforcement of the invariants pyramids assumes of a mesh.

One shared implementation, because every one of these failures is **silent**: a driver
that forgets one produces wrong numbers rather than an error.

1. Connectivity is 0-indexed -- ``original_start_index`` only records the base a writer
   should emit.
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
            that collides with a valid index, a per-node or per-face array whose length
            disagrees with the topology, a variable carrying both or neither of
            ``values``/``loader``, a variable whose shape does not match its element and
            time counts, or a face row padded between its real vertices.

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
    # Rectangularity first: a ragged list of lists is the natural product of a
    # line-oriented text parser, and numpy's own message for it says nothing about
    # which array of whose driver is at fault.
    _require_rectangular(parsed.faces, "faces")
    _require_rectangular(parsed.edges, "edges")
    _require_rectangular(parsed.network_edges, "network_edges")
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
        if reversed_count:
            # Reversal is mandatory normalisation, but it changes the vertex order a
            # writer would emit, so the file that comes back out is not byte-identical
            # to the one read. That is worth saying rather than doing quietly.
            warnings.warn(
                f"{reversed_count} clockwise face(s) were reversed to "
                f"counter-clockwise; a writer will emit the corrected order",
                stacklevel=3,
            )
        if degenerate_count:
            warnings.warn(
                f"{degenerate_count} degenerate face(s) have no winding and were "
                f"left unchanged",
                stacklevel=3,
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
        label: ``"faces"``, ``"edges"`` or ``"network_edges"``, used in error messages.

    Returns:
        np.ndarray | None: The 0-based, ``-1``-padded array, or ``None`` if ``conn``
            was ``None``.

    Raises:
        ValueError: On a ragged array, a fill/index collision, or an out-of-range index.
    """
    if conn is None:
        return None
    array = _require_rectangular(conn, label)
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


def _require_rectangular(conn: np.ndarray | None, label: str) -> np.ndarray | None:
    """Return ``conn`` as a rectangular 2-D array, or raise saying it is ragged.

    Args:
        conn: The connectivity a driver produced, or ``None``.
        label: ``"faces"``, ``"edges"`` or ``"network_edges"``, for the message.

    Returns:
        np.ndarray | None: The array, or ``None`` when ``conn`` was ``None``.

    Raises:
        ValueError: If the rows are of unequal length, whether that reaches here as a
            list of lists or as an object-dtype array.
    """
    if conn is None:
        return None
    message = (
        f"{label} connectivity must be a rectangular 2-D array; a ragged one is a "
        f"driver bug -- pad short rows with the fill value"
    )
    try:
        array = np.asarray(conn)
    except ValueError as error:
        raise ValueError(message) from error
    if array.dtype == object or array.ndim != 2:
        raise ValueError(message)
    return array


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
    A degenerate face -- one whose area is zero to within a tolerance scaled by its own
    span: collinear, or a repeated node -- has no winding, so it is neither reversed nor
    rejected, matching pyramids' own handling.

    Reversal rewrites the leading run of a row, which assumes UGRID's convention that
    padding is a trailing run. A row padded between real vertices is rejected rather
    than silently relaid.

    Args:
        faces: The 0-based, fill-padded face-node connectivity.
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        fill_value: The pad value marking short rows.

    Returns:
        tuple[np.ndarray, int, int]: The corrected faces, the number of faces reversed,
            and the number of degenerate faces found.

    Raises:
        ValueError: If a row pads between real vertices.

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
        vertices = row[row != fill_value]
        if vertices.size < 3:
            continue
        pad_positions = np.flatnonzero(row == fill_value)
        if pad_positions.size and int(pad_positions.min()) < vertices.size:
            raise ValueError(
                f"face row {row.tolist()} pads between real vertices; UGRID pads only "
                f"the tail of a row, and reversing this one would move the padding"
            )
        xs, ys = node_x[vertices], node_y[vertices]
        area2 = _signed_area_2(xs, ys)
        if abs(area2) <= _area_tolerance(xs, ys):
            degenerate_count += 1
        elif area2 < 0:
            row[: vertices.size] = vertices[::-1]
            reversed_count += 1
    return out, reversed_count, degenerate_count


def _area_tolerance(xs: np.ndarray, ys: np.ndarray) -> float:
    """Return the area below which a face counts as degenerate.

    Degeneracy is "smaller than this face's coordinates can resolve", so the threshold
    needs both of the quantities that decide that: the face's **span**, which sets the
    scale of the area, and the coordinate **magnitude**, which sets how finely the
    vertices are representable at all.

    Both halves were learned the hard way. A magnitude-only threshold is not a
    geometric property: at a northing of 7e6 it reached ~0.09 and declared every element
    smaller than that degenerate. A span-only threshold ignores representability: near
    7e6 a coordinate resolves to ~4.7e-10, so three collinear points do not *store* as
    collinear and a truly flat face carries ~1e-9 of irreducible area -- far above
    ``eps * span**2``, so it was read as clockwise and reversed, and no mesh in a
    projected CRS ever reported a degenerate face.

    Args:
        xs: The face's vertex x-coordinates.
        ys: The face's vertex y-coordinates.

    Returns:
        float: The absolute signed-area threshold for this face.
    """
    span = max(float(np.ptp(xs)), float(np.ptp(ys)))
    magnitude = max(float(np.abs(xs).max()), float(np.abs(ys).max()))
    return 8.0 * float(np.finfo(np.float64).eps) * span * max(span, magnitude)


def _check_node_arrays(parsed: "ParsedMesh", n_node: int) -> None:
    """Check every per-node and per-face array is as long as the topology it describes.

    ``node_y`` and ``node_z`` are measured against ``node_x``; ``face_x`` and ``face_y``
    against the face count, and only when the parse has faces at all.

    Without this a short ``node_y`` surfaces as an ``IndexError`` from inside the
    shoelace sum, which never mentions the mismatch and does not fire at all on a parse
    with no faces; a short ``node_z`` is not noticed at any point and builds a variable
    whose declared shape contradicts its data.

    Args:
        parsed: The mesh whose node and face-centre arrays to check.
        n_node: The node count, taken from ``node_x``.

    Raises:
        ValueError: If any per-node or per-face array has a different length.
    """
    for label in ("node_y", "node_z"):
        array = getattr(parsed, label)
        if array is None:
            continue
        length = int(np.asarray(array).shape[0])
        if length != n_node:
            raise ValueError(
                f"{label} has {length} values but node_x has {n_node}; every per-node "
                f"array must be the same length"
            )
    for label in ("face_x", "face_y"):
        array = getattr(parsed, label)
        if array is None or parsed.faces is None:
            continue
        n_face = int(np.asarray(parsed.faces).shape[0])
        length = int(np.asarray(array).shape[0])
        if length != n_face:
            raise ValueError(
                f"{label} has {length} values but there are {n_face} faces"
            )


def _signed_area_2(xs: np.ndarray, ys: np.ndarray) -> float:
    """Return twice the signed polygon area by the shoelace formula (positive is CCW).

    The vertices are translated to the first one before the sum. Over absolute projected
    coordinates the products reach ~1e13 while the area of a metre-scale element is
    ~1e-2, so the subtraction cancels and the result is noise: a real face reads as
    degenerate and a clockwise one is never reversed. Translating first keeps every
    product the size of the element itself, and the shoelace sum is invariant under it.

    Args:
        xs: The vertex x-coordinates, in file order.
        ys: The vertex y-coordinates, in file order.

    Returns:
        float: Twice the signed area; positive counter-clockwise.
    """
    local_x = xs - xs[0]
    local_y = ys - ys[0]
    return float(
        np.dot(local_x, np.roll(local_y, -1)) - np.dot(local_y, np.roll(local_x, -1))
    )


def check_shapes(parsed: "ParsedMesh") -> None:
    """Check every variable's declared shape against its element and time counts.

    Reads ``shape`` only -- never ``values`` or ``loader`` -- so a lazy variable is
    validated without forcing a read. It also checks the two things that otherwise make
    ``shape`` untrustworthy: that every per-node array is as long as ``node_x``, and
    that each variable carries exactly one of an eager array or a lazy loader.

    Args:
        parsed: The mesh carrying the variables.

    Raises:
        ValueError: If a per-node or per-face array length disagrees with the topology,
            a variable carries both or neither of ``values``/``loader``, its location is
            unknown, or its shape does not equal ``(element_count,)`` (static) or
            ``(len(time_values), element_count)`` (temporal).

    Examples:
        - A sound mesh passes silently; the check returns nothing and reads no values,
          so a lazy variable is validated without being loaded:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh, ParsedVariable
            >>> parsed = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     faces=np.array([[0, 1, 2]]),
            ...     variables={
            ...         "depth": ParsedVariable(
            ...             location="face",
            ...             shape=(1,),
            ...             loader=lambda: np.array([2.5]),
            ...         )
            ...     },
            ... )
            >>> print(check_shapes(parsed))
            None

        - A shape that disagrees with the element count is named along with the count
          it should have matched:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh, ParsedVariable
            >>> mismatched = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     faces=np.array([[0, 1, 2]]),
            ...     variables={
            ...         "depth": ParsedVariable(
            ...             location="face",
            ...             shape=(3,),
            ...             values=np.array([1.0, 2.0, 3.0]),
            ...         )
            ...     },
            ... )
            >>> try:
            ...     check_shapes(mismatched)
            ... except ValueError as error:
            ...     print(error)
            variable 'depth' shape (3,) does not match its face count: expected (1,)

        - A per-node array shorter than ``node_x`` is caught here rather than much
          later, inside the shoelace sum:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh
            >>> short = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 0.0]),
            ... )
            >>> try:
            ...     check_shapes(short)
            ... except ValueError as error:
            ...     print(str(error).split(";")[0])
            node_y has 2 values but node_x has 3

        - A variable must carry exactly one of an eager array or a lazy loader;
          carrying neither leaves nothing to read:

            >>> import numpy as np
            >>> from pyramids_mesh._base import ParsedMesh, ParsedVariable
            >>> empty = ParsedMesh(
            ...     node_x=np.array([0.0, 1.0, 1.0]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     faces=np.array([[0, 1, 2]]),
            ...     variables={"depth": ParsedVariable(location="face", shape=(1,))},
            ... )
            >>> try:
            ...     check_shapes(empty)
            ... except ValueError as error:
            ...     print(str(error).split("; ")[1])
            it carries neither
    """
    n_node = int(np.asarray(parsed.node_x).shape[0])
    _check_node_arrays(parsed, n_node)
    counts = {
        "node": n_node,
        "face": 0 if parsed.faces is None else int(np.asarray(parsed.faces).shape[0]),
        "edge": 0 if parsed.edges is None else int(np.asarray(parsed.edges).shape[0]),
    }
    for var_name, var in parsed.variables.items():
        if var.location not in counts:
            raise ValueError(
                f"variable {var_name!r} has unknown location {var.location!r}"
            )
        if (var.values is None) == (var.loader is None):
            carried = "both" if var.values is not None else "neither"
            raise ValueError(
                f"variable {var_name!r} must carry exactly one of values (eager) or "
                f"loader (lazy); it carries {carried}"
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
        if var.active is None:
            continue
        # The mask becomes a sibling variable built straight from this array, and it is
        # synthesised after this loop, so nothing downstream would ever question it.
        active_shape = tuple(np.shape(var.active))
        if active_shape != expected:
            raise ValueError(
                f"the active mask of {var_name!r} has shape {active_shape}, which does "
                f"not match the variable it describes: expected {expected}"
            )
