"""Tests for the shared invariant enforcer (S3).

Each invariant pyramids assumes fails *silently* -- a driver that forgets one yields
wrong numbers, not an error -- so each is proved here with a case that would pass
whether or not the enforcement ran, and an assertion that fails if it is reverted.
"""

import pyramids  # noqa: F401  # isort:skip

import numpy as np
import pytest

from pyramids_mesh._base import ParsedMesh, ParsedVariable
from pyramids_mesh._validate import (
    check_shapes,
    fix_winding,
    repad,
    to_zero_based,
    validate_and_normalise,
)


def _signed_area2(nodes_x, nodes_y, face):
    """Twice the signed area of a face's real (non-``-1``) vertices."""
    real = face[face != -1]
    xs, ys = nodes_x[real], nodes_y[real]
    return float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def test_one_based_input_becomes_zero_based():
    """A 1-based file is re-based to 0, and the base a writer emits is kept."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[1, 2, 3]], dtype=np.int64),
        start_index=1,
    )

    result = validate_and_normalise(parsed)

    np.testing.assert_array_equal(result.faces, np.array([[0, 1, 2]]))
    assert result.original_start_index == 1


def test_fill_is_not_shifted_by_start_index():
    """Padding is masked before the subtraction, so ``-1`` never becomes ``-2``."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        faces=np.array([[1, 2, 3, -1], [1, 3, 4, -1]], dtype=np.int64),
        fill_value=-1,
        start_index=1,
    )

    result = validate_and_normalise(parsed)

    assert result.faces[0, 3] == -1
    assert result.faces[1, 3] == -1
    assert result.faces[:, :3].min() == 0


def test_out_of_range_index_raises_not_clipped():
    """An index past the last node means the base was wrong -- raise, never clip."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[1, 2, 9]], dtype=np.int64),
        start_index=1,
    )

    with pytest.raises(ValueError, match="out of range"):
        validate_and_normalise(parsed)


def test_fill_value_zero_on_zero_based_raises():
    """``fill_value=0`` on 0-based data is unrecoverable: real 0 and fill collide."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        fill_value=0,
        start_index=0,
    )

    with pytest.raises(ValueError, match="collide"):
        validate_and_normalise(parsed)


def test_clockwise_face_is_reversed():
    """A clockwise face comes back counter-clockwise (positive signed area)."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 1.0, 0.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    assert _signed_area2(parsed.node_x, parsed.node_y, parsed.faces[0]) < 0

    result = validate_and_normalise(parsed)

    assert _signed_area2(result.node_x, result.node_y, result.faces[0]) > 0


def test_shoelace_ignores_fill_padding():
    """A padded counter-clockwise face is left alone; ``-1`` never enters the area."""
    faces = np.array([[0, 1, 2, -1]], dtype=np.int64)
    node_x = np.array([0.0, 1.0, 1.0])
    node_y = np.array([0.0, 0.0, 1.0])

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    assert reversed_count == 0
    assert degenerate_count == 0
    np.testing.assert_array_equal(fixed, faces)


def test_degenerate_face_is_counted_not_reversed_not_raised():
    """A zero-area (collinear) face has no winding: kept, counted, warned -- not raised."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )

    with pytest.warns(UserWarning, match="degenerate"):
        result = validate_and_normalise(parsed)

    np.testing.assert_array_equal(result.faces, np.array([[0, 1, 2]]))

    _, reversed_count, degenerate_count = fix_winding(
        parsed.faces, parsed.node_x, parsed.node_y, -1
    )
    assert reversed_count == 0
    assert degenerate_count == 1


def test_numerically_degenerate_sliver_is_not_reversed():
    """A sliver whose area is zero to within rounding is degenerate, not clockwise.

    An exact ``area2 == 0`` test decided such a face on its sign bit alone, reversing
    one that landed a few ULP below zero and leaving its mirror image alone.
    """
    node_x = np.array([0.0, 1.0, 2.0])
    node_y = np.array([0.0, 1e-17, 0.0])
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    np.testing.assert_array_equal(fixed, faces)
    assert reversed_count == 0
    assert degenerate_count == 1


def test_small_clockwise_face_at_projected_coordinates_is_reversed():
    """A metre-scale element far from the origin is wound, not called degenerate.

    This is the realistic case: hydraulic meshes live in projected coordinates. Summing
    the shoelace over absolute coordinates cancels (products ~1e13 against an area of
    ~1e-2), and a tolerance scaled by absolute magnitude reached ~0.09 at a northing of
    7e6 -- so a real clockwise 0.1 m element was declared degenerate and left clockwise.
    """
    node_x = np.array([0.0, 0.1, 0.1])
    node_y = np.array([7.0e6, 7.0e6 + 0.1, 7.0e6])
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    assert (reversed_count, degenerate_count) == (1, 0)
    assert _signed_area2(node_x, node_y, fixed[0]) > 0


def test_collinear_face_at_projected_coordinates_is_degenerate_not_reversed():
    """A collinear face stays degenerate when written at projected coordinates.

    The mirror of the reversal test, and the direction a span-only tolerance got wrong.
    Coordinates near 7e6 resolve to about 4.7e-10, so three collinear points do not
    *store* as collinear: their signed area is ~9.3e-10 of real, irreducible area. A
    tolerance that ignores the coordinate magnitude therefore reads it as clockwise and
    reverses it, and `degenerate_count` never fires on any mesh in a projected CRS.
    """
    node_x = 500000.0 + np.array([0.0, 1.0, 2.0])
    node_y = 7000000.0 + np.array([0.1, 0.2, 0.3])
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    assert (reversed_count, degenerate_count) == (0, 1)
    np.testing.assert_array_equal(fixed, faces)


def test_area_is_measured_the_same_wherever_the_mesh_sits():
    """The signed area of a face does not change when the mesh is translated."""
    xs = np.array([0.0, 1.0, 1.0])
    ys = np.array([0.0, 1.0, 0.0])

    at_origin = fix_winding(np.array([[0, 1, 2]]), xs, ys, -1)
    far_away = fix_winding(np.array([[0, 1, 2]]), xs + 5.0e5, ys + 7.0e6, -1)

    assert at_origin[1:] == far_away[1:] == (1, 0)


def test_padding_between_real_vertices_raises():
    """Interior padding breaks the trailing-pad convention, so it is refused."""
    faces = np.array([[0, -1, 1, 2]], dtype=np.int64)
    node_x = np.array([0.0, 1.0, 1.0])
    node_y = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="pads between real vertices"):
        fix_winding(faces, node_x, node_y, -1)


def test_reversing_a_face_is_reported():
    """A reversal changes what a writer emits, so it is warned about, not silent."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 1.0, 0.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )

    with pytest.warns(UserWarning, match="clockwise face"):
        validate_and_normalise(parsed)


def test_shape_mismatch_raises_without_loading_values():
    """A wrong shape is caught from ``shape`` alone, without touching the loader."""
    loaded = []

    def _boom():
        loaded.append(True)
        return np.zeros(5)

    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        variables={"wl": ParsedVariable(location="face", shape=(5,), loader=_boom)},
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_and_normalise(parsed)
    assert loaded == []


def test_ragged_list_of_lists_raises_the_documented_error():
    """A ragged plain list -- what a text parser naturally produces -- is named as such.

    numpy's own message for an inhomogeneous shape says nothing about which array of
    whose driver is wrong, so the rectangularity check runs before anything touches it.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        faces=[[0, 1, 2], [0, 2, 3, 1]],
    )

    with pytest.raises(ValueError, match="rectangular"):
        validate_and_normalise(parsed)


@pytest.mark.parametrize("field", ["node_y", "node_z"])
def test_per_node_array_length_mismatch_raises(field):
    """A per-node array shorter than ``node_x`` is caught, named, before any indexing."""
    fields = {
        "node_x": np.array([0.0, 1.0, 1.0]),
        "node_y": np.array([0.0, 0.0, 1.0]),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }
    fields[field] = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match=f"{field} has 2 values but node_x has 3"):
        validate_and_normalise(ParsedMesh(**fields))


def test_face_centre_length_mismatch_raises():
    """Face-centre coordinates must match the face count."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        face_x=np.array([0.5, 0.5]),
    )

    with pytest.raises(ValueError, match="face_x has 2 values but there are 1 faces"):
        validate_and_normalise(parsed)


@pytest.mark.parametrize(
    "kwargs, carried",
    [
        ({}, "neither"),
        ({"values": np.zeros(1), "loader": lambda: np.zeros(1)}, "both"),
    ],
)
def test_variable_must_carry_exactly_one_data_source(kwargs, carried):
    """A variable with no data source, or two, is rejected rather than built."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        variables={"wl": ParsedVariable(location="face", shape=(1,), **kwargs)},
    )

    with pytest.raises(ValueError, match=f"it carries {carried}"):
        validate_and_normalise(parsed)


def test_ragged_rows_raise():
    """A ragged (object-dtype) connectivity array is a driver bug, so it raises."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0]),
        faces=np.array([np.array([0, 1, 2]), np.array([0, 2, 3, 1])], dtype=object),
    )

    with pytest.raises(ValueError, match="rectangular"):
        validate_and_normalise(parsed)


def test_to_zero_based_and_repad_helpers():
    """The two rebasing helpers behave as the enforcer relies on."""
    conn = np.array([[1, 2, 3, 0], [1, 3, 4, 0]], dtype=np.int64)

    rebased = to_zero_based(conn, start_index=1, fill_value=0)
    np.testing.assert_array_equal(rebased, np.array([[0, 1, 2, -1], [0, 2, 3, -1]]))

    repadded = repad(np.array([[0, 1, 9], [2, 3, 9]]), old_fill=9, new_fill=-1)
    np.testing.assert_array_equal(repadded, np.array([[0, 1, -1], [2, 3, -1]]))


def test_validation_leaves_the_copy_self_consistent():
    """After re-basing, ``start_index`` is 0 -- the copy describes its own state."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[1, 2, 3]], dtype=np.int64),
        start_index=1,
    )

    result = validate_and_normalise(parsed)

    np.testing.assert_array_equal(result.faces, [[0, 1, 2]])
    assert result.start_index == 0
    assert result.original_start_index == 1


def test_validation_is_idempotent():
    """Validating an already-validated mesh is a no-op, not a double re-basing."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[1, 2, 3]], dtype=np.int64),
        start_index=1,
    )

    once = validate_and_normalise(parsed)
    twice = validate_and_normalise(once)

    np.testing.assert_array_equal(twice.faces, once.faces)
    assert twice.start_index == 0
    assert twice.original_start_index == 1


def test_explicit_writer_base_survives_validation():
    """A driver that resolved ids itself keeps the base it declared for a writer."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        start_index=0,
        original_start_index=1,
    )

    result = validate_and_normalise(parsed)

    assert result.start_index == 0
    assert result.original_start_index == 1


def test_unknown_variable_location_raises():
    """A location that is not node/face/edge is rejected, not silently ignored."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        variables={
            "t": ParsedVariable(location="volume", shape=(1,), values=np.zeros(1))
        },
    )

    with pytest.raises(ValueError, match="unknown location"):
        check_shapes(parsed)


def test_winding_skips_rows_with_fewer_than_three_real_vertices():
    """A row holding fewer than three real nodes is not a polygon, so it is skipped."""
    faces = np.array([[0, 1, -1]], dtype=np.int64)
    node_x = np.array([0.0, 1.0, 1.0])
    node_y = np.array([0.0, 0.0, 1.0])

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    np.testing.assert_array_equal(fixed, faces)
    assert reversed_count == 0
    assert degenerate_count == 0


def test_validate_leaves_the_input_untouched():
    """Normalisation returns a corrected copy; the driver's own arrays are unchanged."""
    faces = np.array([[1, 2, 3]], dtype=np.int64)
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=faces,
        start_index=1,
    )

    result = validate_and_normalise(parsed)

    np.testing.assert_array_equal(faces, [[1, 2, 3]])
    assert result.faces is not faces
    assert result.fill_value == -1


@pytest.mark.parametrize("label", ["edges", "network_edges"])
def test_out_of_range_error_names_the_array_at_fault(label):
    """The range check names which connectivity is wrong, not always "faces".

    All three arrays share one checker, so a label threaded through the wrong argument
    would send a driver author looking at the faces of a file whose edges are broken.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        **{label: np.array([[0, 9]], dtype=np.int64)},
    )

    with pytest.raises(ValueError, match=f"{label} index out of range"):
        validate_and_normalise(parsed)


@pytest.mark.parametrize("label", ["edges", "network_edges"])
def test_ragged_error_names_the_array_at_fault(label):
    """Rectangularity is checked on all three arrays, each named for itself."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        **{label: [[0, 1], [1]]},
    )

    with pytest.raises(ValueError, match=f"{label} connectivity must be a rectangular"):
        validate_and_normalise(parsed)


def test_edge_located_variable_shape_is_checked_against_the_edge_count():
    """An edge variable is sized against the edges, not the faces or the nodes.

    ``check_shapes`` counts all three locations; only the node and face counts were
    ever exercised, so an edge variable compared against the wrong count would pass.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        variables={
            "flux": ParsedVariable(location="edge", shape=(5,), values=np.zeros(5))
        },
    )

    with pytest.raises(ValueError, match=r"its edge count: expected \(2,\)"):
        check_shapes(parsed)


def test_an_explicit_zero_writer_base_is_not_overwritten():
    """``original_start_index=0`` is a declared base, not a missing one.

    The resolution must test ``is not None``; a truthiness test would treat a declared
    0-based writer base as unset and record the file's own 1 instead, which is the one
    value where "the base a writer should emit" and "the base the file used" differ and
    nothing else in the suite distinguishes them.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[1, 2, 3]], dtype=np.int64),
        start_index=1,
        original_start_index=0,
    )

    result = validate_and_normalise(parsed)

    np.testing.assert_array_equal(result.faces, [[0, 1, 2]])
    assert result.start_index == 0
    assert result.original_start_index == 0


@pytest.mark.parametrize("origin", [0.0, 1e3, 1e6, 1e7])
def test_an_ordinary_face_at_projected_coordinates_is_still_wound(origin):
    """The degeneracy tolerance scales with the coordinates but must not swallow a face.

    The tolerance is ``8 * eps * span * max(span, magnitude)``, so at projected
    magnitudes it is no longer negligible. A real 100 m element offset to a UTM-sized
    origin has to keep being reversed rather than reported as having no winding at all.
    """
    node_x = np.array([0.0, 100.0, 100.0]) + origin
    node_y = np.array([0.0, 100.0, 0.0]) + origin
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    np.testing.assert_array_equal(fixed, [[2, 1, 0]])
    assert reversed_count == 1
    assert degenerate_count == 0


def test_check_shapes_accepts_temporal_variable():
    """A temporal variable's shape is ``(len(time), element_count)``."""
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        variables={
            "wl": ParsedVariable(
                location="face",
                shape=(2, 1),
                values=np.zeros((2, 1)),
                time_values=np.array([0.0, 1.0]),
            )
        },
    )

    check_shapes(parsed)  # does not raise
