"""Tests for the shared invariant enforcer (S3).

Each invariant pyramids assumes fails *silently* -- a driver that forgets one yields
wrong numbers, not an error -- so each is proved here with a case that would pass
whether or not the enforcement ran, and an assertion that fails if it is reverted.
"""

import pyramids  # noqa: F401  # isort:skip

from pathlib import Path

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


@pytest.mark.parametrize("field", ["face_x", "face_y"])
def test_face_centre_length_mismatch_raises(field):
    """Face-centre coordinates must match the face count, whichever array is wrong.

    Both labels run through one loop, and only ``face_x`` was ever exercised, so the
    tuple could have named it twice and ``face_y`` would have gone unchecked -- exactly
    the asymmetry the ``node_y``/``node_z`` case is parametrised against.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        **{field: np.array([0.5, 0.5])},
    )

    message = f"{field} has 2 values but there are 1 faces"
    with pytest.raises(ValueError, match=message):
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


@pytest.mark.parametrize(
    "row, real",
    [
        pytest.param([0, 1, -1], 2, id="two-real-vertices"),
        pytest.param([-1, -1, -1], 0, id="nothing-but-padding"),
    ],
)
def test_row_with_fewer_than_three_real_vertices_raises(row, real):
    """A row holding fewer than three real nodes is not a face, so it is refused.

    It was previously skipped, which meant it reached ``Mesh2d`` uncounted by either
    tally and invisible to every check. A 1D topology belongs in ``edges`` or
    ``network_edges``; in ``faces`` it is a driver bug.

    The all-padding row is the arithmetic edge of the interior-padding rule that runs
    first: its first pad sits at column 0 and it has no real vertices, so ``0 < 0``
    must not fire and the row has to be refused for its width instead. A rule written
    ``<=`` would blame the padding and never report the real fault.
    """
    faces = np.array([row], dtype=np.int64)
    node_x = np.array([0.0, 1.0, 1.0])
    node_y = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match=f"holds {real} real node"):
        fix_winding(faces, node_x, node_y, -1)


def test_interior_padding_is_caught_even_on_a_short_row():
    """The padding check is a row-shape rule, so it applies before the width rule."""
    faces = np.array([[0, -1, 1]], dtype=np.int64)
    node_x = np.array([0.0, 1.0, 1.0])
    node_y = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="pads between real vertices"):
        fix_winding(faces, node_x, node_y, -1)


def test_two_dimensional_parse_without_faces_raises():
    """A 2D topology with no faces is a contradiction, refused where it is made.

    Left alone it was not even a loud failure: ``read`` returned a ``UgridDataset``
    whose connectivity was a 0-d object array, and the ``IndexError`` surfaced later
    when the caller touched ``n_face`` -- far from the driver that caused it.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=None,
    )

    with pytest.raises(ValueError, match="topology_dimension 2 but no faces"):
        validate_and_normalise(parsed)


def test_edge_variable_validates_against_a_network_topology():
    """An edge variable is counted against whichever edge topology the mesh has.

    A driver reporting a 1D topology puts its edges in ``network_edges``, so counting
    only ``edges`` made a correct edge variable impossible to validate there.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        faces=None,
        network_edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
        topology_dimension=1,
        variables={
            "q": ParsedVariable(location="edge", shape=(2,), values=np.zeros(2))
        },
    )

    check_shapes(parsed)  # does not raise


@pytest.mark.parametrize("field", ["edges", "network_edges"])
def test_edge_connectivity_must_be_two_columns_wide(field):
    """An edge joins exactly two nodes, so a wider array is refused."""
    fields = {
        "node_x": np.array([0.0, 1.0, 1.0]),
        "node_y": np.array([0.0, 0.0, 1.0]),
        # An edge-only mesh is one-dimensional, and saying so is now required.
        "topology_dimension": 1,
        field: np.array([[0, 1, 2]], dtype=np.int64),
    }

    with pytest.raises(ValueError, match="rows are 3 wide"):
        validate_and_normalise(ParsedMesh(**fields))


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
    faces = np.array([[0, 1, 2]], dtype=np.int64) if label != "faces" else None
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=faces,
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


@pytest.mark.parametrize(
    "fill_value, collides",
    [
        pytest.param(3, True, id="the-last-valid-index"),
        pytest.param(4, False, id="one-past-the-last-valid-index"),
    ],
)
def test_the_fill_collision_window_ends_at_the_last_valid_index(fill_value, collides):
    """The window is ``start_index <= fill < start_index + n_node`` -- both bounds exact.

    With base 1 and three nodes the file's valid ids are 1, 2 and 3, so a fill of
    ``3`` cannot be told from node 3 and is unrecoverable, while ``4`` is outside the
    numbering and is real padding. Every other collision case in the suite is
    ``fill_value=0`` at base 0, which pins the lower bound only: an upper bound
    written ``start_index + n_node - 1`` would accept a fill that shadows the last
    node of every 1-based file and pad the mesh with it.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[1, 2, 3]], dtype=np.int64),
        fill_value=fill_value,
        start_index=1,
    )

    if collides:
        with pytest.raises(ValueError, match="collides with a valid index"):
            validate_and_normalise(parsed)
        return

    np.testing.assert_array_equal(validate_and_normalise(parsed).faces, [[0, 1, 2]])


@pytest.mark.parametrize(
    "faces",
    [
        pytest.param(np.array([0, 1, 2], dtype=np.int64), id="one-dimensional"),
        pytest.param(np.zeros((1, 1, 3), dtype=np.int64), id="three-dimensional"),
    ],
)
def test_connectivity_of_the_wrong_rank_is_refused(faces):
    """Rectangularity is a rank check as well as a raggedness check.

    The guard reads ``dtype == object or ndim != 2``, and every existing case arrives
    through the raggedness half -- a list numpy refuses to stack, or an object-dtype
    array. A flat row vector is the other way a parser gets this wrong, and because
    coverage records the ``if`` rather than each operand, the rank half could be
    deleted with the suite still green at 100% branch coverage.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=faces,
    )

    with pytest.raises(ValueError, match="rectangular 2-D array"):
        validate_and_normalise(parsed)


def test_reversing_a_padded_face_leaves_its_padding_at_the_tail():
    """A reversal rewrites a row's real prefix only, so the pad stays a trailing run.

    Reversal is documented to assume UGRID's trailing-pad convention, and every
    reversal case in the suite is an unpadded triangle. A reversal written over the
    whole row -- ``row[:] = row[::-1]`` -- passes all of them, yet on the quad of a
    mixed-width mesh it moves the padding to the front and breaks the one invariant
    the interior-padding check exists to defend.
    """
    faces = np.array([[0, 1, 2, 3, -1]], dtype=np.int64)
    node_x = np.array([0.0, 0.0, 1.0, 1.0])
    node_y = np.array([0.0, 1.0, 1.0, 0.0])

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    np.testing.assert_array_equal(fixed, [[3, 2, 1, 0, -1]])
    assert (reversed_count, degenerate_count) == (1, 0)
    assert fixed[0, -1] == -1, f"the pad moved: {fixed[0].tolist()}"


def test_every_face_row_is_counted_in_exactly_one_group():
    """The two tallies are per-row, independent, and leave a correct face untouched.

    ``fix_winding`` documents that each row lands in exactly one of three groups --
    reversed, degenerate, or already counter-clockwise -- and each outcome is proved
    on its own single-row array. A counter incremented in the wrong arm, or once per
    call rather than once per row, would satisfy every one of those tests.
    """
    node_x = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    faces = np.array([[0, 1, 2], [0, 2, 1], [3, 4, 5]], dtype=np.int64)

    fixed, reversed_count, degenerate_count = fix_winding(faces, node_x, node_y, -1)

    assert (reversed_count, degenerate_count) == (1, 1)
    np.testing.assert_array_equal(fixed[0], [0, 1, 2])
    np.testing.assert_array_equal(fixed[1], [1, 2, 0])
    np.testing.assert_array_equal(fixed[2], [3, 4, 5])


@pytest.mark.parametrize(
    "faces, node_x, node_y",
    [
        pytest.param([[0, 0, 1]], [0.0, 1.0], [0.0, 1.0], id="repeated-node"),
        pytest.param(
            [[0, 1, 2]],
            [5.0, 5.0, 5.0],
            [7.0, 7.0, 7.0],
            id="collapsed-to-one-point",
        ),
    ],
)
def test_a_face_without_extent_is_degenerate(faces, node_x, node_y):
    """A repeated node, and a face collapsed to a point, have no winding either.

    ``fix_winding`` names "a repeated node" beside collinearity and only collinearity
    was proved. The collapsed face is also the tolerance's own boundary: its span is
    zero, so the threshold is exactly ``0.0`` and the comparison has to be ``<=``. A
    ``<`` would fall through to the sign test and count the face as already
    counter-clockwise, so neither tally would ever mention it.
    """
    fixed, reversed_count, degenerate_count = fix_winding(
        np.array(faces, dtype=np.int64), np.array(node_x), np.array(node_y), -1
    )

    assert (reversed_count, degenerate_count) == (0, 1)
    np.testing.assert_array_equal(fixed, faces)


def test_normalisation_warnings_blame_the_calling_frame():
    """Both warnings use ``stacklevel=2``, so each names the caller's own line.

    Left at the default, a warning points inside ``_validate.py`` and tells a reader
    nothing about which read produced the mesh. Every warning in the package sets the
    level, the whole suite executes those lines, and no test looked at where they were
    attributed.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 1.0, 0.0, 2.0, 2.0, 2.0]),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )

    with pytest.warns(UserWarning) as record:
        validate_and_normalise(parsed)

    blamed = {Path(warning.filename).name for warning in record}
    assert blamed == {Path(__file__).name}, f"warnings blamed {blamed}"
    assert len(record) == 2, "a reversal and a degeneracy are reported separately"


def test_the_active_mask_of_a_temporal_variable_is_sized_over_time_too():
    """A mask is measured against its variable's full shape, time axis included.

    The only rejected mask in the suite belongs to a static variable, so a check
    written against the element count alone would accept a single-step mask for a
    three-step variable. The mask becomes a sibling variable built straight from this
    array after the loop, so nothing downstream would question it.
    """
    parsed = ParsedMesh(
        node_x=np.array([0.0, 1.0, 1.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        variables={
            "wl": ParsedVariable(
                location="face",
                shape=(3, 1),
                values=np.zeros((3, 1)),
                time_values=np.array([0.0, 1.0, 2.0]),
                active=np.ones((1,), dtype=np.int8),
            )
        },
    )

    with pytest.raises(ValueError, match=r"expected \(3, 1\)"):
        check_shapes(parsed)
