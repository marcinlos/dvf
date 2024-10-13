from collections import Counter

import numpy as np
import pytest

from dvf import Edge


def test_indices(grid3x4):
    expected = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    }
    assert Counter(grid3x4.indices) == Counter(expected)


def test_points(grid3x4):
    x, y = grid3x4.points
    assert (x[1, 2], y[1, 2]) == (0.5, 2 / 3)

    x_expected = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ]
    ).T
    y_expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1 / 3, 1 / 3, 1 / 3],
            [2 / 3, 2 / 3, 2 / 3],
            [1.0, 1.0, 1.0],
        ]
    ).T
    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_point_by_index(grid3x4):
    assert grid3x4.point((1, 2)) == pytest.approx((0.5, 2 / 3))


def test_point_by_index_array(grid3x4):
    ii = np.array([[1, 2, 0], [2, 1, 2]])
    jj = np.array([[0, 1, 1], [3, 2, 0]])
    x, y = grid3x4.point([ii, jj])

    x_expected = np.array(
        [
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 1.0],
        ]
    )
    y_expected = np.array(
        [
            [0.0, 1 / 3, 1 / 3],
            [1.0, 2 / 3, 0.0],
        ]
    )

    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_negative_x_index_is_rejected(grid3x4):
    assert not grid3x4.index_valid((-1, 1))


def test_negative_y_index_is_rejected(grid3x4):
    assert not grid3x4.index_valid((1, -1))


def test_too_large_x_index_is_rejected(grid3x4):
    assert not grid3x4.index_valid((3, 1))


def test_too_large_y_index_is_rejected(grid3x4):
    assert not grid3x4.index_valid((1, 4))


def test_index_within_bounds_is_accepted(grid3x4):
    assert grid3x4.index_valid((0, 0))
    assert grid3x4.index_valid((2, 0))
    assert grid3x4.index_valid((0, 3))
    assert grid3x4.index_valid((2, 3))


def test_can_check_validity_of_multiple_inddices(grid3x4):
    ii = np.array([[1, 2, 0], [2, 1, 3]])
    jj = np.array([[0, 3, 4], [-1, 2, 0]])

    actual = grid3x4.index_valid((ii, jj))
    expected = [[True, True, False], [False, True, False]]

    np.testing.assert_array_equal(actual, expected)


def test_size_computed_correctly(grid3x4):
    assert grid3x4.size == 12


def test_can_enumerate_boundary_indices(grid3x4):
    expected = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (2, 1),
        (0, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    }
    assert Counter(grid3x4.boundary()) == Counter(expected)


def test_none_boundary_is_empty(grid3x4):
    assert Counter(grid3x4.boundary(Edge.NONE)) == Counter({})


def test_can_enumerate_top_boundary_indices(grid3x4):
    expected = {(0, 0), (1, 0), (2, 0)}
    actual = grid3x4.boundary(Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_boundary_indices(grid3x4):
    expected = {(0, 3), (1, 3), (2, 3)}
    actual = grid3x4.boundary(Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_bottom_boundary_indices(grid3x4):
    expected = {(0, 0), (1, 0), (2, 0), (0, 3), (1, 3), (2, 3)}
    actual = grid3x4.boundary(Edge.TOP | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_left_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3)}
    actual = grid3x4.boundary(Edge.LEFT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_right_boundary_indices(grid3x4):
    expected = {(2, 0), (2, 1), (2, 2), (2, 3)}
    actual = grid3x4.boundary(Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_left_right_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 2), (2, 3)}
    actual = grid3x4.boundary(Edge.LEFT | Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_left_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0)}
    actual = grid3x4.boundary(Edge.LEFT | Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_right_boundary_indices(grid3x4):
    expected = {(2, 0), (2, 1), (2, 2), (2, 3), (0, 0), (1, 0)}
    actual = grid3x4.boundary(Edge.RIGHT | Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_left_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}
    actual = grid3x4.boundary(Edge.LEFT | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_right_boundary_indices(grid3x4):
    expected = {(2, 0), (2, 1), (2, 2), (2, 3), (0, 3), (1, 3)}
    actual = grid3x4.boundary(Edge.RIGHT | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_top_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0)}
    actual = grid3x4.boundary(~Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_bottom_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0)}
    actual = grid3x4.boundary(~Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_left_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0)}
    actual = grid3x4.boundary(~Edge.LEFT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_right_boundary_indices(grid3x4):
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 0), (1, 0)}
    actual = grid3x4.boundary(~Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_shape(grid3x4):
    assert grid3x4.shape == (3, 4)


def test_ravel_index(grid3x4):
    assert grid3x4.ravel_index((1, 2)) == 1 * 4 + 2


def test_adjacent_interior(grid5x5):
    expected = {(1, 2), (2, 2), (3, 2), (3, 3), (3, 4), (2, 4), (1, 4), (1, 3)}
    actual = grid5x5.adjacent((2, 3))
    assert Counter(actual) == Counter(expected)


def test_adjacent_edge(grid5x5):
    expected = {(0, 1), (1, 1), (1, 2), (1, 3), (0, 3)}
    actual = grid5x5.adjacent((0, 2))
    assert Counter(actual) == Counter(expected)


def test_adjacent_corner(grid5x5):
    expected = {(3, 0), (3, 1), (4, 1)}
    actual = grid5x5.adjacent((4, 0))
    assert Counter(actual) == Counter(expected)


def test_facet_normals(grid3x4):
    directions = {
        (0, 0): (-2, -3),
        (1, 0): (0, -1),
        (2, 0): (2, -3),
        (2, 1): (1, 0),
        (2, 2): (1, 0),
        (2, 3): (2, 3),
        (1, 3): (0, 1),
        (0, 3): (-2, 3),
        (0, 2): (-1, 0),
        (0, 1): (-1, 0),
    }
    expected = {p: v / np.linalg.norm(v) for p, v in directions.items()}
    actual = {idx: grid3x4.facet_normal(idx) for idx in expected}

    for p in directions:
        np.testing.assert_allclose(actual[p], expected[p])
