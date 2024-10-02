from collections import Counter

import numpy as np
import pytest

from dvf import Edge


def test_indices(grid3x3):
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
    }
    assert Counter(grid3x3.indices) == Counter(expected)


def test_points(grid3x3):
    x, y = grid3x3.points
    assert (x[1, 2], y[1, 2]) == (0.5, 1)

    x_expected = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ]
    ).T
    y_expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ]
    ).T
    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_point_by_index(grid3x3):
    assert grid3x3.point((1, 2)) == pytest.approx((0.5, 1.0))


def test_point_by_index_array(grid3x3):
    ii = np.array([[1, 2, 0], [2, 1, 2]])
    jj = np.array([[0, 1, 1], [2, 2, 0]])
    x, y = grid3x3.point([ii, jj])

    x_expected = np.array(
        [
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 1.0],
        ]
    )
    y_expected = np.array(
        [
            [0.0, 0.5, 0.5],
            [1.0, 1.0, 0.0],
        ]
    )

    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_negative_x_index_is_rejected(grid3x3):
    assert not grid3x3.index_valid((-1, 1))


def test_negative_y_index_is_rejected(grid3x3):
    assert not grid3x3.index_valid((1, -1))


def test_too_large_x_index_is_rejected(grid3x3):
    assert not grid3x3.index_valid((3, 1))


def test_too_large_y_index_is_rejected(grid3x3):
    assert not grid3x3.index_valid((1, 3))


def test_index_within_bounds_is_accepted(grid3x3):
    assert grid3x3.index_valid((0, 0))
    assert grid3x3.index_valid((2, 0))
    assert grid3x3.index_valid((0, 2))
    assert grid3x3.index_valid((2, 2))


def test_size_computed_correctly(grid3x3):
    assert grid3x3.size == 9


def test_can_enumerate_boundary_indices(grid3x3):
    expected = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    }
    assert Counter(grid3x3.boundary()) == Counter(expected)


def test_none_boundary_is_empty(grid3x3):
    assert Counter(grid3x3.boundary(Edge.NONE)) == Counter({})


def test_can_enumerate_top_boundary_indices(grid3x3):
    expected = {(0, 0), (1, 0), (2, 0)}
    actual = grid3x3.boundary(Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_boundary_indices(grid3x3):
    expected = {(0, 2), (1, 2), (2, 2)}
    actual = grid3x3.boundary(Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_bottom_boundary_indices(grid3x3):
    expected = {(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)}
    actual = grid3x3.boundary(Edge.TOP | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_left_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2)}
    actual = grid3x3.boundary(Edge.LEFT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_right_boundary_indices(grid3x3):
    expected = {(2, 0), (2, 1), (2, 2)}
    actual = grid3x3.boundary(Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_left_right_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)}
    actual = grid3x3.boundary(Edge.LEFT | Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_left_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)}
    actual = grid3x3.boundary(Edge.LEFT | Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_right_boundary_indices(grid3x3):
    expected = {(2, 0), (2, 1), (2, 2), (0, 0), (1, 0)}
    actual = grid3x3.boundary(Edge.RIGHT | Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_left_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}
    actual = grid3x3.boundary(Edge.LEFT | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_right_boundary_indices(grid3x3):
    expected = {(2, 0), (2, 1), (2, 2), (0, 2), (1, 2)}
    actual = grid3x3.boundary(Edge.RIGHT | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_top_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)}
    actual = grid3x3.boundary(~Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_bottom_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2), (2, 2), (2, 1), (2, 0), (1, 0)}
    actual = grid3x3.boundary(~Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_left_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)}
    actual = grid3x3.boundary(~Edge.LEFT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_right_boundary_indices(grid3x3):
    expected = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 0), (1, 0)}
    actual = grid3x3.boundary(~Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_shape(grid3x3):
    assert grid3x3.shape == (3, 3)


def test_ravel_index(grid3x3):
    assert grid3x3.ravel_index((1, 2)) == 2 * 3 + 1


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
