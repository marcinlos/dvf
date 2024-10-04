import numpy as np
import pytest

from dvf import GridFunction, diff


def assert_function_values_equal(actual, expected):
    __tracebackhide__ = True
    np.testing.assert_allclose(actual.tabulate(), expected)


@pytest.fixture
def function_values():
    return np.array(
        [
            [1, 5, 2, 4],
            [1, 3, 8, 2],
            [3, 1, 1, 5],
            [1, 7, 1, 2],
        ]
    ).T


def test_can_compute_dx_plus(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    df = diff(f, "x", "+")
    h = 1 / 3
    expected = (
        np.array(
            [
                [5 - 1, 2 - 5, 4 - 2, 0],
                [3 - 1, 8 - 3, 2 - 8, 0],
                [1 - 3, 1 - 1, 5 - 1, 0],
                [7 - 1, 1 - 7, 2 - 1, 0],
            ]
        ).T
        / h
    )

    assert_function_values_equal(df, expected)


def test_can_compute_dx_minus(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    df = diff(f, "x", "-")
    h = 1 / 3
    expected = (
        np.array(
            [
                [0, 5 - 1, 2 - 5, 4 - 2],
                [0, 3 - 1, 8 - 3, 2 - 8],
                [0, 1 - 3, 1 - 1, 5 - 1],
                [0, 7 - 1, 1 - 7, 2 - 1],
            ]
        ).T
        / h
    )

    assert_function_values_equal(df, expected)


def test_can_compute_dy_plus(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    df = diff(f, "y", "+")
    h = 1 / 3
    expected = (
        np.array(
            [
                [1 - 1, 3 - 5, 8 - 2, 2 - 4],
                [3 - 1, 1 - 3, 1 - 8, 5 - 2],
                [1 - 3, 7 - 1, 1 - 1, 2 - 5],
                [0, 0, 0, 0],
            ]
        ).T
        / h
    )

    assert_function_values_equal(df, expected)


def test_can_compute_dy_minus(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    df = diff(f, "y", "-")
    h = 1 / 3
    expected = (
        np.array(
            [
                [0, 0, 0, 0],
                [1 - 1, 3 - 5, 8 - 2, 2 - 4],
                [3 - 1, 1 - 3, 1 - 8, 5 - 2],
                [1 - 3, 7 - 1, 1 - 1, 2 - 5],
            ]
        ).T
        / h
    )

    assert_function_values_equal(df, expected)
