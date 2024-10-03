import numpy as np
import pytest

from dvf import GridFunction, shift


def assert_function_values_equal(actual, expected):
    __tracebackhide__ = True
    np.testing.assert_allclose(actual.tabulate(), expected)


@pytest.fixture
def function_values():
    return np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    ).T


def test_can_shift_function_along_x(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    tau_f = shift(f, "x")
    expected = np.array(
        [
            [2, 3, 4, 0],
            [6, 7, 8, 0],
            [10, 11, 12, 0],
            [14, 15, 16, 0],
        ]
    ).T

    assert_function_values_equal(tau_f, expected)


def test_can_shift_function_along_x_by_custom_offset(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    tau_f = shift(f, "x", offset=-2)
    expected = np.array(
        [
            [0, 0, 1, 2],
            [0, 0, 5, 6],
            [0, 0, 9, 10],
            [0, 0, 13, 14],
        ]
    ).T

    assert_function_values_equal(tau_f, expected)


def test_can_shift_function_along_y(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    tau_f = shift(f, "y")
    expected = np.array(
        [
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [0, 0, 0, 0],
        ]
    ).T

    assert_function_values_equal(tau_f, expected)


def test_can_shift_function_along_y_by_custom_offset(grid4x4, function_values):
    f = GridFunction.from_array(function_values, grid4x4)
    tau_f = shift(f, "y", offset=-2)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    ).T

    assert_function_values_equal(tau_f, expected)
