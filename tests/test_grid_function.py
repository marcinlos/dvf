import numpy as np
import pytest

from dvf import Grid, GridFunction


@pytest.fixture
def grid4x4():
    return Grid(3)


def assert_functions_equal(actual, expected):
    __tracebackhide__ = True
    np.testing.assert_allclose(actual.tabulate(), expected.tabulate())


def test_can_create_from_coords_function(grid4x4):
    def fun(x, y):
        return x**2 - y

    f = GridFunction.from_function(fun, grid4x4)
    assert f(2, 1) == (2 / 3) ** 2 - 1 / 3


def test_can_tabulate_values(grid4x4):
    def fun(i, j):
        return i + 2 * j

    f = GridFunction(fun, grid4x4)
    expected = np.array(
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5, 6, 7],
            [6, 7, 8, 9],
        ]
    ).T
    actual = f.tabulate()

    np.testing.assert_allclose(actual, expected)


def test_can_create_from_array(grid4x4):
    data = np.arange(16).reshape(4, 4)
    f = GridFunction.from_array(data, grid4x4)

    actual = f.tabulate()

    np.testing.assert_allclose(actual, data)


def test_can_add_grid_functions(grid4x4):
    f = GridFunction.from_function(lambda x, y: x**2 - y, grid4x4)
    g = GridFunction.from_function(lambda x, y: x + 2 * y**2, grid4x4)
    s = f + g

    h = GridFunction.from_function(lambda x, y: x**2 - y + x + 2 * y**2, grid4x4)

    assert_functions_equal(s, h)


def test_can_add_constant_to_grid_function(grid4x4):
    f = GridFunction.from_function(lambda x, y: x**2 - y, grid4x4)
    g = 2
    s = f + g

    h = GridFunction.from_function(lambda x, y: x**2 - y + 2, grid4x4)

    assert_functions_equal(s, h)


def test_can_add_grid_function_to_constant(grid4x4):
    f = 2
    g = GridFunction.from_function(lambda x, y: x**2 - y, grid4x4)
    s = f + g

    h = GridFunction.from_function(lambda x, y: x**2 - y + 2, grid4x4)

    assert_functions_equal(s, h)


def test_can_subtract_grid_functions(grid4x4):
    f = GridFunction.from_function(lambda x, y: x**2 - y, grid4x4)
    g = GridFunction.from_function(lambda x, y: x + 2 * y**2, grid4x4)
    s = f - g

    h = GridFunction.from_function(lambda x, y: x**2 - y - x - 2 * y**2, grid4x4)

    assert_functions_equal(s, h)


def test_can_subtract_constant_from_grid_functions(grid4x4):
    f = GridFunction.from_function(lambda x, y: x**2 - y, grid4x4)
    g = 2
    s = f - g

    h = GridFunction.from_function(lambda x, y: x**2 - y - 2, grid4x4)

    assert_functions_equal(s, h)


def test_can_subtract_grid_function_from_constant(grid4x4):
    f = 2
    g = GridFunction.from_function(lambda x, y: x**2 - y, grid4x4)
    s = f - g

    h = GridFunction.from_function(lambda x, y: 2 - x**2 + y, grid4x4)

    assert_functions_equal(s, h)


def test_can_multiply_grid_functions(grid4x4):
    f = GridFunction.from_function(lambda x, y: x - y, grid4x4)
    g = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    s = f * g

    h = GridFunction.from_function(lambda x, y: x**2 - y**2, grid4x4)

    assert_functions_equal(s, h)


def test_can_multiply_grid_function_by_constant(grid4x4):
    f = GridFunction.from_function(lambda x, y: x - y, grid4x4)
    g = 2
    s = f * g

    h = GridFunction.from_function(lambda x, y: 2 * (x - y), grid4x4)

    assert_functions_equal(s, h)


def test_can_multiply_constant_by_grid_function(grid4x4):
    f = 2
    g = GridFunction.from_function(lambda x, y: x - y, grid4x4)
    s = f * g

    h = GridFunction.from_function(lambda x, y: 2 * (x - y), grid4x4)

    assert_functions_equal(s, h)


def test_can_divide_grid_functions(grid4x4):
    f = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    g = GridFunction.from_function(lambda x, y: 1 + x * y, grid4x4)
    s = f / g

    h = GridFunction.from_function(lambda x, y: (x + y) / (1 + x * y), grid4x4)

    assert_functions_equal(s, h)


def test_can_divide_grid_function_by_constant(grid4x4):
    f = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    g = 2
    s = f / g

    h = GridFunction.from_function(lambda x, y: (x + y) / 2, grid4x4)

    assert_functions_equal(s, h)


def test_can_divide_constant_by_grid_function(grid4x4):
    f = 1
    g = GridFunction.from_function(lambda x, y: 1 + x * y, grid4x4)
    s = f / g

    h = GridFunction.from_function(lambda x, y: 1 / (1 + x * y), grid4x4)

    assert_functions_equal(s, h)


def test_can_raise_grid_function_to_grid_function_power(grid4x4):
    f = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    g = GridFunction.from_function(lambda x, y: 1 + x * y, grid4x4)
    s = f**g

    h = GridFunction.from_function(lambda x, y: (x + y) ** (1 + x * y), grid4x4)

    assert_functions_equal(s, h)


def test_can_raise_grid_function_to_constant_power(grid4x4):
    f = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    g = 2
    s = f**g

    h = GridFunction.from_function(lambda x, y: (x + y) ** 2, grid4x4)

    assert_functions_equal(s, h)


def test_can_raise_constant_to_grid_function_power(grid4x4):
    f = 2
    g = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    s = f**g

    h = GridFunction.from_function(lambda x, y: 2 ** (x + y), grid4x4)

    assert_functions_equal(s, h)


def test_can_negate_grid_function(grid4x4):
    f = GridFunction.from_function(lambda x, y: x + y, grid4x4)
    s = -f

    h = GridFunction.from_function(lambda x, y: -x - y, grid4x4)

    assert_functions_equal(s, h)


def test_can_take_abs_of_grid_function(grid4x4):
    f = GridFunction.from_function(lambda x, y: x - y, grid4x4)
    s = abs(f)

    h = GridFunction.from_function(lambda x, y: abs(x - y), grid4x4)

    assert_functions_equal(s, h)


def test_coord_functions(grid4x4):
    x, y = GridFunction.coords(grid4x4)

    xx = GridFunction.from_function(lambda x, y: x, grid4x4)
    yy = GridFunction.from_function(lambda x, y: y, grid4x4)

    assert_functions_equal(x, xx)
    assert_functions_equal(y, yy)


def test_cannot_combine_grid_functions_from_different_grids(grid4x4):
    other_grid = Grid(3)
    f = GridFunction.const(3, grid4x4)
    g = GridFunction.const(2, other_grid)

    with pytest.raises(ValueError, match="Grids do not match"):
        f + g
