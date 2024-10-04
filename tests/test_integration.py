import pytest

from dvf import GridFunction, integrate, integrate_bd


def test_can_integrate_constant_function(grid4x4):
    f = GridFunction.const(3, grid4x4)
    total = integrate(f)
    assert total == pytest.approx(16 / 9 * 3)


def test_can_integrate_nonconstant_function(grid4x4):
    f = GridFunction.from_function(lambda x, y: 1 + x * y**2, grid4x4)
    total = integrate(f)
    expected = (
        16
        + (1 / 3) ** 3
        + 2 / 3 * (1 / 3) ** 2
        + (1 / 3) ** 2
        + 1 / 3 * (2 / 3) ** 2
        + (2 / 3) ** 3
        + (2 / 3) ** 2
        + 1 / 3
        + 2 / 3
        + 1
    ) / 9

    assert total == pytest.approx(expected)


def test_can_integrate_over_boundary(grid4x4):
    f = GridFunction.from_function(lambda x, y: (x + 1) * (y + 2) ** 2, grid4x4)
    total = integrate_bd(f)

    expected = (
        2**2
        + (2 + 1 / 3) ** 2
        + (2 + 2 / 3) ** 2
        + 3**2
        + 2 * 2**2
        + 2 * (2 + 1 / 3) ** 2
        + 2 * (2 + 2 / 3) ** 2
        + 2 * 3**2
        + (1 + 1 / 3) * 2**2
        + (1 + 2 / 3) * 2**2
        + (1 + 1 / 3) * 3**2
        + (1 + 2 / 3) * 3**2
    ) / 3

    assert total == pytest.approx(expected)
