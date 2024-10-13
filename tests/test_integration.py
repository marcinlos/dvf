import numpy as np
import pytest

from dvf import GridFunction, integrate, integrate_bd


def test_can_integrate_constant_function(grid3x4):
    f = GridFunction.const(3, grid3x4)
    total = integrate(f)
    assert total == pytest.approx(12 / 6 * 3)


def test_can_integrate_nonconstant_function(grid3x4):
    f = GridFunction.from_function(lambda x, y: (1 + x) * y**2, grid3x4)
    total = integrate(f)
    expected = (
        +(1 + 0) * (1 / 3) ** 2
        + (1 + 1 / 2) * (1 / 3) ** 2
        + (1 + 1) * (1 / 3) ** 2
        + (1 + 0) * (2 / 3) ** 2
        + (1 + 1 / 2) * (2 / 3) ** 2
        + (1 + 1) * (2 / 3) ** 2
        + (1 + 0) * 1**2
        + (1 + 1 / 2) * 1**2
        + (1 + 1) * 1**2
    ) / (2 * 3)

    assert total == pytest.approx(expected)


def test_can_integrate_over_boundary(grid3x4):
    f = GridFunction.from_function(lambda x, y: (x + 1) * (y + 2) ** 2, grid3x4)
    total = integrate_bd(f)
    hx = 1 / 2
    hy = 1 / 3
    h_corner = np.hypot(hx, hy)

    expected = (
        2**2 * h_corner
        + (2 + 1 / 3) ** 2 * hy
        + (2 + 2 / 3) ** 2 * hy
        + 3**2 * h_corner
        + 2 * 2**2 * h_corner
        + 2 * (2 + 1 / 3) ** 2 * hy
        + 2 * (2 + 2 / 3) ** 2 * hy
        + 2 * 3**2 * h_corner
        + (1 + 1 / 2) * 2**2 * hx
        + (1 + 1 / 2) * 3**2 * hx
    )

    assert total == pytest.approx(expected)
