import numpy as np
import pytest

from dvf import GridFunction, norm, product


@pytest.fixture
def f(grid4x4):
    data = np.array(
        [
            [1, 5, 2, 4],
            [1, 3, 8, 2],
            [3, 1, 1, 5],
            [1, 7, 1, 2],
        ]
    ).T
    return GridFunction.from_array(data, grid4x4)


@pytest.fixture
def g(grid4x4):
    data = np.array(
        [
            [2, 1, -2, 0],
            [5, 3, 1, 2],
            [-2, 0, 1, 3],
            [4, 4, 2, 3],
        ]
    ).T
    return GridFunction.from_array(data, grid4x4)


def test_can_compute_h_product(f, g):
    actual = product(f, g, "h")
    expected = f.grid.cell_volume * (
        1 * 2
        + 5 * 1
        + 2 * (-2)
        + 4 * 0
        + 1 * 5
        + 3 * 3
        + 8 * 1
        + 2 * 2
        + 3 * (-2)
        + 1 * 0
        + 1 * 1
        + 5 * 3
        + 1 * 4
        + 4 * 7
        + 1 * 2
        + 2 * 3
    )
    assert actual == pytest.approx(expected)


def test_can_compute_grad_h_product(f, g):
    actual = product(f, g, "grad_h")
    expected = (
        (5 - 1) * (1 - 2)
        + (2 - 5) * (-2 - 1)
        + (4 - 2) * (0 + 2)
        + (3 - 1) * (3 - 5)
        + (8 - 3) * (1 - 3)
        + (2 - 8) * (2 - 1)
        + (1 - 3) * (0 + 2)
        + (1 - 1) * (1 - 0)
        + (5 - 1) * (3 - 1)
        + (7 - 1) * (4 - 4)
        + (1 - 7) * (2 - 4)
        + (2 - 1) * (3 - 2)
        + (1 - 1) * (5 - 2)
        + (3 - 5) * (3 - 1)
        + (8 - 2) * (1 + 2)
        + (2 - 4) * (2 - 0)
        + (3 - 1) * (-2 - 5)
        + (1 - 3) * (0 - 3)
        + (1 - 8) * (1 - 1)
        + (5 - 2) * (3 - 2)
        + (1 - 3) * (4 + 2)
        + (7 - 1) * (4 - 0)
        + (1 - 1) * (2 - 1)
        + (2 - 5) * (3 - 3)
    )
    assert actual == pytest.approx(expected)


def test_can_compute_h_norm(f):
    actual = norm(f, "h")
    expected = np.sqrt(
        f.grid.cell_volume
        * (
            1**2
            + 5**2
            + 2**2
            + 4**2
            + 1**2
            + 3**2
            + 8**2
            + 2**2
            + 3**2
            + 1**2
            + 1**2
            + 5**2
            + 1**2
            + 7**2
            + 1**2
            + 2**2
        )
    )
    assert actual == pytest.approx(expected)


def test_can_compute_grad_h_norm(g):
    actual = norm(g, "grad_h")
    expected = np.sqrt(
        (1 - 2) ** 2
        + (-2 - 1) ** 2
        + (0 + 2) ** 2
        + (3 - 5) ** 2
        + (1 - 3) ** 2
        + (2 - 1) ** 2
        + (0 + 2) ** 2
        + (1 - 0) ** 2
        + (3 - 1) ** 2
        + (4 - 4) ** 2
        + (2 - 4) ** 2
        + (3 - 2) ** 2
        + (5 - 2) ** 2
        + (3 - 1) ** 2
        + (1 + 2) ** 2
        + (2 - 0) ** 2
        + (-2 - 5) ** 2
        + (0 - 3) ** 2
        + (1 - 1) ** 2
        + (3 - 2) ** 2
        + (4 + 2) ** 2
        + (4 - 0) ** 2
        + (2 - 1) ** 2
        + (3 - 3) ** 2
    )
    assert actual == pytest.approx(expected)
