import pytest

from dvf import Grid


@pytest.fixture
def grid3x3():
    return Grid(2, 2)


@pytest.fixture
def grid3x4():
    return Grid(2, 3)


@pytest.fixture
def grid4x4():
    return Grid(3, 3)


@pytest.fixture
def grid5x5():
    return Grid(4, 4)
