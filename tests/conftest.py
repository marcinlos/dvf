from types import SimpleNamespace

import pytest

from dvf import Box, Grid


@pytest.fixture(scope="session")
def domains():
    dom = SimpleNamespace()

    dom.unit_interval = Box(x=(0, 1))
    dom.interval = Box(x=(-2, 1))

    dom.unit_square = Box(x=(0, 1), y=(0, 1))
    dom.rectangle = Box(x=(-2, 1), y=(1, 3))

    dom.unit_cube = Box(x=(0, 1), y=(0, 1), z=(0, 1))
    dom.cuboid = Box(x=(-2, 1), y=(1, 3), z=(0, 4))

    return dom


@pytest.fixture
def grids(domains):
    g = SimpleNamespace()

    g.unit_5 = Grid(domains.unit_interval, 4)
    g._5 = Grid(domains.interval, 4)

    g.unit_3x4 = Grid(domains.unit_square, 2, 3)
    g.unit_4x4 = Grid(domains.unit_square, 3, 3)
    g.unit_5x5 = Grid(domains.unit_square, 4, 4)

    g._3x4 = Grid(domains.rectangle, 2, 3)
    g._4x4 = Grid(domains.rectangle, 3, 3)
    g._5x5 = Grid(domains.rectangle, 4, 4)

    g.unit_3x4x3 = Grid(domains.unit_cube, 2, 3, 2)
    g._3x4x3 = Grid(domains.cuboid, 2, 3, 2)

    return g


@pytest.fixture
def grid3x4(grids):
    return grids.unit_3x4  # Grid(unit_square, 2, 3)


@pytest.fixture
def grid4x4(grids):
    return grids.unit_4x4  # Grid(unit_square, 3, 3)


@pytest.fixture
def grid5x5(grids):
    return grids.unit_5x5  # Grid(unit_square, 4, 4)
