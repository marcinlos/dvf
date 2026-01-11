import numpy as np
import pytest

from dvf import Box, Interval


@pytest.fixture
def interval():
    return Interval(0.2, 1.7)


@pytest.fixture
def box():
    return Box(x=(-1, 2), y=(1, 3))


def test_interval_length_computed_correctly(interval) -> None:
    assert interval.length == pytest.approx(1.5)


def test_interval_contains_numbers_within_its_bounds(interval) -> None:
    assert 0.5 in interval


def test_interval_contains_its_endpoints(interval) -> None:
    assert interval.start in interval
    assert interval.end in interval


def test_interval_contains_numbers_outside_its_bounds(interval) -> None:
    assert 0.1 not in interval
    assert 2 not in interval


def test_box_can_get_span_by_index(box) -> None:
    assert box[1] == Interval(1, 3)


def test_box_can_get_span_by_axis_name(box) -> None:
    assert box["x"] == Interval(-1, 2)


def test_box_can_get_axis_names(box) -> None:
    assert box.axes == ("x", "y")


def test_box_can_get_number_of_dimensions(box) -> None:
    assert box.ndim == 2


def test_box_can_get_edge_lengths(box) -> None:
    np.testing.assert_allclose(box.edge_lengths, [3, 2])
