from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from dvf import Edge, Grid


def test_number_of_sizes_must_match_domain_dimension(domains) -> None:
    dim_error = pytest.raises(ValueError, match="domain dimension")

    # too few sizes
    with dim_error:
        Grid(domains.unit_square, 2)

    # too many sizes
    with dim_error:
        Grid(domains.unit_square, 2, 3, 5)


def test_ndim(grids) -> None:
    assert grids._5.ndim == 1
    assert grids._4x4.ndim == 2
    assert grids._3x4x3.ndim == 3


def test_indices_1d(grids) -> None:
    grid = grids.unit_5
    expected = {(0,), (1,), (2,), (3,), (4,)}
    assert Counter(grid.indices) == Counter(expected)


def test_indices_2d(grids) -> None:
    grid = grids.unit_3x4
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
        (0, 3),
        (1, 3),
        (2, 3),
    }
    assert Counter(grid.indices) == Counter(expected)


def test_indices_3d(grids) -> None:
    grid = grids.unit_3x4x3
    expected = {
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
        (0, 3, 0),
        (0, 3, 1),
        (0, 3, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 0),
        (1, 2, 1),
        (1, 2, 2),
        (1, 3, 0),
        (1, 3, 1),
        (1, 3, 2),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
        (2, 3, 0),
        (2, 3, 1),
        (2, 3, 2),
    }
    assert Counter(grid.indices) == Counter(expected)


def test_points_1d(grids) -> None:
    grid = grids._5

    # (-2, 1) interval
    x_expected = [-2, -1.25, -0.5, 0.25, 1]

    (x,) = grid.points
    np.testing.assert_allclose(x, x_expected)


def test_points_2d_unit_square(grids) -> None:
    grid = grids.unit_3x4
    x, y = grid.points
    assert (x[1, 2], y[1, 2]) == (0.5, 2 / 3)

    x_expected = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ]
    ).T
    y_expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1 / 3, 1 / 3, 1 / 3],
            [2 / 3, 2 / 3, 2 / 3],
            [1.0, 1.0, 1.0],
        ]
    ).T
    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_points_2d(grids) -> None:
    grid = grids._3x4
    # (-2, 1) x (1, 3))

    x_expected = np.array(
        [
            [-2, -0.5, 1],
            [-2, -0.5, 1],
            [-2, -0.5, 1],
            [-2, -0.5, 1],
        ]
    ).T
    y_expected = np.array(
        [
            [1, 1, 1],
            [5 / 3, 5 / 3, 5 / 3],
            [7 / 3, 7 / 3, 7 / 3],
            [3, 3, 3],
        ]
    ).T

    x, y = grid.points

    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_points_3d(grids) -> None:
    grid = grids._3x4x3
    # (-2, 1) x (1, 3) x (0, 4))

    x_expected = [
        [
            [-2, -2, -2],
            [-2, -2, -2],
            [-2, -2, -2],
            [-2, -2, -2],
        ],
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    ]

    y_expected = [
        [
            [1, 1, 1],
            [5 / 3, 5 / 3, 5 / 3],
            [7 / 3, 7 / 3, 7 / 3],
            [3, 3, 3],
        ],
        [
            [1, 1, 1],
            [5 / 3, 5 / 3, 5 / 3],
            [7 / 3, 7 / 3, 7 / 3],
            [3, 3, 3],
        ],
        [
            [1, 1, 1],
            [5 / 3, 5 / 3, 5 / 3],
            [7 / 3, 7 / 3, 7 / 3],
            [3, 3, 3],
        ],
    ]

    z_expected = [
        [
            [0, 2, 4],
            [0, 2, 4],
            [0, 2, 4],
            [0, 2, 4],
        ],
        [
            [0, 2, 4],
            [0, 2, 4],
            [0, 2, 4],
            [0, 2, 4],
        ],
        [
            [0, 2, 4],
            [0, 2, 4],
            [0, 2, 4],
            [0, 2, 4],
        ],
    ]

    x, y, z = grid.points

    np.testing.assert_allclose(x, np.array(x_expected))
    np.testing.assert_allclose(y, np.array(y_expected))
    np.testing.assert_allclose(z, np.array(z_expected))


def test_can_get_points_by_simple_index_1d(grids) -> None:
    grid = grids._5
    assert grid[2] == pytest.approx(-0.5)


def test_can_get_points_by_simple_index_2d(grids) -> None:
    grid = grids._3x4
    assert grid[1, 2] == pytest.approx((-0.5, 7 / 3))


def test_can_get_points_by_simple_index_3d(grids) -> None:
    grid = grids._3x4x3
    assert grid[1, 2, 0] == pytest.approx((-0.5, 7 / 3, 0))


def test_can_get_points_by_index_array_1d(grids) -> None:
    grid = grids._5
    # (-2, 1)
    ii = [[1, 2, 0], [2, 1, 4]]

    x_expected = [[-1.25, -0.5, -2], [-0.5, -1.25, 1]]

    (x,) = grid[ii]

    np.testing.assert_allclose(x, x_expected)


def test_can_get_points_by_index_array_2d(grids) -> None:
    grid = grids._3x4
    # (-2, 1) x (1, 3)

    ii = [[1, 2, 0], [2, 1, 2]]
    jj = [[0, 1, 1], [3, 2, 0]]

    x_expected = [
        [-0.5, 1, -2],
        [1, -0.5, 1],
    ]
    y_expected = [
        [1, 5 / 3, 5 / 3],
        [3, 7 / 3, 1],
    ]

    x, y = grid[ii, jj]

    np.testing.assert_allclose(x, x_expected)
    np.testing.assert_allclose(y, y_expected)


def test_can_get_points_by_row_index_2d(grids) -> None:
    grid = grids._3x4
    # (-2, 1) x (1, 3)

    ii = [1, 2]

    x_expected = [
        [-0.5, -0.5, -0.5, -0.5],
        [1, 1, 1, 1],
    ]
    y_expected = [
        [1, 5 / 3, 7 / 3, 3],
        [1, 5 / 3, 7 / 3, 3],
    ]

    x, y = grid[ii]

    np.testing.assert_allclose(x, np.array(x_expected))
    np.testing.assert_allclose(y, np.array(y_expected))


def test_can_get_points_by_slice_1d(grids) -> None:
    grid = grids._5
    # (-2, 1)

    x_expected = [-1.25, -0.5, 0.25]
    (x,) = grid[1:4]

    np.testing.assert_allclose(x, x_expected)


def test_can_get_points_by_slice_2d(grids) -> None:
    grid = grids._3x4
    # (-2, 1) x (1, 3)

    jj = np.s_[1::2]

    x_expected = [
        [-2, -2],
        [-0.5, -0.5],
        [1, 1],
    ]
    y_expected = [
        [5 / 3, 3],
        [5 / 3, 3],
        [5 / 3, 3],
    ]

    x, y = grid[:, jj]

    np.testing.assert_allclose(x, np.array(x_expected))
    np.testing.assert_allclose(y, np.array(y_expected))


def test_can_get_points_by_bool_array_1d(grids) -> None:
    grid = grids._5
    # (-2, 1)

    ii = [False, True, False, True, True]

    x_expected = [-1.25, 0.25, 1]
    (x,) = grid[ii]

    np.testing.assert_allclose(x, x_expected)


def test_can_get_points_by_bool_array_2d(grids) -> None:
    grid = grids._3x4
    # (-2, 1) x (1, 3)

    ii = [
        [False, True, False, True],
        [True, False, False, False],
        [False, True, True, False],
    ]

    points_expected = [
        [-2, -2, -0.5, 1, 1],
        [5 / 3, 3, 1, 5 / 3, 7 / 3],
    ]

    points = grid[ii]

    np.testing.assert_allclose(points, points_expected)


def test_negative_x_index_is_rejected_2d(grids) -> None:
    grid = grids._3x4
    assert not grid.index_valid((-1, 1))


def test_negative_y_index_is_rejected_2d(grids) -> None:
    grid = grids._3x4
    assert not grid.index_valid((1, -1))


def test_too_large_x_index_is_rejected_2d(grids) -> None:
    grid = grids._3x4
    assert not grid.index_valid((3, 1))


def test_too_large_y_index_is_rejected_2d(grids) -> None:
    grid = grids._3x4
    assert not grid.index_valid((1, 4))


def test_index_within_bounds_is_accepted_2d(grids) -> None:
    grid = grids._3x4
    assert grid.index_valid((0, 0))
    assert grid.index_valid((2, 0))
    assert grid.index_valid((0, 3))
    assert grid.index_valid((2, 3))


def test_can_check_validity_of_array_of_inddices_2d(grids) -> None:
    grid = grids._3x4

    ii = [[1, 2, 0], [2, 1, 3]]
    jj = [[0, 3, 4], [-1, 2, 0]]

    expected = [
        [True, True, False],
        [False, True, False],
    ]

    actual = grid.index_valid((ii, jj))

    np.testing.assert_array_equal(actual, expected)


def test_checking_validity_of_single_element_index_array_returns_array(grids) -> None:
    grid = grids._3x4

    ii = [[[1]]]
    jj = [[[2]]]

    actual = grid.index_valid((ii, jj))
    assert actual.shape == (1, 1, 1)
    assert actual


def test_size_computed_correctly(grids) -> None:
    assert grids._5.size == 5
    assert grids._3x4.size == 12
    assert grids._3x4x3.size == 36


def test_shape_is_computed_correctly(grids) -> None:
    assert grids._5.shape == (5,)
    assert grids._3x4.shape == (3, 4)
    assert grids._3x4x3.shape == (3, 4, 3)


def test_ravel_index_1d(grids) -> None:
    grid = grids._5
    assert grid.ravel_index((3,)) == 3


def test_ravel_index_2d(grids) -> None:
    grid = grids._3x4
    assert grid.ravel_index((2, 3)) == 4 * 2 + 1 * 3


def test_ravel_index_3d(grids) -> None:
    grid = grids._3x4x3
    s = [12, 3, 1]
    assert grid.ravel_index((2, 3, 2)) == s[0] * 2 + s[1] * 3 + s[2] * 2


def test_ravel_index_array_2d(grids) -> None:
    grid = grids._3x4

    ii = [[1, 2, 0], [2, 1, 2]]
    jj = [[0, 1, 1], [3, 2, 0]]

    expected = [
        [4 * 1 + 0, 4 * 2 + 1, 4 * 0 + 1],
        [4 * 2 + 3, 4 * 1 + 2, 4 * 2 + 0],
    ]

    actual = grid.ravel_index((ii, jj))

    np.testing.assert_allclose(actual, np.array(expected))


def test_ravel_index_is_consistent_with_order_of_indices_method(grids) -> None:
    grid = grids._3x4x3

    indices = list(grid.indices)
    i, j, k = np.asarray(indices).T

    raveled = grid.ravel_index((i, j, k))
    expected = np.arange(grid.size)

    np.testing.assert_allclose(raveled, expected)


def test_ravel_index_is_consistent_with_ordder_of_points_method(grids) -> None:
    grid = grids._3x4x3

    indices = list(grid.indices)
    i, j, k = np.asarray(indices).T
    raveled = grid.ravel_index((i, j, k))

    flat = grid.points.reshape(grid.ndim, -1)
    from_flat = flat[:, raveled]
    from_grid = grid[i, j, k]

    np.testing.assert_allclose(from_flat, from_grid)


def test_can_enumerate_boundary_indices(grid3x4) -> None:
    expected = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (2, 1),
        (0, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    }
    assert Counter(grid3x4.boundary()) == Counter(expected)


def test_none_boundary_is_empty(grid3x4) -> None:
    assert Counter(grid3x4.boundary(Edge.NONE)) == Counter({})


def test_can_enumerate_top_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (1, 0), (2, 0)}
    actual = grid3x4.boundary(Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_boundary_indices(grid3x4) -> None:
    expected = {(0, 3), (1, 3), (2, 3)}
    actual = grid3x4.boundary(Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_bottom_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (1, 0), (2, 0), (0, 3), (1, 3), (2, 3)}
    actual = grid3x4.boundary(Edge.TOP | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_left_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3)}
    actual = grid3x4.boundary(Edge.LEFT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_right_boundary_indices(grid3x4) -> None:
    expected = {(2, 0), (2, 1), (2, 2), (2, 3)}
    actual = grid3x4.boundary(Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_left_right_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 2), (2, 3)}
    actual = grid3x4.boundary(Edge.LEFT | Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_left_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0)}
    actual = grid3x4.boundary(Edge.LEFT | Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_top_right_boundary_indices(grid3x4) -> None:
    expected = {(2, 0), (2, 1), (2, 2), (2, 3), (0, 0), (1, 0)}
    actual = grid3x4.boundary(Edge.RIGHT | Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_left_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}
    actual = grid3x4.boundary(Edge.LEFT | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_bottom_right_boundary_indices(grid3x4) -> None:
    expected = {(2, 0), (2, 1), (2, 2), (2, 3), (0, 3), (1, 3)}
    actual = grid3x4.boundary(Edge.RIGHT | Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_top_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0)}
    actual = grid3x4.boundary(~Edge.TOP)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_bottom_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0)}
    actual = grid3x4.boundary(~Edge.BOTTOM)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_left_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0)}
    actual = grid3x4.boundary(~Edge.LEFT)
    assert Counter(actual) == Counter(expected)


def test_can_enumerate_non_right_boundary_indices(grid3x4) -> None:
    expected = {(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 0), (1, 0)}
    actual = grid3x4.boundary(~Edge.RIGHT)
    assert Counter(actual) == Counter(expected)


def test_adjacent_interior(grids) -> None:
    grid = grids._5x5
    expected = {(1, 2), (2, 2), (3, 2), (3, 3), (3, 4), (2, 4), (1, 4), (1, 3)}
    actual = grid.adjacent((2, 3))
    assert Counter(actual) == Counter(expected)


def test_adjacent_edge(grids) -> None:
    grid = grids._5x5
    expected = {(0, 1), (1, 1), (1, 2), (1, 3), (0, 3)}
    actual = grid.adjacent((0, 2))
    assert Counter(actual) == Counter(expected)


def test_adjacent_corner(grids) -> None:
    grid = grids._5x5
    expected = {(3, 0), (3, 1), (4, 1)}
    actual = grid.adjacent((4, 0))
    assert Counter(actual) == Counter(expected)


def make_boundary_data(grid, vec):
    data = SimpleNamespace()
    data.grid = grid

    data.indices = list(vec)
    data.index_tuple = np.array(data.indices).T

    data.area_vector_dict = vec
    vector_list = list(vec.values())
    data.area_vectors = np.array(vector_list).T

    data.normal_dict = {i: np.array(v) / np.linalg.norm(v) for i, v in vec.items()}
    data.normals = np.array([data.normal_dict[i] for i in vec]).T
    data.area_dict = {i: np.linalg.norm(v) for i, v in vec.items()}
    data.areas = np.array([data.area_dict[i] for i in vec])

    return data


def boundary_data_1d(grids):
    grid = grids._5
    # -2, 1
    (h,) = grid.h
    vec = {(0,): (-1,), (4,): (1,)}

    return grid, vec


def boundary_data_2d(grids):
    grid = grids._3x4
    hx, hy = grid.h
    vec = {
        (0, 0): (-hy, -hx),
        (1, 0): (0, -hx),
        (2, 0): (hy, -hx),
        (2, 1): (hy, 0),
        (2, 2): (hy, 0),
        (2, 3): (hy, hx),
        (1, 3): (0, hx),
        (0, 3): (-hy, hx),
        (0, 2): (-hy, 0),
        (0, 1): (-hy, 0),
    }

    return grid, vec


def boundary_data_3d(grids):
    grid = grids._3x4x3
    hx, hy, hz = grid.h
    hyz, hxz, hxy = hx * hy * hz / grid.h
    vec = {
        # X
        (0, 1, 1): (-hyz, 0, 0),
        (0, 2, 1): (-hyz, 0, 0),
        (2, 1, 1): (hyz, 0, 0),
        (2, 2, 1): (hyz, 0, 0),
        # Y
        (1, 0, 1): (0, -hxz, 0),
        (1, 3, 1): (0, hxz, 0),
        # Z
        (1, 1, 0): (0, 0, -hxy),
        (1, 2, 0): (0, 0, -hxy),
        (1, 1, 2): (0, 0, hxy),
        (1, 2, 2): (0, 0, hxy),
        # X edges
        (1, 0, 0): (0, -hxz, -hxy),
        (1, 3, 0): (0, hxz, -hxy),
        (1, 0, 2): (0, -hxz, hxy),
        (1, 3, 2): (0, hxz, hxy),
        # Y edges
        (0, 1, 0): (-hyz, 0, -hxy),
        (0, 2, 0): (-hyz, 0, -hxy),
        (2, 1, 0): (hyz, 0, -hxy),
        (2, 2, 0): (hyz, 0, -hxy),
        (0, 1, 2): (-hyz, 0, hxy),
        (0, 2, 2): (-hyz, 0, hxy),
        (2, 1, 2): (hyz, 0, hxy),
        (2, 2, 2): (hyz, 0, hxy),
        # Z edges
        (0, 0, 1): (-hyz, -hxz, 0),
        (2, 0, 1): (hyz, -hxz, 0),
        (0, 3, 1): (-hyz, hxz, 0),
        (2, 3, 1): (hyz, hxz, 0),
        # corners
        (0, 0, 0): (-hyz, -hxz, -hxy),
        (2, 0, 0): (hyz, -hxz, -hxy),
        (0, 3, 0): (-hyz, hxz, -hxy),
        (0, 0, 2): (-hyz, -hxz, hxy),
        (0, 3, 2): (-hyz, hxz, hxy),
        (2, 0, 2): (hyz, -hxz, hxy),
        (2, 3, 0): (hyz, hxz, -hxy),
        (2, 3, 2): (hyz, hxz, hxy),
    }

    return grid, vec


@pytest.fixture(
    params=[boundary_data_1d, boundary_data_2d, boundary_data_3d],
    ids=["1D", "2D", "3D"],
)
def boundary_data(grids, request):
    fun = request.param
    return make_boundary_data(*fun(grids))


def test_facet_area_vectors(boundary_data) -> None:
    data = boundary_data
    grid = data.grid

    expected = data.area_vector_dict
    actual = {idx: grid.facet_area_vector(idx) for idx in expected}

    for p in expected:
        np.testing.assert_allclose(actual[p], expected[p])


def test_facet_normals(boundary_data) -> None:
    data = boundary_data
    grid = data.grid

    expected = data.normal_dict
    actual = {idx: grid.facet_normal(idx) for idx in expected}

    for p in expected:
        np.testing.assert_allclose(actual[p], expected[p])


def test_facet_area(boundary_data) -> None:
    data = boundary_data
    grid = data.grid

    expected = data.area_dict
    actual = {idx: grid.facet_area(idx) for idx in expected}

    for p in expected:
        assert actual[p] == pytest.approx(expected[p])


def test_facet_area_vectors_index_array(boundary_data) -> None:
    data = boundary_data
    grid = data.grid

    expected = data.area_vectors
    actual = grid.facet_area_vector(data.index_tuple)

    np.testing.assert_allclose(actual, expected)


def test_facet_normals_index_array(boundary_data) -> None:
    data = boundary_data
    grid = data.grid

    expected = data.normals
    actual = grid.facet_normal(data.index_tuple)

    np.testing.assert_allclose(actual, expected)


def test_facet_area_index_array(boundary_data) -> None:
    data = boundary_data
    grid = data.grid

    expected = data.areas
    actual = grid.facet_area(data.index_tuple)

    np.testing.assert_allclose(actual, expected)


def test_facet_area_vector_is_zero_for_interior_point(grids) -> None:
    grid = grids._3x4

    expected = (0, 0)
    actual = grid.facet_area_vector((1, 2))

    np.testing.assert_allclose(actual, expected)


def test_area_is_zero_for_interior_point(grids) -> None:
    grid = grids._3x4

    expected = 0
    actual = grid.facet_area((1, 2))

    assert actual == pytest.approx(expected)


def test_normal_vector_is_zero_for_interior_point(grids) -> None:
    grid = grids._3x4

    expected = (0, 0)
    actual = grid.facet_normal((1, 2))

    np.testing.assert_allclose(actual, expected)
