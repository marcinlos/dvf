import numpy as np
import pytest

from dvf import (
    CompositeFunctionSpace,
    FunctionSpace,
    TensorFunctionSpace,
    VectorFunctionSpace,
)


@pytest.fixture
def U(grid4x4):
    return FunctionSpace(grid4x4)


@pytest.fixture
def U3(grid4x4):
    return VectorFunctionSpace(grid4x4, 3)


@pytest.fixture
def U3x2(grid4x4):
    return TensorFunctionSpace(grid4x4, (3, 2))


@pytest.fixture
def W(U, U3, U3x2):
    return CompositeFunctionSpace(U, U3, U3x2)


def test_function_space_has_trivial_shape(U):
    assert U.shape == ()


def test_function_space_dimension_is_grid_size(U, grid4x4):
    assert U.dim == grid4x4.size


def test_function_space_has_one_basis_function_at_each_point(U):
    us = U.basis_at((1, 2))
    assert len(list(us)) == 1


def test_function_space_basis_index_matches_raveled_tabulate(U, grid4x4):
    idx = (1, 2)
    [u] = U.basis_at(idx)
    values = np.ravel(u.tabulate())

    expected = np.zeros(U.dim)
    expected[u.index] = 1

    np.testing.assert_allclose(values, expected)


def test_vector_function_space_shape(U3):
    assert U3.shape == (3,)


def test_vector_function_space_dimension(U3, grid4x4):
    assert U3.dim == grid4x4.size * 3


def test_vector_function_space_has_k_basis_functions_at_each_point(U3):
    us = U3.basis_at((1, 2))
    assert len(list(us)) == 3


def test_vector_function_space_basis_index_matches_raveled_tabulate(U3, grid4x4):
    idx = (1, 2)
    [u0, u1, u2] = U3.basis_at(idx)
    values = np.ravel(u1.tabulate())

    expected = np.zeros(U3.dim)
    expected[u1.index] = 1

    np.testing.assert_allclose(values, expected)


def test_vector_function_space_basis_functions_have_component_index(U3, grid4x4):
    idx = (1, 2)
    us = list(U3.basis_at(idx))

    u = us[1]
    val = u(*idx)
    assert val[u.component] == 1


def test_tensor_function_space_shape(U3x2):
    assert U3x2.shape == (3, 2)


def test_tensor_function_space_dimension(U3x2, grid4x4):
    assert U3x2.dim == grid4x4.size * 6


def test_tensor_function_space_has_enough_basis_functions_at_each_point(U3x2):
    us = U3x2.basis_at((1, 2))
    assert len(list(us)) == 6


def test_tensor_function_space_basis_index_matches_raveled_tabulate(U3x2, grid4x4):
    idx = (1, 2)
    us = list(U3x2.basis_at(idx))
    u = us[4]
    values = np.ravel(u.tabulate())

    expected = np.zeros(U3x2.dim)
    expected[u.index] = 1

    np.testing.assert_allclose(values, expected)


def test_tensor_function_space_basis_functions_have_component_index(U3x2, grid4x4):
    idx = (1, 2)
    us = list(U3x2.basis_at(idx))
    u = us[4]
    val = u(*idx)
    assert val[u.component] == 1


def test_composite_function_space_dimension(W, grid4x4):
    assert W.dim == grid4x4.size * (1 + 3 + 6)


def test_composite_function_space_has_enough_basis_functions_at_each_point(W):
    us = W.basis_at((1, 2))
    assert len(list(us)) == 1 + 3 + 6


def test_can_extract_components_of_composite_function_space_basis_functions(W):
    u, *_ = W.basis_at((1, 2))
    (a, b, c) = u.components


def test_composite_function_space_basis_index_matches_raveled_tabulate(W, grid4x4):
    idx = (1, 2)
    us = list(W.basis_at(idx))
    u = us[5]
    values = np.concat([np.ravel(a.tabulate()) for a in u.components])

    expected = np.zeros(W.dim)
    expected[u.index] = 1

    np.testing.assert_allclose(values, expected)
