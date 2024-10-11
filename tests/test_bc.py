import numpy as np
import pytest

from dvf import reinsert_dofs, remove_dofs


@pytest.fixture
def mat():
    return np.arange(16).reshape(4, 4)


def test_can_remove_dofs_from_vector():
    vec = np.arange(10)
    dofs = [1, 4, 5, 8]
    reduced = remove_dofs(vec, dofs)

    np.testing.assert_allclose(reduced, [0, 2, 3, 6, 7, 9])


def test_cannot_remove_test_dofs_from_vector():
    vec = np.arange(10)
    with pytest.raises(ValueError, match="test DoFs"):
        remove_dofs(vec, test_dofs=[1])


def test_cannot_remove_trial_dofs_from_vector():
    vec = np.arange(10)
    with pytest.raises(ValueError, match="trial DoFs"):
        remove_dofs(vec, trial_dofs=[1])


def test_must_specify_dofs_to_remove_from_vector():
    vec = np.arange(10)
    with pytest.raises(ValueError, match="not specified"):
        remove_dofs(vec)


def test_can_remove_dofs_from_matrix(mat):
    dofs = [0, 2]

    reduced = remove_dofs(mat, dofs)

    expected = np.array([[5, 7], [13, 15]])
    np.testing.assert_allclose(reduced, expected)


def test_can_remove_trial_dofs_from_matrix(mat):
    dofs = [0, 2]

    reduced = remove_dofs(mat, trial_dofs=dofs)

    expected = np.array([[1, 3], [5, 7], [9, 11], [13, 15]])
    np.testing.assert_allclose(reduced, expected)


def test_can_remove_test_dofs_from_matrix(mat):
    dofs = [0, 2]

    reduced = remove_dofs(mat, test_dofs=dofs)

    expected = np.array([[4, 5, 6, 7], [12, 13, 14, 15]])
    np.testing.assert_allclose(reduced, expected)


def test_can_remove_trial_and_test_dofs_from_matrix(mat):
    trial_dofs = [0, 2]
    test_dofs = [1, 3]

    reduced = remove_dofs(mat, trial_dofs=trial_dofs, test_dofs=test_dofs)

    expected = np.array([[1, 3], [9, 11]])
    np.testing.assert_allclose(reduced, expected)


def test_cannot_remove_dofs_and_trial_dofs_from_matrix(mat):
    with pytest.raises(ValueError, match="trial_dofs"):
        remove_dofs(mat, [0], trial_dofs=[0])


def test_cannot_remove_dofs_and_test_dofs_from_matrix(mat):
    with pytest.raises(ValueError, match="test_dofs"):
        remove_dofs(mat, [0], test_dofs=[0])


def test_must_specify_some_dofs_to_remove_from_matrix(mat):
    with pytest.raises(ValueError, match="not specified"):
        remove_dofs(mat)


def test_cannot_remove_from_higher_order_tensors():
    vec = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="rank 3"):
        remove_dofs(vec, [])


def test_can_reinsert_dofs_into_vector():
    vec = np.arange(10)
    dofs = [1, 4, 5, 8]
    reduced = remove_dofs(vec, dofs)
    restored = reinsert_dofs(reduced, dofs)

    np.testing.assert_allclose(restored, [0, 0, 2, 3, 0, 0, 6, 7, 0, 9])


def test_can_reinsert_dofs_into_vector_with_custom_value():
    vec = np.arange(10)
    dofs = [1, 4, 5, 8]
    reduced = remove_dofs(vec, dofs)
    restored = reinsert_dofs(reduced, dofs, -2)

    np.testing.assert_allclose(restored, [0, -2, 2, 3, -2, -2, 6, 7, -2, 9])
