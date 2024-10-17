import numpy as np
import pytest

import dvf
from dvf import (
    CompositeFunctionSpace,
    FunctionSpace,
    GridFunction,
    TensorFunctionSpace,
    VectorFunctionSpace,
    assemble,
    div,
    grad,
    integrate,
    lift_to_gridfun,
)


def test_mass_matrix_assembly(grid4x4):
    U = FunctionSpace(grid4x4)
    u = U.trial_function()
    v = U.test_function()

    M = np.zeros((U.dim, U.dim))
    assemble(u * v, M, u, v)

    expected = grid4x4.cell_volume * np.identity(U.dim)

    np.testing.assert_allclose(M, expected)


def test_stiffness_matrix_acts_correctly(grid4x4):
    f = GridFunction.from_function(lambda x, y: x**2 + np.sin(x - 2 * y), grid4x4)
    g = GridFunction.from_function(lambda x, y: np.cos(3 * x**2 - 2 * y), grid4x4)

    def form(u, v):
        dot = dvf.lift_to_gridfun(np.vdot)
        return dot(grad(u, "+"), grad(v, "+"))

    U = FunctionSpace(grid4x4)
    u = U.trial_function()
    v = U.test_function()

    K = np.zeros((U.dim, U.dim))
    assemble(form(u, v), K, u, v)

    from_integration = integrate(form(f, g))

    f_data = np.ravel(f.tabulate())
    g_data = np.ravel(g.tabulate())
    from_matrix = g_data @ K @ f_data

    assert from_integration == pytest.approx(from_matrix)


def test_vector_space_matrix_assembly(grid4x4):
    f = GridFunction.from_function(
        lambda x, y: np.array([np.sin(x - 2 * y), x + y**2]), grid4x4
    )
    g = GridFunction.from_function(
        lambda x, y: np.array([np.cos(3 * x**2 - 2 * y), x**2 - y]), grid4x4
    )

    def form(u, v):
        dot = dvf.lift_to_gridfun(np.vdot)
        return dot(u, v) + dot(grad(u, "+"), grad(v, "-"))

    U = VectorFunctionSpace(grid4x4, 2)
    u = U.trial_function()
    v = U.test_function()

    A = np.zeros((U.dim, U.dim))
    assemble(form(u, v), A, u, v)

    from_integration = integrate(form(f, g))

    f_data = np.ravel(f.tabulate())
    g_data = np.ravel(g.tabulate())
    from_matrix = g_data @ A @ f_data

    assert from_integration == pytest.approx(from_matrix)


def test_composite_space_matrix_assembly(grid4x4):
    f = GridFunction.from_function(
        lambda x, y: np.array([np.sin(x - 2 * y), x + y**2]), grid4x4
    )
    r = GridFunction.from_function(lambda x, y: x**2 + np.sin(y), grid4x4)

    g = GridFunction.from_function(
        lambda x, y: np.array([np.cos(3 * x**2 - 2 * y), x**2 - y]), grid4x4
    )
    s = GridFunction.from_function(lambda x, y: np.cos(x) + 2 * y, grid4x4)

    def form(u, p, v, q):
        dot = dvf.lift_to_gridfun(np.vdot)
        return dot(u, v) + dot(div(u, "+"), q) - dot(p, div(v, "+"))

    U = VectorFunctionSpace(grid4x4, 2)
    P = FunctionSpace(grid4x4)
    W = CompositeFunctionSpace(U, P)

    uu = W.trial_function()
    vv = W.test_function()
    u, p = uu.components
    v, q = vv.components

    A = np.zeros((W.dim, W.dim))
    assemble(form(u, p, v, q), A, uu, vv)

    from_integration = integrate(form(f, r, g, s))

    fr_data = np.concat([np.ravel(f.tabulate()), np.ravel(r.tabulate())])
    gs_data = np.concat([np.ravel(g.tabulate()), np.ravel(s.tabulate())])
    from_matrix = gs_data @ A @ fr_data

    assert from_integration == pytest.approx(from_matrix)


@pytest.mark.skip
def test_full_stokes(grid4x4):
    grid = grid4x4
    grid = dvf.Grid(10)

    S = TensorFunctionSpace(grid, (2, 2))
    U = VectorFunctionSpace(grid, 2)
    P = FunctionSpace(grid)
    W = CompositeFunctionSpace(S, U, P)

    dot = lift_to_gridfun(np.vdot)

    def form(uu, vv):
        sigma, u, p = uu.components
        tau, v, q = vv.components

        return (
            dot(u, v)
            + dot(sigma, tau)
            + dot(p, q)
            + dot(div(sigma, "+") - grad(p, "+"), div(tau, "+") - grad(q, "+"))
            + dot(div(u, "-"), div(v, "-"))
            + dot(sigma + grad(u, "-"), tau + grad(v, "-"))
        )

    A = np.zeros((W.dim, W.dim))

    uu = W.trial_function()
    vv = W.test_function()

    assemble(form(uu, vv), A, uu, vv)
