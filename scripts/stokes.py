# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np

from dvf import (
    CompositeFunctionSpace,
    Edge,
    FunctionSpace,
    Grid,
    GridFunction,
    TensorFunctionSpace,
    VectorFunctionSpace,
    assemble,
    div,
    integrate,
    lift_to_gridfun,
    nabla,
)

# %%
grid = Grid(5)

# %%
S = TensorFunctionSpace(grid, (2, 2))
U = VectorFunctionSpace(grid, 2)
P = FunctionSpace(grid)
W = CompositeFunctionSpace(S, U, P)

# %%
U_offset = S.dim
P_offset = S.dim + U.dim

U_bc = np.array([f.index for idx in grid.boundary() for f in U.basis_at(idx)])
P_bc = np.array(
    [f.index for idx in grid.boundary(Edge.LEFT | Edge.TOP) for f in P.basis_at(idx)]
    + [grid.ravel_index((grid.n, grid.n))]
)
W_bc = np.concat([U_bc + U_offset, P_bc + P_offset])

# %%
dot = lift_to_gridfun(np.vdot)


def pi0(f):
    grid = f.grid

    def fun(i, j):
        val = f(i, j)
        if i in (0, grid.n) or j in (0, grid.n):
            return np.zeros_like(val)
        else:
            return val

    return GridFunction(fun, grid)


def A_form(sigma, u, p, tau, v, q):
    return (
        dot(-div(sigma, "+") + nabla(p, "+"), v)
        + dot(div(u, "-"), q)
        + dot(sigma - nabla(u, "-"), tau)
    )


def AT_form(tau, v, q, sigma, u, p):
    return (
        dot(pi0(div(tau, "+") - nabla(q, "+")), u)
        + dot(-div(v, "-"), p)
        + dot(tau + nabla(v, "-"), sigma)
    )


def L2_product(sigma, u, p, tau, v, q):
    return dot(sigma, tau) + dot(u, v) + p * q


def AT_product(sigma, u, p, tau, v, q):
    return (
        dot(pi0(div(sigma, "+") - nabla(p, "+")), pi0(div(tau, "+") - nabla(q, "+")))
        + dot(div(u, "-"), div(v, "-"))
        + dot(sigma + nabla(u, "-"), tau + nabla(v, "-"))
    )


def AT_graph_product(*args):
    return L2_product(*args) + AT_product(*args)


# %%
def random_functions():
    sigma_data = np.random.rand(2, 2, *grid.shape)
    sigma = GridFunction.from_array(sigma_data, grid)

    u_data = np.random.rand(2, *grid.shape)
    u_data.flat[U_bc] = 0
    u = GridFunction.from_array(u_data, grid)

    p_data = np.random.rand(*grid.shape)
    p_data.flat[P_bc] = 0
    p = GridFunction.from_array(p_data, grid)

    return sigma, u, p


# %%
sigma, u, p = random_functions()
tau, v, q = random_functions()

# %%
lhs = integrate(A_form(sigma, u, p, tau, v, q))
rhs = integrate(AT_form(tau, v, q, sigma, u, p))

# %%
print(f"(Au, v)  = {lhs}")
print(f"(u, A*v) = {rhs}")
print(f"difference: {abs(rhs - lhs)}")

# %%
uu = W.trial_function()
vv = W.test_function()

sigma, u, p = uu.components
tau, v, q = vv.components

A = np.zeros((W.dim, W.dim))
AT = np.zeros((W.dim, W.dim))
AT_p = np.zeros((W.dim, W.dim))
M = np.zeros((W.dim, W.dim))
AT_graph = np.zeros((W.dim, W.dim))

assemble(A_form(sigma, u, p, tau, v, q), A, uu, vv)
assemble(AT_form(tau, v, q, sigma, u, p), AT, vv, uu)
assemble(AT_product(sigma, u, p, tau, v, q), AT_p, uu, vv)
assemble(L2_product(sigma, u, p, tau, v, q), M, uu, vv)
assemble(AT_graph_product(sigma, u, p, tau, v, q), AT_graph, uu, vv)


# %%
def apply_bc(matrix, dofs):
    return np.delete(np.delete(matrix, dofs, axis=0), dofs, axis=1)


A_ = apply_bc(A, W_bc)
AT_ = apply_bc(AT, W_bc)
AT_p_ = apply_bc(AT_p, W_bc)
M_ = apply_bc(M, W_bc)
AT_graph_ = apply_bc(AT_graph, W_bc)

# %%
A_ @ np.linalg.inv(M_) @ AT_

# %%
np.max(np.abs(A_ @ np.linalg.inv(M_) @ AT_ - AT_p_))
