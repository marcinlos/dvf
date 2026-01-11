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

# %% [markdown]
# # Laplace RPINN

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy

from dvf import (
    CompositeFunctionSpace,
    FunctionSpace,
    Grid,
    GridFunction,
    VectorFunctionSpace,
    assemble,
    div,
    grad,
    integrate,
    lift_to_gridfun,
    norm,
    random_function,
    reinsert_dofs,
    remove_dofs,
    select_dofs,
)

# %%
grid = Grid(50, 50)

# %%
S = VectorFunctionSpace(grid, 2)
U = FunctionSpace(grid)
W = CompositeFunctionSpace(S, U)


# %%
def S_mask_fun(i, j):
    mdx = 0 if i == 0 else 1
    mdy = 0 if j == 0 else 1
    return np.array([mdx, mdy])


# %%
def U_mask_fun(i, j):
    edges_x = (0, grid.n[0])
    edges_y = (0, grid.n[1])
    m = 0 if i in edges_x or j in edges_y else 1
    return m


# %%
s_mask = GridFunction(S_mask_fun, grid)
u_mask = GridFunction(U_mask_fun, grid)

# %%
S_bc = select_dofs(S, s_mask, invert=True)
U_bc = select_dofs(U, u_mask, invert=True)

W_bc = W.combine_dofs(S_bc, U_bc)

# %%
dot = lift_to_gridfun(np.vdot)


def A_form(sigma, u, tau, v):
    return -dot(div(sigma, "+"), v) + dot(sigma - grad(u, "-"), tau)


def AT_form(tau, v, sigma, u):
    return dot(div(tau, "+"), u) + dot(tau + grad(v, "-"), sigma)


# %% [markdown]
# ### Verification of the adjoint
#
# We use random functions to avoid assembling the matrix of $A^*$, which is expensive
# for large grids.


# %%
def random_functions():
    sigma = random_function(grid, shape=(2,), bc=S_bc)
    u = random_function(grid, bc=U_bc)
    return sigma, u


# %%
sigma, u = random_functions()
tau, v = random_functions()

# %%
lhs = integrate(A_form(sigma, u, tau, v))
rhs = integrate(AT_form(tau, v, sigma, u))

# %% [markdown]
# $(A\boldsymbol{u}, \boldsymbol{v})_h = (\boldsymbol{u}, A^* \boldsymbol{v})_h$ should
# hold for all $\boldsymbol{u}$, $\boldsymbol{v}$.

# %%
print(f"(Au, v)  = {lhs}")
print(f"(u, A*v) = {rhs}")
print(f"difference: {abs(rhs - lhs)}")

# %%
uu = W.trial_function()
vv = W.test_function()

sigma, u = uu.components
tau, v = vv.components

A = np.zeros((W.dim, W.dim))
M = np.zeros((W.dim, W.dim))

# %%
assemble(A_form(sigma, u, tau, v), A, uu, vv)

# %%
M = grid.cell_volume * np.identity(W.dim)


# %%
A_ = remove_dofs(A, W_bc)
M_ = remove_dofs(M, W_bc)

# %%
x, y = sympy.symbols("x y")
u_exact = sympy.sin(sympy.pi * x) * sympy.sin(3 * sympy.pi * y)


def laplacian(expr, *vars):
    return sum(sympy.diff(expr, v, v) for v in vars)


rhs = -laplacian(u_exact, x, y)

# %%
u_exact = GridFunction.from_function(sympy.lambdify([x, y], u_exact), grid)


# %%
def vector_of_values(*funs):
    return np.concat([np.ravel(f.tabulate()) for f in funs])


rhs_f = GridFunction.from_function(sympy.lambdify([x, y], rhs), grid)
rhs_vec = remove_dofs(vector_of_values(S.zero_fun, rhs_f), W_bc)

# %%
lu, piv = scipy.linalg.lu_factor(A_)
solution_data = scipy.linalg.lu_solve((lu, piv), M_ @ rhs_vec)

# %%
first_U = S.dim - len(S_bc)

solution_sigma_vec = solution_data[:first_U]
solution_u_vec = solution_data[first_U:]


# %%
def vec_to_fun(vec, bc, shape=()):
    data = reinsert_dofs(vec, bc)
    return GridFunction.from_array(data.reshape(*shape, *grid.shape), grid)


solution_sigma = vec_to_fun(solution_sigma_vec, S_bc, (2,))
solution_u = vec_to_fun(solution_u_vec, U_bc)

# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
aspect = grid.h[1] / grid.h[0]
axs[0].imshow(solution_u.tabulate().T, aspect=aspect)
axs[1].imshow(solution_sigma.tabulate()[0].T, aspect=aspect)
axs[2].imshow(solution_sigma.tabulate()[1].T, aspect=aspect)
plt.show()

# %%
norm((solution_u - u_exact) * u_mask, "h")

# %%
a = np.arange(12).reshape(3, 4)

# %%
a

# %%
a[[[1, 2], [1, 0]]]

# %%
a[(0, 3)]

# %%
