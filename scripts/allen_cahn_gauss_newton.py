# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Allen-Cahn equation with Gauss-Newton

# %%
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy

from dvf import (
    FunctionSpace,
    Grid,
    GridFunction,
    assemble,
    assemble_residuum,
    grad,
    lift_to_gridfun,
    norm,
    reinsert_dofs,
    remove_dofs,
    select_dofs,
)

# %%
grid = Grid(60, 60)

# %%
U = FunctionSpace(grid)


# %%
def U_mask_fun(i, j):
    edges_x = (0, grid.n[0])
    edges_y = (0, grid.n[1])
    m = 0 if i in edges_x or j in edges_y else 1
    return m


# %%
u_mask = GridFunction(U_mask_fun, grid)

# %%
U_bc = select_dofs(U, u_mask, invert=True)

# %%
dot = lift_to_gridfun(np.vdot)


# %%
def vector_of_values(*funs):
    return np.concat([np.ravel(f.tabulate()) for f in funs])


# %%
def laplacian(expr, *vars):
    return sum(sympy.diff(expr, v, v) for v in vars)


# %%
x, y = sympy.symbols("x y")
u_expr = sympy.sin(2 * sympy.pi * x) * sympy.sin(3 * sympy.pi * y)
f_expr = -laplacian(u_expr, x, y) + u_expr**3 - u_expr

# %%
u_exact = GridFunction.from_function(sympy.lambdify([x, y], u_expr), grid)
forcing = GridFunction.from_function(sympy.lambdify([x, y], f_expr), grid)


# %%
def vec_to_fun(vec, bc, shape=()):
    data = reinsert_dofs(vec, bc)
    return GridFunction.from_array(data.reshape(*shape, *grid.shape), grid)


# %%
F_ = remove_dofs(vector_of_values(forcing), U_bc)

# %%
aspect = grid.h[1] / grid.h[0]
plt.imshow(u_exact.tabulate().T, aspect=aspect)
plt.show()


# %% [markdown]
# ## Gauss-Newton


# %%
def A(u, v):
    return dot(grad(u, "+"), grad(v, "+")) + (u**3 - u) * v


# %%
def jacobian(u):
    def linearization(w, v):
        return dot(grad(w, "+"), grad(v, "+")) + (3 * u**2 * w - w) * v

    return linearization


# %%
def R(u, v):
    return A(u, v) - forcing * v


# %%
w = U.trial_function()
v = U.test_function()


def compute_J(u):
    J = np.zeros((U.dim, U.dim))
    assemble(jacobian(u)(w, v), J, w, v)
    return remove_dofs(J, U_bc)


# %%
class LearningEntry(NamedTuple):
    iter: int
    loss: float
    error: float


# %%
def solution_error(solution, exact):
    return norm((solution - exact) * u_mask, "grad_h")


# %%
u_vec = remove_dofs(np.zeros(grid.shape).ravel(), U_bc)

# %%
log = []

# %%
for i in range(5):
    u = vec_to_fun(u_vec, U_bc)
    J = compute_J(u)
    r = assemble_residuum(R(u, v), v)
    r_ = remove_dofs(r, U_bc)
    du = -scipy.linalg.solve(J, r_)
    u_vec += du

    loss = np.dot(r_, r_)
    error = solution_error(u, u_exact)
    entry = LearningEntry(i + 1, loss, error)
    log.append(entry)

    print(
        f"Iter {i:>2}  loss: {loss:12.7g}, √loss: {np.sqrt(loss):12.7g}, "
        f"error: {error:12.7g}, ",
        f"Δu: {np.linalg.norm(du):12.7g}",
        flush=True,
    )

# %%
aspect = grid.h[1] / grid.h[0]
u = vec_to_fun(u_vec, U_bc)
plt.imshow(u.tabulate().T, aspect=aspect)
plt.show()

# %%
np.abs((u - u_exact).tabulate()).max()

# %%
until = -1
iters = [e.iter for e in log][:until]
loss = np.array([e.loss for e in log])[:until]
error = np.array([e.error for e in log])[:until]

plt.figure(figsize=(6.4, 4.8))
plt.plot(iters, np.sqrt(loss), "--", label=r"$\sqrt{\text{LOSS}}$")
plt.semilogy(iters, error, label=r"$\|u_\theta - u_\text{exact}\|_h$")
plt.xlabel("iteration")
plt.legend()
# plt.savefig("errors.pdf", bbox_inches="tight")
plt.show()
