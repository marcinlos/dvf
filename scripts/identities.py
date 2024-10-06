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
# # Testing discrete integration identities

# %% [markdown]
# We start by introducing two measures:
# * measure $\mu_h$ on $\Omega_h$, given by $\mu_h(A) = h^2|A|$
# * measure $\sigma_h$ on $\partial\Omega_h$, given by $\sigma_h(A) = h|A|$
#
# Using these definitions,
# $$
# \int f\,d\mu_h = h^2 \sum_{p \in \Omega_h}f(p),
# \quad
# \int g\,d\sigma_h = h \sum_{p \in \partial \Omega_h}f(p)
# $$

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy

import dvf
from dvf import (
    Edge,
    FunctionSpace,
    Grid,
    GridFunction,
    assemble,
    div,
    dx,
    integrate,
    integrate_bd,
    lift_to_gridfun,
    nabla,
    norm,
    shift,
)

# %%
grid = Grid(20)


# %%
def random_function(grid, *, shape=(), zero_bd=Edge.NONE):
    data = np.random.rand(*grid.shape, *shape)
    ids = np.fromiter(grid.boundary(zero_bd), dtype=np.dtype((np.intp, 2)))
    data[*ids.T] = 0
    return GridFunction.from_array(data, grid)


# %%
normal = GridFunction(lambda *idx: grid.boundary_normal(idx), grid)
normal_x = dvf.apply_to_gridfun(lambda n: n[0], normal)
normal_y = dvf.apply_to_gridfun(lambda n: n[1], normal)

# %% [markdown]
# ## Divergence theorem
#
# Given a vector field $F \colon \Omega_h \rightarrow \mathbb{R}^2$, we have
# $$
# \int \nabla_{+} \cdot F \, d\mu_h = \int F \cdot \hat{n} \, d\sigma_h
# $$

# %%
f = random_function(grid, shape=(2,), zero_bd=dvf.Edge.NONE)

# %%
lhs = integrate(div(f, "+"))

# %%
fn = dvf.apply_to_gridfun(np.dot, f, normal)
rhs = integrate_bd(fn)

# %%
print(f"LHS: {lhs}")
print(f"RHS: {rhs}")
print(f"difference: {abs(lhs - rhs)}")

# %% [markdown]
# ## Integration by parts

# %% [markdown]
# Given $f, g \in D_h$ we have
# \begin{equation}
# \int (\nabla_{x+}f)g \,d\mu_h =
# - \int f \nabla_{x-}g\,d\mu_H + \int fg\, \hat{n}_x\,d\sigma_h
# \end{equation}

# %%
f = random_function(grid)
g = random_function(grid)

lhs = integrate(dx(f, "+") * g)
rhs = -integrate(f * dx(g, "-")) + integrate_bd(f * g * normal_x)

# %%
print(f"LHS: {lhs}")
print(f"RHS: {rhs}")
print(f"difference: {abs(lhs - rhs)}")

# %% [markdown]
# ## Product rule
#
# Given $f, g \in D_h$ we have
#
# $$
# \nabla_{x+}(fg) = \tau_x f\, \nabla_{x+}g + (\nabla_{x+}f)g
# $$

# %%
f = random_function(grid)
g = random_function(grid)

lhs = dx(f * g, "+")
rhs = shift(f, "x") * dx(g, "+") + dx(f, "+") * g

# %%
diff = lhs - rhs
print(f"difference: {np.max(np.abs(diff.tabulate()))}")


# %% [markdown]
# ## Norm equivalence

# %% [markdown]
# We are interested in the bounds on the quotient
# $$
# Q(f) = \frac{\|f\|_h}{\|f\|_{\nabla,h}}
# $$


# %%
def norm_quotient(f):
    return norm(f, "h") / norm(f, "grad_h")


# %%
cs = [norm_quotient(random_function(grid, zero_bd=Edge.ALL)) for _ in range(100)]
c, C = min(cs), max(cs)
print(f"{c} < q < {C}")


# %%
def check_eigen(m, n):
    def f(x, y):
        return np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

    q = norm_quotient(GridFunction.from_function(f, grid))
    print(f"Quotient for Δ eigenfunction ({m}, {n}): {q}")


check_eigen(1, 1)
check_eigen(3, 3)

# %%
print(f"Lower bound (paper): {grid.h / (2*np.sqrt(2))}")

# %% [markdown]
# To get the exact values, we need to look at the generalized eigenvalue problem
# \begin{equation}
# Ax = \lambda M x
# \end{equation}
# where $A$ is the Gram matrix of the $(\nabla, h)$-scalar product, and $M$ -- of the
# $h$-scalar product on $D_{0, h}$.

# %%
U = FunctionSpace(grid)
u = U.trial_function()
v = U.test_function()

A = np.zeros((U.dim, U.dim))
M = np.zeros((U.dim, U.dim))

assemble(u * v, M, u, v)
dot = lift_to_gridfun(np.dot)
assemble(dot(nabla(u, "+"), nabla(v, "+")), A, u, v)

to_remove = [grid.ravel_index(idx) for idx in grid.boundary()]
A = np.delete(np.delete(A, to_remove, axis=0), to_remove, axis=1)
M = np.delete(np.delete(M, to_remove, axis=0), to_remove, axis=1)

# %%
w, vr = scipy.linalg.eig(A, M)

# %% [markdown]
# Eigenvalues should all be real and positive.

# %%
print(f"Real? {np.all(np.real(w) == w)}")
w = np.real(w)
print(f"Positive? {np.all(w > 0)}")

# %%
vals = 1 / np.sqrt(w)

# %%
order = np.argsort(vals)
imin = order[0]
imax = order[-1]

# %%
print(f"c = {vals[imin]}")
print(f"C = {vals[imax]}")


# %%
def on_grid(eigfun):
    data = np.zeros(grid.shape)
    data[1:-1, 1:-1] = eigfun.reshape((grid.n - 1, grid.n - 1))
    return data


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(on_grid(vr[:, imin].T))
axs[0].set_title(f"minimum q = {vals[imin]:6.5f}")

axs[1].imshow(on_grid(vr[:, imax].T))
axs[1].set_title(f"maximum q = {vals[imax]:6.5f}")

N = w.size
k = int(N * 0.88)
idx = order[k]
axs[2].imshow(on_grid(vr[:, idx].T))
axs[2].set_title(f"eig {k}, q = {vals[idx]:6.5f}")
