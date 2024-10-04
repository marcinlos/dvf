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
import numpy as np

import dvf
from dvf import (
    Edge,
    Grid,
    GridFunction,
    div,
    dx,
    integrate,
    integrate_bd,
    norm,
    shift,
)

# %%
grid = Grid(4)


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
