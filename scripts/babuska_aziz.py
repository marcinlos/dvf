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
# # Babuska-Aziz inequality

# %% jupyter={"source_hidden": true}
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from dvf import (
    Edge,
    FunctionSpace,
    Grid,
    GridFunction,
    VectorFunctionSpace,
    assemble,
    div,
    grad,
    lift_to_gridfun,
    norm,
    remove_dofs,
)

# %%
grid = Grid(10, 10)


# %% [markdown]
# We start by constructing matrices of relevant bilinear forms.


# %%
def build_matrices(grid):
    U = VectorFunctionSpace(grid, 2)
    P = FunctionSpace(grid)

    u = U.trial_function()
    v = U.test_function()

    p = P.trial_function()
    q = P.test_function()

    A = np.zeros((P.dim, U.dim))
    M = np.zeros((P.dim, P.dim))
    G = np.zeros((U.dim, U.dim))

    dot = lift_to_gridfun(np.vdot)
    assemble(div(u, "-") * q, A, u, q)
    assemble(p * q, M, p, q)
    assemble(dot(grad(u, "+"), grad(v, "+")), G, u, v)

    U_bc = [f.index for idx in grid.boundary() for f in U.basis_at(idx)]
    P_bc = [
        f.index for idx in grid.boundary(Edge.LEFT | Edge.TOP) for f in P.basis_at(idx)
    ] + [grid.ravel_index((grid.nx, grid.ny))]
    A = remove_dofs(A, trial_dofs=U_bc, test_dofs=P_bc)
    M = remove_dofs(M, P_bc)
    G = remove_dofs(G, U_bc)

    return A, M, G, U, P, U_bc, P_bc


A, M, G, U, P, U_bc, P_bc = build_matrices(grid)

# %% [markdown]
# We shall construct a right inverse $B$ of $A$, i.e. matrix $B$ such that $A B = M$,
# with the additional property that $(\nabla,h)$-norm of $Bq$ is minimal for each $q$.
# It is given by
# \begin{equation}
# B = G^{-1} A^T R^+ M
# \end{equation}
# where $R = A G^{-1} A^T$, and $R^+$ denotes its Moore-Penrose pseudoinverse.

# %%
G_inv = np.linalg.inv(G)
R = A @ G_inv @ A.T
R_inv = np.linalg.pinv(R)
B = G_inv @ A.T @ R_inv @ M

# %% [markdown]
# We can test it on a random pressure function.

# %%
p_data = np.random.rand(*grid.shape)
p_data.flat[P_bc] = 0

# %% [markdown]
# First, we remove unnecessary degrees of freedom.

# %%
p_vec = np.delete(np.ravel(p_data), P_bc)

# %% [markdown]
# Then, we ensure the mean pressure is zero.

# %%
mean = np.mean(p_vec)
print(f"Mean before adjustment: {mean}")
p_vec -= mean
new_mean = np.mean(p_vec)
print(f"Mean after adjustment: {new_mean}")

# %% [markdown]
# Now we compute the minimal norm inverse.

# %%
v_vec = B @ p_vec

# %% [markdown]
# Let us convert pressure and velocity vectors to `GridFunction`s for plotting.

# %%
p = GridFunction.from_array(p_data, grid)

# %%
vx_vec, vy_vec = np.split(v_vec, 2)
vx_data = np.zeros(grid.shape)
vy_data = np.zeros(grid.shape)
vx_data[1:-1, 1:-1] = vx_vec.reshape((grid.nx - 1, grid.ny - 1))
vy_data[1:-1, 1:-1] = vy_vec.reshape((grid.nx - 1, grid.ny - 1))

v_data = np.stack([vx_data, vy_data])
v = GridFunction.from_array(v_data, grid)

unit_p_data = np.ones(grid.shape)
unit_p_data.flat[P_bc] = 0
unit_p = GridFunction.from_array(unit_p_data, grid)

# %% [markdown]
# Pressure $p$ and $\nabla \cdot v$ differ by a constant equal to the mean of $p$.

# %%
difference = div(v, "-") - p
print(f"error: {norm(difference + mean * unit_p, "h")}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
aspect = grid.h[1] / grid.h[0]

axs[0].imshow(p.tabulate().T, aspect=aspect)
axs[0].set_title(r"original $p$")

axs[1].imshow(div(v, "-").tabulate().T, aspect=aspect)
axs[1].set_title(r"$\nabla \cdot v$")

axs[2].imshow(difference.tabulate().T, aspect=aspect)
axs[2].set_title(r"$\nabla \cdot v - p$")
plt.show()

# %% [markdown]
# ## By Lagrange multipliers

# %% [markdown]
# First way to compute the *inf-sup* constant is by fixing $q$ and maximizing $q^T A v$
# under constraint $v^T G v = 1$, and then minimizing the resulting functional, which
# depends only on $q$. This leads to the problem of minimizing the Rayleigh quotient
# \begin{equation}
# \frac{q^T R q}{q^T M q}
# \end{equation}
# which in turn requires computing the smallest (nonzero) generalized eigenvalue of the
# problem
# \begin{equation}
# R q = \lambda M q
# \end{equation}

# %%
w, vr = scipy.linalg.eig(R, M)

# %% [markdown]
# All the eigenvalues should be real and non-negative.

# %%
print(f"largest imag: {np.max(np.abs(np.imag(w)))}")
w = np.real(w)
print(f"minimum: {np.min(w)}")
print(np.sort(w)[:10])
w = np.abs(w)

# %% [markdown]
# Smallest eigenvalue is $\lambda = 0$, corresponding to the one-dimensional subspace of
# constant pressures.

# %%
order = np.argsort(w)
imin = order[1]  # skip 0
imax = order[-1]

# %% [markdown]
# The sought *inf-sup* constant is the square root of the smallest nonzero eigenvalue.

# %%
vals = np.sqrt(w)

# %%
print(f"inf-sup constant: {vals[imin]}")

# %%
vals[order[:10]]

# %% [markdown]
# Generalized eigenfunctions form an orthonormal (with respect to $M$) basis of the
# pressure space, so all the eigenfunctions corresponding to non-zero eigenvalues are
# orthogonal to constant pressures, i.e. they have zero mean pressure, so they lie in
# the image of $A$.

# %%
print(f"First mean: {np.sum(vr[:, order[0]])}")
print("other means:")
np.sum(vr[:, order[1:10]], axis=0)

# %% [markdown]
# ## By the minimal right inverse of $A$

# %% [markdown]
# Another way to find a lower bound of the *inf-sup* constant is to use the
# aforementioned minimum-norm right inverse of $A$. This approach also leads to a
# generalized eigenvalue problem of the form
# \begin{equation}
# R^+Mq = \lambda q
# \end{equation}
# This time, we are interested in the largest eigenvalue.

# %%
w, vr = scipy.linalg.eig(R_inv @ M)

# %% [markdown]
# All the eigenvalues should be real and non-negative.

# %%
print(f"largest imag: {np.max(np.abs(np.imag(w)))}")
w = np.real(w)
print(f"minimum: {np.min(w)}")
print(np.sort(w)[:10])
w = np.abs(w)

# %%
order = np.argsort(w)
imax = order[-1]

# %%
vals = 1 / np.sqrt(w)

# %%
print(f"inf-sup constant (lower bound): {vals[imax]}")

# %%
vals[order[:-10:-1]]


# %% [markdown]
# ## Dependence of *inf-sup* constant on $h$


# %%
def inf_sup_const(n):
    # print(f"{n} ", end=" ", flush=True)
    grid = Grid(n, n)
    A, M, G, *_ = build_matrices(grid)
    G_inv = np.linalg.inv(G)
    R = A @ G_inv @ A.T

    w, vr = scipy.linalg.eig(R, M)
    w = np.abs(np.real(w))
    order = np.argsort(w)
    imin = order[1]  # skip 0
    return np.sqrt(w[imin])


# %%
n_max = 20
ns = np.arange(3, n_max + 1, 2)
gen = (inf_sup_const(n) for n in ns)
vals = list(tqdm(gen, total=len(ns)))

# %%
plt.plot(ns, vals)
plt.ylabel("inf-sup constant")
plt.xlabel("grid size (N)")
plt.title("inf-sup constant as a function of grid size")
plt.show()
