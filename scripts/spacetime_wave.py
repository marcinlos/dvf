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
# # Space-time wave equation DPG formulation

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy

from dvf import (
    CompositeFunctionSpace,
    Dx,
    Dy,
    FunctionSpace,
    Grid,
    GridFunction,
    VectorFunctionSpace,
    assemble,
    integrate,
    random_function,
    reinsert_dofs,
    remove_dofs,
    select_dofs,
)

# %%
grid = Grid(10, 10)

# %%
Q = VectorFunctionSpace(grid, 1)
U = FunctionSpace(grid)
W = CompositeFunctionSpace(Q, U)


# %%
def Q_mask_fun(t_edge):
    def fun(i, j):
        return 0 if i == 0 or j == t_edge else 1

    return fun


# %%
def U_mask_fun(t_edge):
    def fun(i, j):
        edges_x = (0, grid.n[0])
        m = 0 if i in edges_x or j == t_edge else 1
        return m

    return fun


# %%
q_mask = GridFunction(Q_mask_fun(0), grid)
u_mask = GridFunction(U_mask_fun(0), grid)

q_star_mask = GridFunction(Q_mask_fun(grid.n[1]), grid)
u_star_mask = GridFunction(U_mask_fun(grid.n[1]), grid)

# %%
c = 1
Dt = Dy
grad_x = Dx
div_x = Dx

# %%
Q_bc = select_dofs(Q, q_mask, invert=True)
U_bc = select_dofs(U, u_mask, invert=True)
Q_star_bc = select_dofs(Q, q_star_mask, invert=True)
U_star_bc = select_dofs(U, u_star_mask, invert=True)

W_bc = W.combine_dofs(Q_bc, U_bc)
W_star_bc = W.combine_dofs(Q_star_bc, U_star_bc)


# %%
def A_form(q, u, r, v):
    Ar = Dt(q, "+") - c * grad_x(u, "-")
    Av = Dt(u, "+") - c * div_x(q, "+")
    return Ar * r + Av * v


def AT_form(r, v, q, u):
    Aq = -Dt(r, "-") + c * grad_x(v, "-")
    Au = -Dt(v, "-") + c * div_x(r, "+")
    return q * Aq + u * Au


# %% [markdown]
# ## Verification of the adjoint


# %%
def random_functions(Q_bc, U_bc):
    q = random_function(grid, shape=(1,), bc=Q_bc)
    u = random_function(grid, bc=U_bc)
    return q, u


# %%
q, u = random_functions(Q_bc, U_bc)
r, v = random_functions(Q_star_bc, U_star_bc)

lhs = integrate(A_form(q, u, r, v))
rhs = integrate(AT_form(r, v, q, u))

# %%
print(f"(Au, v)  = {lhs}")
print(f"(u, A*v) = {rhs}")
print(f"difference: {abs(rhs - lhs)}")


# %%
def dt_form(q, u):
    return Dt(q, "+") * q + Dt(u, "+") * u


# %%
integrate(dt_form(q, u))

# %%
integrate(A_form(q, u, q, u))


# %%
def dt_u2_form(q, u):
    return Dt(u * u + q * q, "+")


def dt_sq_form(q, u):
    return Dt(u, "+") * Dt(u, "+") + Dt(q, "+") * Dt(q, "+")


def A_norm_form(q, u):
    Ar = (Dt(q, "+") - c * grad_x(u, "-")) * q_star_mask
    Av = (Dt(u, "+") - c * div_x(q, "+")) * u_star_mask
    return Ar * Ar + Av * Av


def A_weird_form(q, u, r, v):
    Ar = Dt(q, "+") - c * grad_x(u, "-")  # + 0.5 * grid.h[1] * Dt(Dt(q, "+"), "-")
    Av = Dt(u, "+") - c * div_x(q, "+")  # + 0.5 * grid.h[1] * Dt(Dt(u, "+"), "-")
    return (
        Ar * r
        + Av * v
        + 0.5 * grid.h[1] * (Dt(q, "+") * Dt(r, "+") + Dt(u, "+") * Dt(v, "+"))
    )


# %%
integrate(dt_u2_form(q, u) - 2 * A_form(q, u, q, u) - grid.h[1] * dt_sq_form(q, u))

# %%
integrate(dt_u2_form(q, u) - 2 * A_weird_form(q, u, q, u))

# %%
u_norm = np.sqrt(integrate(u * u + q * q))

# %%
A_norm = np.sqrt(integrate(A_norm_form(q, u)))

# %%
u_dt_norm2 = integrate(dt_sq_form(q, u))

# %%
ht = grid.h[1]


# %%
def at_t(t):
    def fun(i, j):
        return 1 / ht if j == t else 0

    return GridFunction(fun, grid)


def between(t0, t1):
    def fun(i, j):
        return 1 if t0 <= j <= t1 else 0

    return GridFunction(fun, grid)


# %% [markdown]
# Check if $(Au, u)_k = (\nabla_{t+} u, u)_k$ holds

# %%
integrate((Dt(q, "+") * q + Dt(u, "+") * u - A_form(q, u, q, u)) * at_t(3))

# %% [markdown]
# Check time derivative of $\|u\|_k$

# %%
integrate(Dt(u**2 + q**2, "+") * at_t(2))

# %%
integrate(
    (2 * A_form(q, u, q, u) + grid.h[1] * (Dt(q, "+") ** 2 + Dt(u, "+") ** 2)) * at_t(2)
)

# %% [markdown]
# Check

# %%
k = 1

# %%
integrate((u**2 + q**2) * at_t(k))

# %%
integrate(Dt(u**2 + q**2, "+") * between(0, k - 1))


# %%
def slice(k):
    partA = 2 * A_form(q, u, q, u) * between(1, k - 1)
    part_dtu = ht * (Dt(q, "+") ** 2 + Dt(u, "+") ** 2) * between(0, k - 1)
    return partA + part_dtu


integrate(slice(k))

# %%
integrate(u**2 + q**2)

# %%
integrate(sum(ht * slice(i) for i in range(1, grid.n[1] + 1)))

# %%
integrate(
    sum(
        ht * 2 * A_form(q, u, q, u) * between(1, i - 1) for i in range(1, grid.n[1] + 1)
    )
)

# %%
integrate(
    sum(
        ht * ht * (Dt(q, "+") ** 2 + Dt(u, "+") ** 2) * between(0, i - 1)
        for i in range(1, grid.n[1] + 1)
    )
)

# %% [markdown]
# ## Matrix assembly

# %%
uu = W.trial_function()
vv = W.test_function()

q, u = uu.components
r, v = vv.components

A = scipy.sparse.dok_array((W.dim, W.dim))
Astar = scipy.sparse.dok_array((W.dim, W.dim))

# %%
assemble(A_form(q, u, r, v), A, uu, vv)

# %%
A = A.tocsr()

# %%
M = grid.cell_volume * scipy.sparse.eye_array(W.dim).tocsr()


# %% [markdown]
# ## Problem definition


# %%
class SmoothProblem:
    def u(x, t):
        return np.pi * np.sin(np.pi * x) * np.sin(2 * np.pi * t)

    def q(x, t):
        return c * np.pi * np.cos(np.pi * x) * np.sin(np.pi * t) ** 2

    def g(x, t):
        return np.zeros_like(x)

    def f(x, t):
        return (
            np.pi**2
            * np.sin(np.pi * x)
            * (2 * np.cos(2 * np.pi * t) + c**2 * np.sin(np.pi * t) ** 2)
        )


# %% jupyter={"source_hidden": true}
class WaveSource:
    def u(x, t):
        return np.zeros_like(x)

    def q(x, t):
        return np.zeros_like(x)

    def g(x, t):
        return np.zeros_like(x)

    def f(x, t):
        return np.exp(-50 * (x - 0.5) ** 2 - 50 * t**2)


# %%
X, Y = grid.points
wave = SmoothProblem

u_data = wave.u(X, Y)
exact_u = GridFunction.from_array(u_data, grid)

q_data = wave.q(X, Y)
exact_q = GridFunction.from_array(q_data, grid)

g_data = wave.g(X, Y)
g_fun = GridFunction.from_array(g_data, grid)

f_data = wave.f(X, Y)
f_fun = GridFunction.from_array(f_data, grid)


# %%
def plot_wave(q, u, title, file=None):
    q_vals = q.tabulate()
    u_vals = u.tabulate()

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(12, 5))
    aspect = grid.h[1] / grid.h[0]
    kwargs = dict(aspect=aspect)  # , vmin=-0.1, vmax=0.1)

    img = axs[0].imshow(np.flipud(q_vals.T), **kwargs)
    axs[0].set_title(r"$q$")
    fig.colorbar(img, ax=axs[0], shrink=0.7)

    img = axs[1].imshow(np.flipud(u_vals.T), **kwargs)
    axs[1].set_title(r"$u$")
    fig.colorbar(img, ax=axs[1], shrink=0.7)

    fig.suptitle(title)

    if file is not None:
        plt.savefig(file, bbox_inches="tight")

    plt.show()


# %%
plot_wave(exact_q, exact_u, "Exact solution")

# %% [markdown]
# ## Solving discrete formulation

# %%
A_ = remove_dofs(A, trial_dofs=W_bc, test_dofs=W_star_bc)
M_ = remove_dofs(M, W_bc)

# %%
rhs_full = grid.cell_volume * np.concat([f.tabulate().ravel() for f in (g_fun, f_fun)])
rhs_vec = remove_dofs(rhs_full, W_star_bc)


# %%
def funs_from_vec(vec):
    data = reinsert_dofs(vec, W_bc)
    data_q = data[: Q.dim]
    data_u = data[Q.dim :]
    q = GridFunction.from_array(data_q.reshape(*grid.shape), grid)
    u = GridFunction.from_array(data_u.reshape(*grid.shape), grid)
    return q, u


# %% [markdown]
# ### Kernel verification

# %%
kernel = scipy.linalg.null_space(A_.todense())

# %%
print(kernel.shape)

# %%
# idx = 0
# for idx in range(kernel.shape[1]):
#     vec = kernel[:, idx]
#     q, u = funs_from_vec(vec)

#     plot_wave(q, u, f"kernel {idx}")

# %% [markdown]
# ### Solver invocation

# %%
solution_vec = scipy.sparse.linalg.spsolve(A_, rhs_vec)

# %%
solution_q, solution_u = funs_from_vec(solution_vec)

# %%
plot_wave(solution_q, solution_u, "Discrete solution")

# %%
err_q = (solution_q - exact_q) * q_mask
err_u = (solution_u - exact_u) * u_mask
plot_wave(err_q, err_u, "Discrete solution error")

# %%
G_ = M_ + A_ @ A_.T / grid.cell_volume

# %%
# R_ = A_.T.todense() @ np.linalg.solve(G_.todense(), A_.todense())

G_inv = scipy.sparse.linalg.factorized(G_.tocsc())
A_inv = scipy.sparse.linalg.factorized(A_.tocsc())
AT_inv = scipy.sparse.linalg.factorized(A_.T.tocsc())


def R_matvec(v):
    return A_.T @ G_inv(A_ @ v)


def Rinv_matvec(v):
    return A_inv(G_ @ AT_inv(v))


R_ = scipy.sparse.linalg.LinearOperator(matvec=R_matvec, shape=A_.shape)
R_inv = scipy.sparse.linalg.LinearOperator(matvec=Rinv_matvec, shape=A_.shape)

# %%
# w, vr = scipy.linalg.eigh(R_, M_.todense())
w, vr = scipy.sparse.linalg.eigsh(R_, M=M_, sigma=0, which="LM", OPinv=R_inv)

# %%
print(f"largest imag: {np.max(np.abs(np.imag(w)))}")
w = np.real(w)
print(f"minimum: {np.min(w)}")
print(np.sort(w)[:10])
w = np.abs(w)

# %%
order = np.argsort(w)
imin = order[0]
imax = order[-1]

# %%
vals = np.sqrt(w)

# %%
print(f"inf-sup constant: {vals[imin]}")

# %%
print(f"continuity constant: {vals[imax]}")

# %%
plt.scatter(np.arange(order.size), vals[order], marker=".")
plt.xlabel("eigenvalue number")
plt.ylabel("eigenvalue")
plt.ylim(bottom=0)
# plt.savefig("spectrum.pdf", bbox_inches="tight")
plt.show()

# %%
betas = {
    10: 0.0062256339819853354,
    15: 0.0001629756116503361,
    20: 3.834655404598689e-06,
    25: 8.441510630162633e-08,
    30: 2.9070771288013025e-09,
}

# %%
plt.semilogy(betas.keys(), betas.values())
plt.show()

# %%
betas = {
    10: 0.015330265955641244,
    15: 0.0012499666141974515,
    20: 2.8556726524788463e-05,
    25: 1.1036781884561912e-06,
    30: 4.1086424327053176e-08,
}

# %%
plt.semilogy(betas.keys(), betas.values())
plt.show()
