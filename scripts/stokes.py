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
# # Discrete Stokes formulation

# %% jupyter={"source_hidden": true}
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch import nn

from dvf import (
    CompositeFunctionSpace,
    FunctionSpace,
    Grid,
    GridFunction,
    TensorFunctionSpace,
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
grid = Grid(10, 10)

# %% [markdown]
# ### Spaces and boundary conditions

# %%
S = TensorFunctionSpace(grid, (2, 2))
U = VectorFunctionSpace(grid, 2)
P = FunctionSpace(grid)
W = CompositeFunctionSpace(S, U, P)


# %%
def S_mask_fun(i, j):
    mdx = 0 if i == 0 else 1
    mdy = 0 if j == 0 else 1
    return np.array([[mdx, mdy], [mdx, mdy]])


# %%
def U_mask_fun(i, j):
    edges_x = (0, grid.n[0])
    edges_y = (0, grid.n[1])
    m = 0 if i in edges_x or j in edges_y else 1
    return np.array([m, m])


# %%
def P_mask_fun(i, j):
    return 0 if 0 in (i, j) or (i, j) == (grid.n[0], grid.n[1]) else 1


# %%
s_mask = GridFunction(S_mask_fun, grid)
u_mask = GridFunction(U_mask_fun, grid)
p_mask = GridFunction(P_mask_fun, grid)

# %%
S_bc = select_dofs(S, s_mask, invert=True)
U_bc = select_dofs(U, u_mask, invert=True)
P_bc = select_dofs(P, p_mask, invert=True)

W_bc = W.combine_dofs(S_bc, U_bc, P_bc)


# %%
def fix_pressure(p):
    return p - p_mask * integrate(p) / integrate(p_mask)


# %% [markdown]
# ### Bilinear forms defining the problem and norms

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
        dot(-div(sigma, "+") + grad(p, "+"), v)
        + dot(div(u, "-"), q)
        + dot(sigma - grad(u, "-"), tau)
    )


def AT_form(tau, v, q, sigma, u, p):
    return (
        dot(div(tau, "+") - grad(q, "+"), u)
        + dot(-div(v, "-"), p)
        + dot(tau + grad(v, "-"), sigma)
    )


def L2_product(sigma, u, p, tau, v, q):
    return dot(sigma, tau) + dot(u, v) + p * q


def AT_product(sigma, u, p, tau, v, q):
    return (
        dot(
            pi0(div(sigma, "+") - grad(p, "+")),
            pi0(div(tau, "+") - grad(q, "+")),
        )
        + dot(div(u, "-"), div(v, "-"))
        + dot(sigma + grad(u, "-"), tau + grad(v, "-"))
    )


def AT_graph_product(*args):
    return L2_product(*args) + AT_product(*args)


# %% [markdown]
# ### Verification of the adjoint
#
# We use random functions to avoid assembling the matrix of $A^*$, which is expensive
# for large grids.


# %%
def random_functions():
    sigma = random_function(grid, shape=(2, 2), bc=S_bc)
    u = random_function(grid, shape=(2,), bc=U_bc)
    p = fix_pressure(random_function(grid, bc=P_bc))

    return sigma, u, p


# %%
sigma, u, p = random_functions()
tau, v, q = random_functions()

# %%
lhs = integrate(A_form(sigma, u, p, tau, v, q))
rhs = integrate(AT_form(tau, v, q, sigma, u, p))

# %% [markdown]
# $(A\boldsymbol{u}, \boldsymbol{v})_h = (\boldsymbol{u}, A^* \boldsymbol{v})_h$ should
# hold for all $\boldsymbol{u}$, $\boldsymbol{v}$.

# %%
print(f"(Au, v)  = {lhs}")
print(f"(u, A*v) = {rhs}")
print(f"difference: {abs(rhs - lhs)}")

# %% [markdown]
# ### Matrix assembly

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

# %%
assemble(A_form(sigma, u, p, tau, v, q), A, uu, vv)

# %% [markdown]
# The Gram matrix $M$ of $(\cdot,\cdot)_h$ is just $h^2 I$.

# %%
# assemble(L2_product(sigma, u, p, tau, v, q), M, uu, vv)
M = grid.cell_volume * np.identity(W.dim)


# %% [markdown]
# All of these can be defined in terms of $A$ and $M$.

# %%
# assemble(AT_form(tau, v, q, sigma, u, p), AT, vv, uu)
# assemble(AT_product(sigma, u, p, tau, v, q), AT_p, uu, vv)
# assemble(AT_graph_product(sigma, u, p, tau, v, q), AT_graph, uu, vv)


# %% [markdown]
# To avoid fiddling with deactivating degrees of freedom by modifying rows/columns, we
# shall simply remove them from the matrices.


# %%
A_ = remove_dofs(A, W_bc)
AT_ = remove_dofs(AT, W_bc)
AT_p_ = remove_dofs(AT_p, W_bc)
M_ = remove_dofs(M, W_bc)
AT_graph_ = remove_dofs(AT_graph, W_bc)


# %% [markdown]
# ### Manufactured solution problem definition


# %% jupyter={"source_hidden": true}
class StokesManufactured:
    @staticmethod
    def rhs1(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)
        px = (
            ex
            * (y - 1)
            * y
            * (
                x**4 * (yy - y + 12)
                + 6 * x**3 * (yy - y - 4)
                + xx * (yy - y + 12)
                - 8 * x * (y - 1) * y
                + 2 * (y - 1) * y
            )
        )

        Lux = (
            2
            * ex
            * (
                x**4 * (2 * y**3 - 3 * yy + 13 * y - 6)
                + 6 * x**3 * (2 * y**3 - 3 * yy - 3 * y + 2)
                + xx * (2 * y**3 - 3 * yy + 13 * y - 6)
                - 8 * x * y * (2 * yy - 3 * y + 1)
                + 2 * y * (2 * yy - 3 * y + 1)
            )
        )

        res = -Lux + px
        return res

    @staticmethod
    def rhs2(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)

        py = (
            2
            * (2 * y - 1)
            * (
                ex
                * (
                    x**4 * (yy - y + 6)
                    + 2 * x**3 * (yy - y - 18)
                    + xx * (-5 * yy + 5 * y + 114)
                    + 2 * x * (yy - y - 114)
                    + 228
                )
                - 228
            )
        )

        Luy = -ex * (
            x**4 * (y**4 - 2 * y**3 + 13 * yy - 12 * y + 2)
            + 2 * x**3 * (5 * y**4 - 10 * y**3 + 17 * yy - 12 * y + 2)
            + xx * (19 * y**4 - 38 * y**3 - 41 * yy + 60 * y - 10)
            + x * (-6 * y**4 + 12 * y**3 + 18 * yy - 24 * y + 4)
            - 6 * (y - 1) ** 2 * yy
        )

        res = -Luy + py
        return res

    @staticmethod
    def u1(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)
        res = 2 * ex * (-1 + x) ** 2 * xx * (yy - y) * (-1 + 2 * y)
        return res

    @staticmethod
    def u2(x, y):
        yy = y * y
        ex = np.exp(x)
        res = -ex * (-1 + x) * x * (-2 + x * (3 + x)) * (-1 + y) * (-1 + y) * yy
        return res

    @staticmethod
    def u1_dx(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)
        res = 2 * ex * x * (2 - 5 * x + 2 * xx + x * x * x) * y * (1 - 3 * y + 2 * yy)
        return res

    @staticmethod
    def u1_dy(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)
        res = 2 * ex * (-1 + x) * (-1 + x) * xx * (1 - 6 * y + 6 * yy)
        return res

    @staticmethod
    def u2_dx(x, y):
        yy = y * y
        ex = np.exp(x)
        res = (
            -ex
            * (2 - 8 * x + x * x + 6 * x * x * x + x * x * x * x)
            * (-1 + y)
            * (-1 + y)
            * yy
        )
        return res

    @staticmethod
    def u2_dy(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)
        res = -2 * ex * x * (2 - 5 * x + 2 * xx + x * x * x) * y * (1 - 3 * y + 2 * yy)
        return res

    @staticmethod
    def p(x, y):
        xx = x * x
        yy = y * y
        ex = np.exp(x)
        res = (
            -424
            + 156 * 2.718
            + (yy - y)
            * (
                -456
                + ex
                * (
                    456
                    + xx * (228 - 5 * (yy - y))
                    + 2 * x * (-228 + (yy - y))
                    + 2 * x**3 * (-36 + (yy - y))
                    + x**4 * (12 + (yy - y))
                )
            )
        )
        return res


# %%
X, Y = grid.points
stokes = StokesManufactured

rhs_data = np.stack([stokes.rhs1(X, Y), stokes.rhs2(X, Y)])
rhs_f = GridFunction.from_array(rhs_data, grid)

exact_p_data = stokes.p(X, Y)
exact_p = GridFunction.from_array(exact_p_data, grid)

exact_u_data = np.array([stokes.u1(X, Y), stokes.u2(X, Y)])
exact_u = GridFunction.from_array(exact_u_data, grid)

exact_sigma_blocks = [
    [stokes.u1_dx(X, Y), stokes.u1_dy(X, Y)],
    [stokes.u2_dx(X, Y), stokes.u2_dy(X, Y)],
]
exact_sigma_data = np.array(exact_sigma_blocks)
exact_sigma = GridFunction.from_array(exact_sigma_data, grid)


# %%
def plot_stokes(sigma, u, p, title, file=None):
    sigma_vals = sigma.tabulate()
    u_vals = u.tabulate()
    p_vals = p.tabulate()

    fig = plt.figure(layout="constrained", figsize=(14, 8.7))
    subfigs = fig.subfigures(nrows=2, hspace=-0.4)

    axs = subfigs[0].subplots(ncols=3, sharey=True)

    items = [(u_vals[0], r"$u_1$"), (u_vals[1], r"$u_2$"), (p_vals, "$p$")]
    for i, (data, label) in enumerate(items):
        img = axs[i].imshow(np.flipud(data.T))
        axs[i].set_title(label)
        subfigs[0].colorbar(img, ax=axs[i], shrink=0.7)

    axs = subfigs[1].subplots(ncols=4, sharey=True)

    for i, j in np.ndindex(2, 2):
        ax = axs[2 * i + j]
        img = ax.imshow(np.flipud(sigma_vals[i, j].T))
        ax.set_title(rf"$\sigma_{{{i + 1}{j + 1}}}$")
        subfigs[1].colorbar(img, ax=ax, location="bottom", shrink=0.9)

    fig.suptitle(title)
    if file is not None:
        plt.savefig(file, bbox_inches="tight")
    plt.show()


# %%
plot_stokes(exact_sigma, exact_u, exact_p, "Exact solution plotted on the grid")

# %%
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(15, 10))

sigma = grad(exact_u, "-")
difference = (exact_sigma - grad(exact_u, "-")) * s_mask

sigma_vals = sigma.tabulate()
difference_vals = difference.tabulate()

for i, j in np.ndindex(2, 2):
    ax = axs[0, 2 * i + j]
    img = ax.imshow(np.flipud(sigma_vals[i, j].T))
    var = ["x", "y"][j]
    ax.set_title(rf"$\nabla_{{{var}-}}u_{{{i + 1}}}$")
    fig.colorbar(img, ax=ax, location="bottom", shrink=0.9)

    ax = axs[1, 2 * i + j]
    img = ax.imshow(np.flipud(difference_vals[i, j].T))
    ax.set_title(rf"$\sigma_{{{i + 1}{j + 1}}} - \nabla_{{{var}-}}u_{{{i + 1}}}$")
    fig.colorbar(img, ax=ax, location="bottom", shrink=0.9)

fig.suptitle(r"Components of $\nabla_{-} u$ and how it differs from exact $\sigma$")

plt.tight_layout()
plt.show()

# %% [markdown]
# Relative differences between components of exact $\sigma$ and $\nabla_{-} u$ are quite
# large for the off-diagonal components.

# %%
norm(exact_sigma - grad(exact_u, "-"), "h") / norm(exact_sigma, "h")


# %% [markdown]
# This makes sense, since $\nabla_{-} u$ is not defined on the left ($\nabla_{x-}$) or
# top ($\nabla_{y-}$) edges of the grid. Removing these parts of the boundary from the
# error computation significantly reduces the error.


# %%
norm((exact_sigma - grad(exact_u, "-")) * s_mask, "h") / norm(exact_sigma * s_mask, "h")


# %% [markdown]
# ## Solving discrete formulation

# %% [markdown]
# We start by preparing the right-hand side.


# %%
def vector_of_values(*funs):
    return np.concat([np.ravel(f.tabulate()) for f in funs])


rhs_vec = remove_dofs(vector_of_values(S.zero_fun, rhs_f, P.zero_fun), W_bc)

# %%
# Too slow for large grids
# solution_data = np.linalg.pinv(A_) @ M_ @ rhs_vec
# solution_data, *_ = np.linalg.lstsq(A_, M_ @ rhs_vec)

# %% [markdown]
# Matrix `A_` is singular, but the system has solutions as long as `rhs_q_vec` has zero
# mean.

# %%
lu, piv = scipy.linalg.lu_factor(A_)
solution_data = scipy.linalg.lu_solve((lu, piv), M_ @ rhs_vec)

# %%
first_U = S.dim - len(S_bc)
first_P = first_U + U.dim - len(U_bc)

solution_sigma_vec = solution_data[:first_U]
solution_u_vec = solution_data[first_U:first_P]
solution_p_vec = solution_data[first_P:]

# %% [markdown]
# Since `A_` has non-trivial kernel consisting of constant pressures, we need remove the
# mean pressure from our solution.

# %%
solution_p_vec -= np.mean(solution_p_vec)


# %%
def vec_to_fun(vec, bc, shape=()):
    data = reinsert_dofs(vec, bc)
    return GridFunction.from_array(data.reshape(*shape, *grid.shape), grid)


solution_sigma = vec_to_fun(solution_sigma_vec, S_bc, (2, 2))
solution_u = vec_to_fun(solution_u_vec, U_bc, (2,))
solution_p = vec_to_fun(solution_p_vec, P_bc)

# %%
plot_stokes(
    solution_sigma,
    solution_u,
    solution_p,
    "Discrete formulation solution",
    file="transposed_rhs.pdf",
)

# %%
diff_u = solution_u - exact_u
diff_sigma = (solution_sigma - exact_sigma) * s_mask
diff_p = (solution_p - exact_p) * p_mask
title = "Difference between the exact solution and the discrete formulation solution"
plot_stokes(diff_sigma, diff_u, diff_p, title)

# %%
norm(diff_sigma, "h") / norm(exact_sigma, "h")

# %% [markdown]
# Removing contributions of the edges that should be removed:

# %%
norm(diff_sigma * s_mask, "h") / norm(exact_sigma * s_mask, "h")

# %%
norm(diff_u, "h") / norm(exact_u, "h")

# %%
norm(diff_p, "h") / norm(exact_p, "h")


# %% [markdown]
# Removing the DoFs that are not part of discrete pressure domain:


# %%
norm(diff_p * p_mask, "h") / norm(exact_p * p_mask, "h")

# %% [markdown]
# Ensuring zero mean of the exact pressure:

# %%
fixed_exact_p = fix_pressure(exact_p)
norm(fixed_exact_p - solution_p, "h") / norm(fixed_exact_p, "h")

# %%
tau, v, q = random_functions()
lhs = integrate(A_form(solution_sigma, solution_u, solution_p, tau, v, q))
lhs_exact = integrate(A_form(exact_sigma, exact_u, exact_p, tau, v, q))
rhs = integrate(dot(rhs_f, v))

print(f"(f, v) = {rhs}")
print()
print(f"(A solution, v)  = {lhs}")
print(f"difference: {abs(rhs - lhs)}")
print()
print(f"(A exact, v)  = {lhs_exact}")
print(f"difference: {abs(rhs - lhs_exact)}")

# %% [markdown]
# ### Error estimation

# %% [markdown]
# Gram matrix of the "ideal" scalar product is $G = A M^{-1} A^*$

# %%
G_ = A_ @ A_.T / grid.cell_volume

# %%
ex_sigma, ex_u, ex_p = random_functions()

# %%
on_v = -div(ex_sigma, "+") + grad(ex_p, "+") - rhs_f
on_tau = ex_sigma - grad(ex_u, "-")
on_q = div(ex_u, "-")

ex_rhs_vec = grid.cell_volume * vector_of_values(on_tau, on_v, on_q)
residuum_vec = remove_dofs(ex_rhs_vec, W_bc)
residuum_rep = np.linalg.solve(G_, residuum_vec)

first_U = S.dim - len(S_bc)
first_P = first_U + U.dim - len(U_bc)
residuum_rep[first_P:] -= np.mean(residuum_rep[first_P:])

residuum_norm = np.sqrt(np.dot(residuum_vec, residuum_rep))
print(f"Residuum norm: {residuum_norm}")

# %%
error = np.linalg.norm(
    np.concat(
        [
            [norm(ex_p - solution_p, "h")],
            norm(ex_u - solution_u, "h").flat,
            norm(ex_sigma - solution_sigma, "h").flat,
        ]
    )
)
print(f"Error: {error}")

# %%
print(f"difference: {np.abs(residuum_norm - error)}")

# %% [markdown]
# ### Exact inf-sup value

# %% [markdown]
# Adjoin graph norm Gram matrix is $G_* = M + A M^{-1} A^*$

# %%
G2_ = M_ + A_ @ A_.T / grid.cell_volume

# %%
residuum2_rep = np.linalg.solve(G2_, residuum_vec)
residuum2_norm = np.sqrt(np.dot(residuum_vec, residuum2_rep))
print(f"Residuum v2 norm: {residuum2_norm}")

# %%
print(f"error / √loss = {error / residuum2_norm}")

# %%
R_ = A_.T @ np.linalg.solve(G2_, A_)

# %%
w, vr = scipy.linalg.eig(R_, M_)

# %%
print(f"largest imag: {np.max(np.abs(np.imag(w)))}")
w = np.real(w)
print(f"minimum: {np.min(w)}")
print(np.sort(w)[:10])
w = np.abs(w)

# %%
order = np.argsort(w)
imin = order[1]  # skip 0
imax = order[-1]

# %%
vals = np.sqrt(w)

# %%
print(f"inf-sup constant: {vals[imin]}")

# %%
print(f"continuity constant: {vals[imax]}")

# %%
plt.scatter(np.arange(order.size - 1), vals[order[1:]], marker=".")
plt.xlabel("eigenvalue number")
plt.ylabel("eigenvalue")
plt.ylim(bottom=0)
plt.savefig("spectrum.pdf", bbox_inches="tight")
plt.show()

# %%
gamma = vals[imin]
C = vals[imax]
print(f"{1 / C} < |error|/√loss < {1 / gamma}")

# %% [markdown]
# ### Eigenvalue analysis using sparse tools from `scipy`

# %%
spA = scipy.sparse.csr_array(A_)
spG2 = scipy.sparse.csc_array(G2_)
spG2_inv = scipy.sparse.linalg.factorized(spG2)
spM = scipy.sparse.csr_array(M_)


def applyR(v):
    return spA.T @ spG2_inv(spA @ v)


spR = scipy.sparse.linalg.LinearOperator(A_.shape, matvec=applyR)

# %% [markdown]
# Computing the largest eigenvalue, corresponding to the continuity constant takes too
# long. There are multiple eigenvalues clustered in the vicinity of 1, which is
# detrimental to power iteration performance.

# %%
# w, vr = scipy.sparse.linalg.eigsh(spR, 1, spM, which="LM")
# print(w)

# %%
w, vr = scipy.sparse.linalg.eigsh(spR, 3, spM, which="SM")
print(w)

# %% [markdown]
# ## Pytorch

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# %%
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 7),
        )

    def forward(self, x):
        values = self.linear_relu_stack(x)
        return values


pinn = PINN().to(device)
print(pinn)


# %%
def pinn_to_gridfuns(pinn):
    points = np.concat(grid.points).reshape(2, -1)
    args = torch.from_numpy(points.T.astype(np.float32)).to(device)
    pinn_values = torch.ravel(pinn(args).t()).cpu().detach().numpy()
    pinn_vec = remove_dofs(pinn_values, W_bc)

    first_U = S.dim - len(S_bc)
    first_P = first_U + U.dim - len(U_bc)

    pinn_sigma_vec = pinn_vec[:first_U]
    pinn_u_vec = pinn_vec[first_U:first_P]
    pinn_p_vec = pinn_vec[first_P:]

    pinn_p_vec -= np.mean(pinn_p_vec)

    pinn_sigma = vec_to_fun(pinn_sigma_vec, S_bc, (2, 2))
    pinn_u = vec_to_fun(pinn_u_vec, U_bc, (2,))
    pinn_p = vec_to_fun(pinn_p_vec, P_bc)

    return pinn_sigma, pinn_u, pinn_p


# %%
G2_LU = scipy.linalg.lu_factor(G2_)


class ResiduumNormSq(torch.autograd.Function):
    @staticmethod
    def forward(vals):
        vals_vec = remove_dofs(vals.cpu().numpy(), W_bc)
        residuum_vec = A_ @ vals_vec - M_ @ rhs_vec
        residuum_rep = scipy.linalg.lu_solve(G2_LU, residuum_vec)
        return torch.tensor(np.dot(residuum_vec, residuum_rep))

    @staticmethod
    def setup_context(ctx, args, output):
        (vals,) = args
        ctx.save_for_backward(vals)

    @staticmethod
    def backward(ctx, grad_output):
        (vals,) = ctx.saved_tensors
        vals_vec = remove_dofs(vals.cpu().numpy(), W_bc)
        residuum_vec = A_ @ vals_vec - M_ @ rhs_vec
        residuum_rep = scipy.linalg.lu_solve(G2_LU, residuum_vec)
        full_residuum_rep = reinsert_dofs(A_.T @ residuum_rep, W_bc)
        grad_vals = grad_output * 2 * torch.tensor(full_residuum_rep)
        return grad_vals.to(device)


def loss_function(pinn):
    points = np.concat(grid.points).reshape(2, -1)
    args = torch.from_numpy(points.T.astype(np.float32)).to(device)
    values = torch.ravel(pinn(args).t())
    loss = ResiduumNormSq.apply(values)
    return loss


# %%
class SolutionError(NamedTuple):
    sigma: np.ndarray
    u: np.ndarray
    p: float

    @property
    def sigma_total(self):
        return np.linalg.norm(self.sigma.flat)

    @property
    def u_total(self):
        return np.linalg.norm(self.u)

    @property
    def total(self):
        vec = np.array([self.p, self.sigma_total, self.u_total])
        return np.linalg.norm(vec)


class LearningEntry(NamedTuple):
    epoch: int
    loss: float
    error_exact: SolutionError
    error_discrete: SolutionError


def stokes_solution_error(solution, exact):
    sigma, u, p = solution
    tau, v, q = exact

    return SolutionError(
        sigma=norm((sigma - tau) * s_mask, "h"),
        u=norm((u - v) * u_mask, "h"),
        p=norm((p - q) * p_mask, "h"),
    )


# %%
def train(pinn, optimizer, max_epochs):
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = loss_function(pinn)
        yield epoch, loss.item()
        loss.backward()
        optimizer.step()


exact_funs = (exact_sigma, exact_u, exact_p)
discrete_funs = (solution_sigma, solution_u, solution_p)

log = []
optimizer = torch.optim.Adamax(pinn.parameters())

for epoch, loss in train(pinn, optimizer, 1000):
    pinn_funs = pinn_to_gridfuns(pinn)
    error_exact = stokes_solution_error(pinn_funs, exact_funs)
    error_discrete = stokes_solution_error(pinn_funs, discrete_funs)

    entry = LearningEntry(epoch, loss, error_exact, error_discrete)
    log.append(entry)

    ratio = error_discrete.total / np.sqrt(loss)
    ok = 1 / C < ratio < 1 / gamma
    if not ok:
        print(f"|error|/√loss ratio = {ratio}, out of bounds ({1 / C}, {1 / gamma})")

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:>5}  loss: {loss:.7g}, √loss: {np.sqrt(loss):.7g}, "
            f"error discrete: {error_discrete.total:.7g}, "
            f"error exact: {error_exact.total:.7g}",
            flush=True,
        )
        print(f"   ratio: {1 / C:.3f} < {ratio:.4f} < {1 / gamma:.3f} ? {ok}")

# %%
pinn_sigma, pinn_u, pinn_p = pinn_to_gridfuns(pinn)

# %%
plot_stokes(pinn_sigma, pinn_u, pinn_p, "RPINN solution", "rpinn.pdf")

# %%
diff_u = pinn_u - solution_u
diff_sigma = (pinn_sigma - solution_sigma) * s_mask
diff_p = (pinn_p - solution_p) * p_mask
title = "Difference between the PINN solution and the discrete formulation solution"
plot_stokes(diff_sigma, diff_u, diff_p, title, "rpinn-error.pdf")

# %%
until = 3000
epochs = [e.epoch for e in log][:until]
loss = np.array([e.loss for e in log])[:until]
error_discrete = np.array([e.error_discrete.total for e in log])[:until]
error_exact = np.array([e.error_exact.total for e in log])[:until]
lower_bound = 1 / C * np.sqrt(loss)
upper_bound = 1 / gamma * np.sqrt(loss)

plt.figure(figsize=(6.4, 4.8))
plt.plot(epochs, error_discrete, label=r"$\|u_\theta - u_\text{discrete}\|_h$")
plt.plot(epochs, upper_bound, "--", label=r"$\frac{1}{\gamma}\sqrt{\text{LOSS}}$")
plt.loglog(epochs, lower_bound, "--", label=r"$\frac{1}{M}\sqrt{\text{LOSS}}$")
plt.fill_between(epochs, lower_bound, upper_bound, alpha=0.1)
# plt.plot(epochs, error_exact, label=r"$\|u_\theta - u_\text{exact}\|_h$")
plt.xlabel("epoch")
plt.legend()
plt.savefig("errors-loglog.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Residuum norm gradient test

# %%
input = torch.randn(W.dim, dtype=torch.double, requires_grad=True)
test = torch.autograd.gradcheck(ResiduumNormSq.apply, input, eps=1e-6, atol=1e-4)
print(test)
