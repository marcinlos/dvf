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
# # Discrete Stokes formulation

# %% jupyter={"source_hidden": true}
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    norm,
)

# %%
grid = Grid(50)

# %% [markdown]
# ### Spaces and boundary conditions

# %%
S = TensorFunctionSpace(grid, (2, 2))
U = VectorFunctionSpace(grid, 2)
P = FunctionSpace(grid)
W = CompositeFunctionSpace(S, U, P)

# %%
U_offset = S.dim
P_offset = S.dim + U.dim

S_bc = np.array([], dtype=np.intp)
U_bc = np.array([f.index for idx in grid.boundary() for f in U.basis_at(idx)])
P_bc = np.array(
    [f.index for idx in grid.boundary(Edge.LEFT | Edge.TOP) for f in P.basis_at(idx)]
    + [grid.ravel_index((grid.n, grid.n))]
)
W_bc = np.concat([S_bc, U_bc + U_offset, P_bc + P_offset])

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


# %% [markdown]
# ### Verification of the adjoint
#
# We use random functions to avoid assembling the matrix of $A^*$, which is expensive
# for large grids.


# %%
def random_functions():
    sigma_data = np.random.rand(2, 2, *grid.shape)
    sigma_data.flat[S_bc] = 0
    sigma = GridFunction.from_array(sigma_data, grid)

    u_data = np.random.rand(2, *grid.shape)
    u_data.flat[U_bc] = 0
    u = GridFunction.from_array(u_data, grid)

    p_data = np.random.rand(*grid.shape)

    mask = np.ones(grid.shape)
    mask.flat[P_bc] = 0
    p_data -= np.sum(p_data * mask) / np.sum(mask)
    p_data.flat[P_bc] = 0
    p = GridFunction.from_array(p_data, grid)

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
M = grid.h**2 * np.identity(W.dim)


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
def apply_bc(matrix, dofs):
    return np.delete(np.delete(matrix, dofs, axis=0), dofs, axis=1)


A_ = apply_bc(A, W_bc)
AT_ = apply_bc(AT, W_bc)
AT_p_ = apply_bc(AT_p, W_bc)
M_ = apply_bc(M, W_bc)
AT_graph_ = apply_bc(AT_graph, W_bc)


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

exact_u_data = np.stack([stokes.u1(X, Y), stokes.u2(X, Y)])
exact_u = GridFunction.from_array(exact_u_data, grid)

exact_sigma11_data = stokes.u1_dx(X, Y)
exact_sigma12_data = stokes.u1_dy(X, Y)
exact_sigma21_data = stokes.u2_dx(X, Y)
exact_sigma22_data = stokes.u2_dy(X, Y)
exact_sigma_data = np.array(
    [
        [exact_sigma11_data, exact_sigma12_data],
        [exact_sigma21_data, exact_sigma22_data],
    ]
)
exact_sigma = GridFunction.from_array(exact_sigma_data, grid)

# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
fig.suptitle("Exact solution plotted on the grid")

img0 = axs[0].imshow(exact_u.tabulate()[0, ...].T)
axs[0].set_title(r"$u_1$")
fig.colorbar(img0, ax=axs[0], shrink=0.7)

img1 = axs[1].imshow(exact_u.tabulate()[1, ...].T)
axs[1].set_title(r"$u_2$")
fig.colorbar(img1, ax=axs[1], shrink=0.7)

img2 = axs[2].imshow(exact_p.tabulate().T)
axs[2].set_title(r"$p$")
fig.colorbar(img2, ax=axs[2], shrink=0.7)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(ncols=4, figsize=(15, 5))

img00 = axs[0].imshow(exact_sigma.tabulate()[0, 0, ...].T)
axs[0].set_title(r"$\sigma_{11} = \partial_x u_1$")
fig.colorbar(img00, ax=axs[0], location="bottom", shrink=0.9)

img01 = axs[1].imshow(exact_sigma.tabulate()[0, 1, ...].T)
axs[1].set_title(r"$\sigma_{12} = \partial_y u_1$")
fig.colorbar(img01, ax=axs[1], location="bottom", shrink=0.9)

img10 = axs[2].imshow(exact_sigma.tabulate()[1, 0, ...].T)
axs[2].set_title(r"$\sigma_{21} = \partial_x u_2$")
fig.colorbar(img10, ax=axs[2], location="bottom", shrink=0.9)

img11 = axs[3].imshow(exact_sigma.tabulate()[1, 1, ...].T)
axs[3].set_title(r"$\sigma_{22} = \partial_y u_y$")
fig.colorbar(img11, ax=axs[3], location="bottom", shrink=0.9)


plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(15, 10))
sigma = nabla(exact_u, "-")

img00 = axs[0, 0].imshow(sigma.tabulate()[0, 0, ...].T)
axs[0, 0].set_title(r"$\nabla_{x-}u_1$")
fig.colorbar(img00, ax=axs[0, 0], location="bottom", shrink=0.9)

img01 = axs[0, 1].imshow(sigma.tabulate()[0, 1, ...].T)
axs[0, 1].set_title(r"$\nabla_{y-}u_1$")
fig.colorbar(img01, ax=axs[0, 1], location="bottom", shrink=0.9)

img10 = axs[0, 2].imshow(sigma.tabulate()[1, 0, ...].T)
axs[0, 2].set_title(r"$\nabla_{x-}u_2$")
fig.colorbar(img10, ax=axs[0, 2], location="bottom", shrink=0.9)

img11 = axs[0, 3].imshow(sigma.tabulate()[1, 1, ...].T)
axs[0, 3].set_title(r"$\nabla_{y-}u_2$")
fig.colorbar(img11, ax=axs[0, 3], location="bottom", shrink=0.9)

difference = exact_sigma - nabla(exact_u, "-")

img00 = axs[1, 0].imshow(difference.tabulate()[0, 0, ...].T)
axs[1, 0].set_title(r"$\sigma_{11} - \nabla_{x-}u_1$")
fig.colorbar(img00, ax=axs[1, 0], location="bottom", shrink=0.9)

img01 = axs[1, 1].imshow(difference.tabulate()[0, 1, ...].T)
axs[1, 1].set_title(r"$\sigma_{12} - \nabla_{y-}u_1$")
fig.colorbar(img01, ax=axs[1, 1], location="bottom", shrink=0.9)

img10 = axs[1, 2].imshow(difference.tabulate()[1, 0, ...].T)
axs[1, 2].set_title(r"$\sigma_{21} - \nabla_{x-}u_2$")
fig.colorbar(img10, ax=axs[1, 2], location="bottom", shrink=0.9)

img11 = axs[1, 3].imshow(difference.tabulate()[1, 1, ...].T)
axs[1, 3].set_title(r"$\sigma_{22} - \nabla_{x-}u_2$")
fig.colorbar(img11, ax=axs[1, 3], location="bottom", shrink=0.9)

fig.suptitle(r"Components of $\nabla_{-} u$ and how it differs from exact $\sigma$")

plt.tight_layout()
plt.show()

# %% [markdown]
# Relative differences between components of exact $\sigma$ and $\nabla_{-} u$ are quite
# large for the off-diagonal components.

# %%
norm(exact_sigma - nabla(exact_u, "-"), "h") / norm(exact_sigma, "h")


# %% [markdown]
# This makes sense, since $\nabla_{-} u$ is not defined on the left ($\nabla_{x-}$) or
# top ($\nabla_{y-}$) edges of the grid. Removing these parts of the boundary from the
# error computation significantly reduces the error.


# %%
def mask_fun(i, j):
    mdx = 0 if i == 0 else 1
    mdy = 0 if j == 0 else 1
    return np.array([[mdx, mdy], [mdx, mdy]])


mask = GridFunction(mask_fun, grid)

norm((exact_sigma - nabla(exact_u, "-")) * mask, "h") / norm(exact_sigma * mask, "h")

# %% [markdown]
# ## Solving discrete formulation

# %% [markdown]
# We start by preparing the right-hand side.

# %%
rhs_tau_vec = np.zeros(S.dim)
rhs_v_vec = np.delete(np.ravel(rhs_f.tabulate()), U_bc)
rhs_q_vec = np.delete(np.zeros(P.dim), P_bc)

rhs_vec = np.concat([rhs_tau_vec, rhs_v_vec, rhs_q_vec])

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
first_U = S.dim
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
def vec_to_fun(vec, bc, shape):
    # kind of a hack - `np.insert` inserts items at the indices
    # of the vector as it is, not as it was before removing them,
    # so we need to modify them accordingly
    data = np.insert(vec, np.sort(bc) - np.arange(len(bc)), 0)
    return GridFunction.from_array(data.reshape(shape), grid)


solution_sigma = vec_to_fun(solution_sigma_vec, S_bc, (2, 2, *grid.shape))
solution_u = vec_to_fun(solution_u_vec, U_bc, (2, *grid.shape))
solution_p = vec_to_fun(solution_p_vec, P_bc, grid.shape)

# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

img0 = axs[0].imshow(solution_u.tabulate()[0, ...].T)
axs[0].set_title(r"$u_1$")
fig.colorbar(img0, ax=axs[0], shrink=0.7)

img1 = axs[1].imshow(solution_u.tabulate()[1, ...].T)
axs[1].set_title(r"$u_2$")
fig.colorbar(img1, ax=axs[1], shrink=0.7)

img2 = axs[2].imshow(solution_p.tabulate().T)
axs[2].set_title(r"$p$")
fig.colorbar(img2, ax=axs[2], shrink=0.7)

fig.suptitle("Discrete formulation solution")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(ncols=4, figsize=(15, 5))

img00 = axs[0].imshow(solution_sigma.tabulate()[0, 0, ...].T)
axs[0].set_title(r"$\sigma_{11}$")
fig.colorbar(img00, ax=axs[0], location="bottom", shrink=0.9)

img01 = axs[1].imshow(solution_sigma.tabulate()[0, 1, ...].T)
axs[1].set_title(r"$\sigma_{12}$")
fig.colorbar(img01, ax=axs[1], location="bottom", shrink=0.9)

img10 = axs[2].imshow(solution_sigma.tabulate()[1, 0, ...].T)
axs[2].set_title(r"$\sigma_{21}$")
fig.colorbar(img10, ax=axs[2], location="bottom", shrink=0.9)

img11 = axs[3].imshow(solution_sigma.tabulate()[1, 1, ...].T)
axs[3].set_title(r"$\sigma_{22}$")
fig.colorbar(img11, ax=axs[3], location="bottom", shrink=0.9)

plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
diff_u = solution_u - exact_u
diff_sigma = solution_sigma - exact_sigma
diff_p = solution_p - exact_p

img0 = axs[0].imshow(diff_u.tabulate()[0, ...].T)
axs[0].set_title(r"$u_1$")
fig.colorbar(img0, ax=axs[0], shrink=0.7)

img1 = axs[1].imshow(diff_u.tabulate()[1, ...].T)
axs[1].set_title(r"$u_2$")
fig.colorbar(img1, ax=axs[1], shrink=0.7)

img2 = axs[2].imshow(diff_p.tabulate().T)
axs[2].set_title(r"$p$")
fig.colorbar(img2, ax=axs[2], shrink=0.7)

fig.suptitle(
    "Difference between the exact solution and the discrete formulation solution"
)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(ncols=4, figsize=(15, 5))

img00 = axs[0].imshow(diff_sigma.tabulate()[0, 0, ...].T)
axs[0].set_title(r"$\sigma_{11}$")
fig.colorbar(img00, ax=axs[0], location="bottom", shrink=0.9)

img01 = axs[1].imshow(diff_sigma.tabulate()[0, 1, ...].T)
axs[1].set_title(r"$\sigma_{12}$")
fig.colorbar(img01, ax=axs[1], location="bottom", shrink=0.9)

img10 = axs[2].imshow(diff_sigma.tabulate()[1, 0, ...].T)
axs[2].set_title(r"$\sigma_{21}$")
fig.colorbar(img10, ax=axs[2], location="bottom", shrink=0.9)

img11 = axs[3].imshow(diff_sigma.tabulate()[1, 1, ...].T)
axs[3].set_title(r"$\sigma_{22}$")
fig.colorbar(img11, ax=axs[3], location="bottom", shrink=0.9)


plt.tight_layout()
plt.show()

# %%
norm(diff_sigma, "h") / norm(exact_sigma, "h")

# %% [markdown]
# Removing contributions of the edges that should be removed:

# %%
norm(diff_sigma * mask, "h") / norm(exact_sigma * mask, "h")

# %%
norm(diff_u, "h") / norm(exact_u, "h")

# %%
norm(diff_p, "h") / norm(exact_p, "h")


# %% [markdown]
# Removing the DoFs that are not part of discrete pressure domain:


# %%
def p_mask_fun(i, j):
    return 0 if i == 0 or j == 0 or (i, j) == (grid.n, grid.n) else 1


p_mask = GridFunction(p_mask_fun, grid)

# %%
norm(diff_p * p_mask, "h") / norm(exact_p * p_mask, "h")

# %% [markdown]
# Ensuring zero mean of the exact pressure:

# %%
fixed_exact_p = (exact_p - integrate(exact_p * p_mask) / integrate(p_mask)) * p_mask
norm((fixed_exact_p - solution_p) * p_mask, "h") / norm(fixed_exact_p * p_mask, "h")

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
G_ = A_ @ A_.T / grid.h**2

# %%
ex_sigma, ex_u, ex_p = random_functions()

# %%
on_v = pi0(-div(ex_sigma, "+") + nabla(ex_p, "+") - rhs_f)
on_tau = ex_sigma - nabla(ex_u, "-")
on_q = div(ex_u, "-")

ex_rhs_vec = grid.h**2 * np.concat(
    [
        np.ravel(on_tau.tabulate()),
        np.ravel(on_v.tabulate()),
        np.ravel(on_q.tabulate()),
    ]
)
residuum_vec = np.delete(ex_rhs_vec, W_bc)
residuum_rep = np.linalg.solve(G_, residuum_vec)
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
# Adjoin graph norm Gram matrix is $G_* = M + A M^{-1} A^*$

# %%
G2_ = M_ + A_ @ A_.T / grid.h**2
