# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Poisson on a fancy domain

# %%
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy
import torch
from matplotlib.image import imread
from torch import nn

from dvf import (
    FunctionSpace,
    Grid,
    GridFunction,
    assemble,
    grad,
    lift_to_gridfun,
    norm,
    reinsert_dofs,
    remove_dofs,
    select_dofs,
)

# %%
grid = Grid(50, 50)

# %%
U = FunctionSpace(grid)

# %%


domain_img = imread("domain.png") < 0.3
plt.imshow(domain_img)

# %%
u_mask = GridFunction.from_array(domain_img, grid)

# %%
U_bc = select_dofs(U, u_mask, invert=True)

# %%
u_mask_array = u_mask.tabulate()

extra_col = np.zeros((grid.n[0] + 1, 1))
extra_row = np.zeros((1, grid.n[1] + 1))

u_mask_left = np.c_[extra_col, u_mask_array[:, :-1]]
u_mask_right = np.c_[u_mask_array[:, 1:], extra_col]
u_mask_top = np.r_[extra_row, u_mask_array[:-1, :]]
u_mask_bottom = np.r_[u_mask_array[1:, :], extra_row]

u_mask_max = np.max([u_mask_left, u_mask_right, u_mask_top, u_mask_bottom], axis=0)
true_boundary_array = u_mask_max - u_mask_array
domain_array = np.max([u_mask_array, true_boundary_array], axis=0)
domain_mask = GridFunction.from_array(domain_array, grid)

true_boundary_mask = GridFunction.from_array(true_boundary_array, grid)
U_true_boundary_dofs = select_dofs(U, true_boundary_mask)
U_dofs_outside_domain = select_dofs(U, domain_mask, invert=True)

# %%
plt.imshow(true_boundary_array)
# plt.imshow(domain_array)

# %%
dot = lift_to_gridfun(np.vdot)


# %%
def B_form(u, v):
    return dot(grad(u, "+"), grad(v, "+"))


def G_form(u, v):
    return dot(grad(u, "+"), grad(v, "+"))


# %%
u = U.trial_function()
v = U.test_function()

G = np.zeros((U.dim, U.dim))
assemble(G_form(u, v), G, u, v)

B = np.zeros((U.dim, U.dim))
assemble(B_form(u, v), B, u, v)

# %%
M = np.identity(U.dim) * grid.cell_volume


# %%
def vector_of_values(*funs):
    return np.concat([np.ravel(f.tabulate()) for f in funs])


# %%
G_ = remove_dofs(G, U_bc)
B_ = remove_dofs(B, U_bc)
M_ = remove_dofs(M, U_bc)

# %%
x, y = sympy.symbols("x y")
z0 = 0.5 + sympy.I * 0.5
z = x + sympy.I * y - z0
u_exact = sympy.re(sympy.cos(3 * z**5 - z) - z**2)

u_exact = GridFunction.from_function(sympy.lambdify([x, y], u_exact), grid)


# %%
def rhs(x, y):
    p = np.array([x, y])
    c = np.array([0, 0.1])
    r = np.linalg.norm(p - c) / 0.05
    return 0 * np.exp(-(r**2))


rhs_f = GridFunction.from_function(rhs, grid)

# %%
shift_arr = np.zeros(grid.n + 1)
k = 10

u_exact_array = vector_of_values(u_exact)

shift_arr.flat[U_true_boundary_dofs] = u_exact_array[U_true_boundary_dofs]

shift_fun = GridFunction.from_array(shift_arr, grid)
shift_vec = vector_of_values(shift_fun)

B_shift = B @ shift_vec
B_shift_ = remove_dofs(B_shift, U_bc)

# %%
plt.imshow(shift_arr)
plt.colorbar()
plt.show()


# %%
def vec_to_fun(vec, bc, shape=()):
    data = reinsert_dofs(vec, bc, value=0) + shift_vec
    data[U_dofs_outside_domain] = np.nan
    return GridFunction.from_array(data.reshape(*shape, *grid.shape), grid)


# %%
rhs_vec = grid.cell_volume * remove_dofs(vector_of_values(rhs_f), U_bc) - B_shift_

# %%
lu, piv = scipy.linalg.lu_factor(B_)
solution_data = scipy.linalg.lu_solve((lu, piv), rhs_vec)

# %%
solution_u = vec_to_fun(solution_data, U_bc)

# %%
aspect = grid.h[1] / grid.h[0]
plt.imshow(np.flipud(solution_u.tabulate().T), aspect=aspect)
plt.colorbar()
plt.show()

# %% [markdown]
# ## RVPINN

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
            nn.Linear(512, 1),
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
    pinn_vec = remove_dofs(pinn_values, U_bc)
    return vec_to_fun(pinn_vec, U_bc)


# %%
G_LU = scipy.linalg.lu_factor(B_)


class ResiduumNormSq(torch.autograd.Function):
    @staticmethod
    def forward(vals):
        vals_vec = remove_dofs(vals.cpu().numpy(), U_bc)
        residuum_vec = B_ @ vals_vec - rhs_vec
        residuum_rep = scipy.linalg.lu_solve(G_LU, residuum_vec)
        return torch.tensor(np.dot(residuum_vec, residuum_rep))

    @staticmethod
    def setup_context(ctx, args, output):
        (vals,) = args
        ctx.save_for_backward(vals)

    @staticmethod
    def backward(ctx, grad_output):
        (vals,) = ctx.saved_tensors
        vals_vec = remove_dofs(vals.cpu().numpy(), U_bc)
        residuum_vec = B_ @ vals_vec - rhs_vec
        residuum_rep = scipy.linalg.lu_solve(G_LU, residuum_vec)
        full_residuum_rep = reinsert_dofs(B_.T @ residuum_rep, U_bc)
        grad_vals = grad_output * 2 * torch.tensor(full_residuum_rep)
        return grad_vals.to(device)


def loss_function(pinn):
    points = np.concat(grid.points).reshape(2, -1)
    args = torch.from_numpy(points.T.astype(np.float32)).to(device)
    values = torch.ravel(pinn(args).t())
    loss = ResiduumNormSq.apply(values)
    return loss


# %%
class LearningEntry(NamedTuple):
    epoch: int
    loss: float
    error_exact: float
    error_discrete: float


def laplace_solution_error(solution, exact):
    denan = lift_to_gridfun(np.nan_to_num)
    return norm(denan((solution - exact) * u_mask), "grad_h")


# %%
def train(pinn, optimizer, max_epochs):
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = loss_function(pinn)
        yield epoch, loss.item()
        loss.backward()
        optimizer.step()


log = []
optimizer = torch.optim.Adamax(pinn.parameters())

for epoch, loss in train(pinn, optimizer, 1000):
    pinn_u = pinn_to_gridfuns(pinn)
    error_exact = laplace_solution_error(pinn_u, u_exact)
    error_discrete = laplace_solution_error(pinn_u, solution_u)

    entry = LearningEntry(epoch, loss, error_exact, error_discrete)
    log.append(entry)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:>5}  loss: {loss:.7g}, √loss: {np.sqrt(loss):.7g}, "
            f"error discrete: {error_discrete:.7g}, "
            f"error exact: {error_exact:.7g}",
            flush=True,
        )
        # print(f"   ratio: {1/C:.3f} < {ratio:.4f} < {1/gamma:.3f} ? {ok}")

# %%
pinn_u = pinn_to_gridfuns(pinn)

# %%
aspect = grid.h[1] / grid.h[0]
plt.imshow(np.flipud(pinn_u.tabulate().T), aspect=aspect)
plt.colorbar()
plt.show()

# %%
until = 1000
epochs = [e.epoch for e in log][:until]
loss = np.array([e.loss for e in log])[:until]
error_discrete = np.array([e.error_discrete for e in log])[:until]
error_exact = np.array([e.error_exact for e in log])[:until]
lower_bound = 1 * np.sqrt(loss)
upper_bound = 1 * np.sqrt(loss)

plt.figure(figsize=(6.4, 4.8))
plt.plot(epochs, error_discrete, label=r"$\|u_\theta - u_\text{discrete}\|_h$")
plt.plot(epochs, upper_bound, "--", label=r"$\frac{1}{\gamma}\sqrt{\text{LOSS}}$")
plt.loglog(epochs, lower_bound, "--", label=r"$\frac{1}{M}\sqrt{\text{LOSS}}$")
plt.fill_between(epochs, lower_bound, upper_bound, alpha=0.1)
plt.plot(epochs, error_exact, label=r"$\|u_\theta - u_\text{exact}\|_h$")
plt.xlabel("epoch")
plt.legend()
# plt.savefig("errors-loglog.pdf", bbox_inches="tight")
plt.show()

# %%
