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
# # Second-order Laplace equation formulation

# %%
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy
import torch
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
grid = Grid(30, 30)

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
def B_form(u, v):
    return dot(grad(u, "+"), grad(v, "+"))


# %%
u = U.trial_function()
v = U.test_function()

B = np.zeros((U.dim, U.dim))
assemble(B_form(u, v), B, u, v)

# %%
M = np.identity(U.dim) * grid.cell_volume


# %%
def vector_of_values(*funs):
    return np.concat([np.ravel(f.tabulate()) for f in funs])


# %%
B_ = remove_dofs(B, U_bc)
M_ = remove_dofs(M, U_bc)

# %%
x, y = sympy.symbols("x y")
u_exact = sympy.sin(sympy.pi * x) * sympy.sin(3 * sympy.pi * y)


def laplacian(expr, *vars):
    return sum(sympy.diff(expr, v, v) for v in vars)


rhs = -laplacian(u_exact, x, y)

# %%
u_exact = GridFunction.from_function(sympy.lambdify([x, y], u_exact), grid)
rhs_f = GridFunction.from_function(sympy.lambdify([x, y], rhs), grid)


# %%
def vec_to_fun(vec, bc, shape=()):
    data = reinsert_dofs(vec, bc)
    return GridFunction.from_array(data.reshape(*shape, *grid.shape), grid)


# %%
rhs_vec = remove_dofs(vector_of_values(rhs_f), U_bc)

# %%
lu, piv = scipy.linalg.lu_factor(B_)
solution_data = scipy.linalg.lu_solve((lu, piv), grid.cell_volume * rhs_vec)

# %%
solution_u = vec_to_fun(solution_data, U_bc)

# %%
aspect = grid.h[1] / grid.h[0]
plt.imshow(solution_u.tabulate().T, aspect=aspect)
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
B_LU = scipy.linalg.lu_factor(B_)


class ResiduumNormSq(torch.autograd.Function):
    @staticmethod
    def forward(vals):
        vals_vec = remove_dofs(vals.cpu().numpy(), U_bc)
        residuum_vec = B_ @ vals_vec - M_ @ rhs_vec
        residuum_rep = scipy.linalg.lu_solve(B_LU, residuum_vec)
        return torch.tensor(np.dot(residuum_vec, residuum_rep))

    @staticmethod
    def setup_context(ctx, args, output):
        (vals,) = args
        ctx.save_for_backward(vals)

    @staticmethod
    def backward(ctx, grad_output):
        (vals,) = ctx.saved_tensors
        vals_vec = remove_dofs(vals.cpu().numpy(), U_bc)
        residuum_vec = B_ @ vals_vec - M_ @ rhs_vec
        residuum_rep = scipy.linalg.lu_solve(B_LU, residuum_vec)
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
    return norm((solution - exact) * u_mask, "grad_h")


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
plt.imshow(pinn_u.tabulate().T, aspect=aspect)
plt.show()

# %%
until = 1000
epochs = [e.epoch for e in log][:until]
loss = np.array([e.loss for e in log])[:until]
error_discrete = np.array([e.error_discrete for e in log])[:until]
error_exact = np.array([e.error_exact for e in log])[:until]
lower_bound = 1 / 1 * np.sqrt(loss)
upper_bound = 1 / 1 * np.sqrt(loss)

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
