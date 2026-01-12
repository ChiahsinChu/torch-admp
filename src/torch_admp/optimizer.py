# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Callable

import torch


@torch.jit.unused
def line_search(
    func_value: Callable,
    func_grads: Callable,
    x0: torch.Tensor,
    eps: float = 1e-6,
    fk: torch.Tensor = None,
    gk: torch.Tensor = None,
    pk: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    history_x = torch.arange(3, dtype=torch.float64, device=x0.device)
    if fk is None:
        x0 = x0.detach()
        x0.requires_grad = True
        if x0.grad is not None:
            x0.grad.detach_()
            x0.grad.zero_()
        fk = func_value(x0, **kwargs)
        gk = func_grads(fk, x0)
        pk = -gk / torch.norm(gk)
    history_f = [fk]

    xk = x0.detach()
    # xk.requires_grad = True
    for _ in range(2):
        if torch.norm(gk) / xk.shape[0] < eps:
            return xk
        xk = xk + pk
        xk.detach_()
        xk.requires_grad = True
        if xk.grad is not None:
            xk.grad.detach_()
            xk.grad.zero_()
        fk = func_value(xk, **kwargs)
        gk = func_grads(fk, xk)
        fk.detach_()
        history_f.append(fk)

    coeff_matrix = torch.stack(
        [history_x**2, history_x, torch.ones_like(history_x)], dim=1
    )
    y = torch.stack(history_f)
    coeff = torch.linalg.solve(coeff_matrix, y)
    # print(coeff[0])
    x_opt = x0 - coeff[1] / (2 * coeff[0]) * pk
    return x_opt


@torch.jit.unused
def quadratic_optimize(
    func_value: Callable,
    func_grads: Callable,
    xk: torch.Tensor,
    eps: float = 1e-4,
    ls_eps: float = 1e-4,
    max_iter: int = 20,
    **kwargs,
):
    converge_iter: int = -1

    if xk.grad is not None:
        xk.grad.detach_()
        xk.grad.zero_()

    fk = func_value(xk, **kwargs)
    gk = func_grads(fk, xk)
    pk = -gk / torch.norm(gk)
    for ii in range(max_iter):
        fk.detach_()
        pk.detach_()
        # Selecting the step length
        x_new = line_search(func_value, func_grads, xk, ls_eps, fk, gk, pk, **kwargs)
        x_new.detach_()
        x_new.requires_grad = True
        if x_new.grad is not None:
            x_new.grad.detach_()
            x_new.grad.zero_()
        fk_new = func_value(x_new, **kwargs)
        gk_new = func_grads(fk_new, x_new)

        xk = x_new
        fk = fk_new

        norm_grad = torch.norm(gk_new) / xk.shape[0]
        # print(norm_grad)
        if norm_grad < eps:
            gk = gk_new
            converge_iter = ii
            break
        else:
            pk = update_pr(gk, pk, gk_new)
            gk = gk_new

    return xk, fk, gk, converge_iter


# class QuadraticProjectedGradientOptimizer(Optimizer):
#     def __init__(
#         self,
#         params,
#         lr: float = 1e-2,
#         tol: float = 1e-4,
#         max_iter: int = 20,
#         constraint_matrix: torch.Tensor = None,
#     ):
#         defaults = dict(lr=lr, tol=tol, max_iter=max_iter)
#         Optimizer.__init__(self, params, defaults)
#         self.constraint_matrix = constraint_matrix

#     def value_func(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.empty_like(x)

#     def grad_func(self, x: torch.Tensor) -> torch.Tensor:
#         pass

#     def line_search(
#         self,
#         _x0: torch.Tensor,
#         positions: torch.Tensor,
#         box: Optional[torch.Tensor] = None,
#         fk: torch.Tensor = None,
#         gk: torch.Tensor = None,
#         pk: torch.Tensor = None,
#     ) -> torch.Tensor:
#         x0 = _x0.detach()
#         x0.requires_grad = True

#         history_x = torch.arange(3, dtype=torch.float64, device=positions.device)
#         if fk is None:
#             fk = self.value_func(x0, positions, box)
#             gk = self.func_grads(fk, x0)
#             pk = -gk / torch.norm(gk)
#         history_f = [fk]
#         xk = x0.detach()
#         xk.requires_grad = True
#         for _ in range(2):
#             if torch.norm(gk) / x0.shape[0] < self.ls_eps:
#                 return xk
#             xk = xk + pk
#             fk = self.func_energy(xk, positions, box)
#             gk = self.func_grads(fk, xk)
#             history_f.append(fk)

#         coeff_matrix = torch.stack(
#             [history_x**2, history_x, torch.ones_like(history_x)], dim=1
#         )
#         y = torch.stack(history_f)
#         coeff = torch.linalg.solve(coeff_matrix, y)
#         print(coeff)
#         x_opt = x0 - coeff[1] / (2 * coeff[0]) * pk
#         return x_opt


# class ConjugateGradientQuadraticOptimizer(Optimizer):
#     def __init__(self, params, lr=1e-2, tol=1e-6, max_iter=20):
#         defaults = dict(lr=lr, tol=tol, max_iter=max_iter)
#         Optimizer.__init__(self, params, defaults)

#     # The step method for the optimizer, fully compatible with torch.jit.script
#     def step(self, grads: torch.Tensor, params: list):
#         for group in self.param_groups:
#             lr = group["lr"]
#             tol = group["tol"]
#             max_iter = group["max_iter"]

#             # Initialization for conjugate gradient
#             r = -grads
#             p = r.clone()
#             r_norm = r.dot(r)

#             # Iterative conjugate gradient solver
#             for i in range(max_iter):
#                 # Compute the product of A and p
#                 Ap = torch.autograd.grad(p.dot(grads), params, retain_graph=True)
#                 Ap_flat = torch.cat([a.view(-1) for a in Ap])

#                 # Compute alpha
#                 alpha = r_norm / p.dot(Ap_flat)

#                 # Update parameters using alpha and search direction p
#                 for param, direction in zip(
#                     params, p.split([param.numel() for param in params])
#                 ):
#                     param.data.add_(lr * alpha * direction.view_as(param))

#                 r_new = r + alpha * Ap_flat
#                 r_new_norm = r_new.dot(r_new)

#                 if r_new_norm < tol:
#                     break

#                 beta = r_new_norm / r_norm
#                 p = r_new + beta * p
#                 r = r_new
#                 r_norm = r_new_norm

#     def zero_grad(self):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is not None:
#                     p.grad.detach_()
#                     p.grad.zero_()


# # Define a simple model with the optimizer
# class SimpleModelWithOptimizer(nn.Module):
#     def __init__(self):
#         super(SimpleModelWithOptimizer, self).__init__()
#         self.fc = torch.nn.Linear(10, 1)
#         self.optimizer = JITConjugateGradient(self.fc.parameters(), lr=1e-2)

#     def forward(self, x, target):
#         output = self.fc(x)
#         loss = torch.nn.functional.mse_loss(output, target)
#         return loss

#     def optimize(self, x, target):
#         loss = self.forward(x, target)
#         self.optimizer.zero_grad()
#         grads = torch.autograd.grad(loss, self.fc.parameters(), create_graph=True)
#         grads_flat = torch.cat([g.view(-1) for g in grads])
#         self.optimizer.step(grads_flat, list(self.fc.parameters()))
#         return loss


# # Script the model
# model = SimpleModelWithOptimizer()

# # Dummy input and target for demonstration
# x = torch.randn(100, 10)
# target = torch.randn(100, 1)

# # Optimize the model
# for i in range(100):
#     loss = model.optimize(x, target)
#     print(f"Iteration {i}, Loss: {loss.item()}")

# # Now we can script the entire model
# scripted_model = torch.jit.script(model)

# # Saving and loading the scripted model
# scripted_model.save("model_with_optimizer.pt")
# loaded_model = torch.jit.load("model_with_optimizer.pt")

# # You can now use the loaded model in the same way


def update_sd(
    gk: torch.Tensor,
    pk: torch.Tensor,
    gk_new: torch.Tensor,
) -> torch.Tensor:
    """
    Steepest Descent Algorithm
    """
    gk = gk_new
    # Selection of the direction of the steepest descent
    pk = -gk / torch.linalg.norm(gk)
    return pk


def update_fr(
    gk: torch.Tensor,
    pk: torch.Tensor,
    gk_new: torch.Tensor,
) -> torch.Tensor:
    """
    Feltcher-Reeves Algorithm
    """
    old_gk = gk
    gk = gk_new
    # Line (16) of the Fletcher-Reeves algorithm
    chi = torch.linalg.norm(gk) ** 2 / torch.linalg.norm(old_gk) ** 2
    # Updated descent direction
    pk = -gk + chi * pk
    return pk


def update_pr(
    gk: torch.Tensor,
    pk: torch.Tensor,
    gk_new: torch.Tensor,
) -> torch.Tensor:
    """
    Polak-Ribiere Algorithm
    """
    old_gk = gk
    gk = gk_new
    # Line (16) of the Polak-Ribiere Algorithm
    chi = (gk - old_gk).dot(gk) / torch.linalg.norm(old_gk) ** 2
    chi = torch.where(chi > 0, chi, torch.zeros_like(chi))
    # Updated descent direction
    pk = -gk + chi * pk
    return pk


def update_hs(
    gk: torch.Tensor,
    pk: torch.Tensor,
    gk_new: torch.Tensor,
) -> torch.Tensor:
    """ "
    Hestenes-Stiefel Algorithm
    """
    old_gk = gk
    gk = gk_new
    chi = gk.dot(gk - old_gk) / pk.dot(gk - old_gk)
    # Updated descent direction
    pk = -gk + chi * pk
    return pk


def update_dy(
    gk: torch.Tensor,
    pk: torch.Tensor,
    gk_new: torch.Tensor,
) -> torch.Tensor:
    """
    Dai-Yuan Algorithm
    """
    old_gk = gk
    gk = gk_new
    chi = torch.linalg.norm(gk) ** 2 / pk.dot(gk - old_gk)
    # Updated descent direction
    pk = -gk + chi * pk
    return pk


def update_hz(
    gk: torch.Tensor,
    pk: torch.Tensor,
    gk_new: torch.Tensor,
) -> torch.Tensor:
    """
    Hager-Zhang Algorithm
    """
    old_gk = gk
    gk = gk_new
    delta_gk = gk - old_gk
    m = delta_gk - 2 * pk * torch.linalg.norm(delta_gk) ** 2 / pk.dot(delta_gk)
    n = gk / pk.dot(delta_gk)
    chi = m.dot(n)
    # Updated descent direction
    pk = -gk + chi * pk
    return pk
