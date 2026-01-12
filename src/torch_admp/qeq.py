# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchopt
from torchopt.diff.implicit import custom_root

from torch_admp.base_force import BaseForceModule
from torch_admp.optimizer import update_pr
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import (
    calc_pgrads,
    safe_inverse,
    vector_projection,
    vector_projection_coeff_matrix,
)

# try:
#     import ncg_optimizer
# except ImportError:
#     Warning("ncg_optimizer is not installed. CG optimization is not available.")


class GaussianDampingForceModule(BaseForceModule):
    def __init__(
        self,
        units_dict: Optional[Dict] = None,
    ) -> None:
        BaseForceModule.__init__(self, units_dict)

    def forward(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Gaussian short-range damping energy model

        Parameters
        ----------
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        params : Dict[str, torch.Tensor]
            {
                "charge": t_charges, # atomic charges
                "eta": t_eta, # Gaussian width in length unit
            }

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        return self.forward_lower(
            charges=params["charge"],
            eta=params["eta"],
            pairs=pairs,
            ds=ds,
            buffer_scales=buffer_scales,
        )

    def forward_lower(
        self,
        charges: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        eta = eta * self.const_lib.length_coeff
        ds = ds * self.const_lib.length_coeff

        q_i = charges[pairs[:, 0]].reshape(-1)
        q_j = charges[pairs[:, 1]].reshape(-1)

        eta_ij = torch.sqrt((eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2) * 2)
        # avoid nan when calculating grad if eta_ij is zero
        eta_ij = torch.where(eta_ij == 0, 1e-10, eta_ij)
        pre_pair = -torch.erfc(ds / eta_ij)
        e_sr_pair = torch.sum(
            pre_pair * q_i * q_j * safe_inverse(ds, threshold=1e-4) * buffer_scales
        )

        pre_self = safe_inverse(eta, threshold=1e-4) / (2 * self.const_lib.sqrt_pi)
        e_sr_self = torch.sum(pre_self * charges * charges)

        e_sr = (e_sr_pair + e_sr_self) * self.const_lib.dielectric
        # eV to user-defined energy unit
        return e_sr / self.const_lib.energy_coeff


class SiteForceModule(BaseForceModule):
    def __init__(
        self,
        units_dict: Optional[Dict] = None,
    ) -> None:
        BaseForceModule.__init__(self, units_dict)

    def forward(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Chemical site energy model

        Parameters
        ----------
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        params : Dict[str, torch.Tensor]
            {
                "charge": t_charges, # atomic charges
                "chi": t_chi, # eletronegativity in energy / charge unit
                "hardness": t_hardness, # atomic hardness in energy / charge^2 unit
            }

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        return self.forward_lower(params["chi"], params["hardness"], params["charge"])

    def forward_lower(
        self,
        chi: torch.Tensor,
        hardness: torch.Tensor,
        charges: torch.Tensor,
    ):
        chi = chi * self.const_lib.energy_coeff
        hardness = hardness * self.const_lib.energy_coeff
        e = chi * charges + hardness * charges**2
        return torch.sum(e) / self.const_lib.energy_coeff


class QEqForceModule(BaseForceModule):
    """Charge equilibrium (QEq) model

    Parameters
    ----------
    rcut : float
        cutoff radius for short-range interactions
    ethresh : float, optional
        energy threshold for electrostatic interaction, by default 1e-5
    kspace: bool
        whether the reciprocal part is included
    rspace: bool
        whether the real space part is included
    slab_corr: bool
        whether the slab correction is applied
    slab_axis: int
        axis at which the slab correction is applied
    max_iter: int, optional
        maximum number of iterations for optimization, by default 20
        only used for projected gradient method
    ls_eps: float, optional
        threshold for line search, by default 1e-4
        only used for projected gradient method
    eps: float, optional
        threshold for convergence, by default 1e-4
        only used for projected gradient method
    units_dict: Dict, optional
        dictionary of units, by default None
    """

    def __init__(
        self,
        rcut: float,
        ethresh: float = 1e-5,
        kspace: bool = True,
        rspace: bool = True,
        slab_corr: bool = False,
        slab_axis: int = 2,
        max_iter: int = 20,
        ls_eps: float = 1e-4,
        eps: float = 1e-4,
        units_dict: Optional[Dict] = None,
        damping: bool = True,
        sel: list[int] = None,
        kappa: Optional[float] = None,
        spacing: Optional[List[float]] = None,
    ) -> None:
        BaseForceModule.__init__(self, units_dict)

        models: Dict[str, BaseForceModule] = {
            "site": SiteForceModule(units_dict=units_dict),
            "coulomb": CoulombForceModule(
                rcut=rcut,
                ethresh=ethresh,
                kspace=kspace,
                rspace=rspace,
                slab_corr=slab_corr,
                slab_axis=slab_axis,
                units_dict=units_dict,
                kappa=kappa,
                spacing=spacing,
            ),
        }
        if damping:
            models["damping"] = GaussianDampingForceModule(units_dict=units_dict)

        self.submodels = torch.nn.ModuleDict(models)

        self.rcut = rcut
        self.max_iter = max_iter
        self.ls_eps = ls_eps
        self.eps = eps
        self.converge_iter: int = -1
        self.sel = sel

        self.slab_axis = slab_axis
        self.slab_corr = slab_corr

    def get_rcut(self) -> float:
        return self.rcut

    def get_sel(self):
        return self.sel

    def forward(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Charge equilibrium (QEq) model

        Parameters
        ----------
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        params : Dict[str, torch.Tensor]
            {
                "charge": t_charges, # (optional) initial guess for atomic charges,
                "chi": t_chi, # eletronegativity in energy / charge unit
                "hardness": t_hardness, # atomic hardness in energy / charge^2 unit
                "eta": t_eta, # Gaussian width in length unit
            }

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        energy = torch.zeros(1, device=positions.device)
        for model in self.submodels.values():
            energy = energy + model(positions, box, pairs, ds, buffer_scales, params)
        return energy

    @torch.jit.export
    def func_energy(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Energy method for QEq model

        Parameters
        ----------
        charges : torch.Tensor
            atomic charges
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        chi : torch.Tensor
            eletronegativity in energy / charge unit
        hardness : torch.Tensor
            atomic hardness in energy / charge^2 unit
        eta : torch.Tensor
            Gaussian width in length unit
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        params = {
            "charge": charges,  # (optional) initial guess for atomic charges,
            "chi": chi,  # eletronegativity in energy / charge unit
            "hardness": hardness,  # atomic hardness in energy / charge^2 unit
            "eta": eta,  # Gaussian width in length unit
        }
        energy = torch.zeros(1, device=positions.device)
        for model in self.submodels.values():
            energy = energy + model(positions, box, pairs, ds, buffer_scales, params)
        return energy

    @torch.jit.ignore
    def calc_hessian(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ):
        n_atoms = positions.shape[0]
        q_tmp = torch.zeros(
            n_atoms, device=positions.device, dtype=torch.float64, requires_grad=True
        )
        # calculate hessian
        # hessian = torch.func.hessian(self.func_energy)(
        #     q_tmp, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        # )
        y = self.func_energy(
            q_tmp, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        )
        grad = torch.autograd.grad(y, q_tmp, retain_graph=True, create_graph=True)
        hessian = []
        for anygrad in grad[0]:
            hessian.append(torch.autograd.grad(anygrad, q_tmp, retain_graph=True)[0])
        hessian = torch.stack(hessian)
        return hessian.reshape([n_atoms, n_atoms])

    @torch.jit.ignore
    def solve_matrix_inversion(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        constraint_matrix: Optional[torch.Tensor],
        constraint_vals: Optional[torch.Tensor],
        check_hessian: bool = False,
    ):
        """Solve QEq with matrix inversion method

        Parameters
        ----------
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        chi : torch.Tensor
            eletronegativity in energy / charge unit
        hardness : torch.Tensor
            atomic hardness in energy / charge^2 unit
        eta : torch.Tensor
            Gaussian width in length unit
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        constraint_matrix : torch.Tensor
            n_const * natoms, constraint matrix
        constraint_vals : torch.Tensor
            n_const, constraint values
        check_hessian : bool, optional
            (debugger) check whether hessian matrix is positive definite, by default False

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        q_opt: torch.Tensor
            optimized atomic charges
        """
        # calculate hessian
        # n_atoms * n_atoms
        hessian = self.calc_hessian(
            positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        )

        # coeff matrix as [[hessian, constraint_matrix.T], [constraint_matrix, 0]]
        # (n_atoms + n_const) * (n_atoms + n_const)
        n_atoms = positions.shape[0]
        if constraint_matrix is None:
            coeff_matrix = hessian
            vector = -chi
        else:
            n_const = constraint_matrix.shape[0]
            coeff_matrix = torch.cat(
                [
                    torch.cat([hessian, constraint_matrix.T], dim=1),
                    torch.cat(
                        [constraint_matrix, torch.zeros(n_const, n_const)], dim=1
                    ),
                ],
                dim=0,
            )
            vector = torch.concat([-chi, constraint_vals])

        if check_hessian:
            print(torch.all(torch.diag(hessian) > 0.0))

        _q_opt = torch.linalg.solve(coeff_matrix, vector.reshape(-1, 1)).reshape(-1)

        q_opt = _q_opt[:n_atoms].detach()
        q_opt.requires_grad = True
        energy = self.func_energy(
            q_opt, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        )
        # forces = -calc_grads(energy, positions)
        fermi = torch.mean(chi + torch.matmul(hessian, _q_opt[:n_atoms]))
        return energy, _q_opt[:n_atoms], torch.diag(hessian), fermi

    @torch.jit.ignore
    def solve_pgrad(
        self,
        q0: Optional[torch.Tensor],
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        constraint_matrix: torch.Tensor,
        constraint_vals: torch.Tensor,
        coeff_matrix: Optional[torch.Tensor] = None,
        reinit_q: bool = False,
        method: str = "lbfgs",
    ):
        """Solve QEq with projected gradient method

        Parameters
        ----------
        q0 : torch.Tensor
            initial guess for atomic charges, all zeros for None
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        chi : torch.Tensor
            eletronegativity in energy / charge unit
        hardness : torch.Tensor
            atomic hardness in energy / charge^2 unit
        eta : torch.Tensor
            Gaussian width in length unit
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        constraint_matrix : torch.Tensor
            n_const * natoms, constraint matrix
        constraint_vals : torch.Tensor
            n_const, constraint values
        coeff_matrix : torch.Tensor
            n_atoms * n_const, coefficient matrix
        reinit_q : bool, optional
            if reinitialize the atomic charges based on constraints, by default False
        method : str, optional
            optimization method, by default "quadratic"

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        q_opt: torch.Tensor
            optimized atomic charges
        """
        n_atoms = positions.shape[0]
        # n_const = constraint_matrix.shape[0]

        if q0 is None:
            q0 = torch.rand(n_atoms, device=positions.device, dtype=torch.float64)
            reinit_q = True
        assert q0.shape[0] == n_atoms
        # make sure the initial guess satisfy constraints
        if reinit_q:
            q0 = vector_projection(q0, constraint_matrix, constraint_vals)

        if coeff_matrix is None:
            coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

        # choose iterative algorithm
        try:
            solver_fn = getattr(self, f"_optimize_{method}")()
        except KeyError:
            raise ValueError(f"Method {method} is not supported.")

        _q_opt = custom_root(
            self.optimality,
            argnums=1,
            solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
        )(solver_fn)(
            q0,
            positions,
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            coeff_matrix,
        )

        q_opt = _q_opt.detach()
        q_opt.requires_grad = True
        energy = self.func_energy(
            q_opt, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        )
        # forces = -calc_grads(energy, positions)
        return energy, q_opt

    @torch.jit.export
    def optimality(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        constraint_matrix: torch.Tensor,
        coeff_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Optimality function (normalized projected gradient)

        Parameters
        ----------
        charges : torch.Tensor
            atomic charges
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        chi : torch.Tensor
            eletronegativity in energy / charge unit
        hardness : torch.Tensor
            atomic hardness in energy / charge^2 unit
        eta : torch.Tensor
            Gaussian width in length unit
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        constraint_matrix : torch.Tensor
            n_const * natoms, constraint matrix
        coeff_matrix : torch.Tensor
            n_atoms * n_const, coefficient matrix

        Returns
        -------
        pgrad_norm: torch.Tensor
            normalized projected gradient, ~zero at for optimal charges
        """
        energy = self.func_energy(
            charges, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        )
        pgrads = calc_pgrads(energy, charges, constraint_matrix, coeff_matrix)
        return torch.norm(pgrads) / charges.shape[0]

    def _optimize_lbfgs(self):
        def solver_fn(
            charges: torch.Tensor,
            positions: torch.Tensor,
            box: Optional[torch.Tensor],
            chi: torch.Tensor,
            hardness: torch.Tensor,
            eta: torch.Tensor,
            pairs: torch.Tensor,
            ds: torch.Tensor,
            buffer_scales: torch.Tensor,
            constraint_matrix: torch.Tensor,
            coeff_matrix: torch.Tensor,
        ):
            n_atoms = charges.shape[0]
            x0 = charges.detach()
            x0.requires_grad = True
            # optimize the energy function based on the projected gradients
            # stop the optimization according to the convergence criteria and the max iteration
            default_opt_setup = {
                "max_iter": 5,
            }
            opt_setup = default_opt_setup.copy()
            # user_opt_setup = self.kwargs.get("opt_setup", {})
            # opt_setup.update(user_opt_setup)
            optimizer = torch.optim.LBFGS([x0], **opt_setup)
            self.converge_iter = -1
            for ii in range(self.max_iter):

                def closure():
                    x0.grad = None
                    loss = self.func_energy(
                        x0, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
                    )
                    pgrads = calc_pgrads(loss, x0, constraint_matrix, coeff_matrix)
                    x0.grad = pgrads.detach()
                    return loss

                optimizer.step(closure)

                err = torch.linalg.norm(x0.grad) / n_atoms
                if err <= self.eps:
                    self.converge_iter = ii
                    # self.grads = x0.grad
                    break
            x_opt = x0
            return x_opt

        return solver_fn

    # def _optimize_cg(self):
    #     def solver_fn(
    #         charges: torch.Tensor,
    #         positions: torch.Tensor,
    #         box: Optional[torch.Tensor],
    #         chi: torch.Tensor,
    #         hardness: torch.Tensor,
    #         eta: torch.Tensor,
    #         pairs: torch.Tensor,
    #         ds: torch.Tensor,
    #         buffer_scales: torch.Tensor,
    #         constraint_matrix: torch.Tensor,
    #         coeff_matrix: torch.Tensor,
    #     ):
    #         n_atoms = charges.shape[0]
    #         x0 = charges.detach()
    #         x0.requires_grad = True
    #         # optimize the energy function based on the projected gradients
    #         # stop the optimization according to the convergence criteria and the max iteration
    #         default_opt_setup = {
    #             "method": "HZ",
    #             "line_search": "Strong_Wolfe",
    #             "c1": 1e-4,
    #             "c2": 0.9,
    #             "lr": 1,
    #         }
    #         opt_setup = default_opt_setup.copy()
    #         # user_opt_setup = self.kwargs.get("opt_setup", {})
    #         # opt_setup.update(user_opt_setup)
    #         optimizer = ncg_optimizer.BASIC([x0], **opt_setup)
    #         self.converge_iter = -1
    #         for ii in range(self.max_iter):

    #             def closure():
    #                 x0.grad = None
    #                 loss = self.func_energy(
    #                     x0, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
    #                 )

    #                 pgrads = calc_pgrads(loss, x0, constraint_matrix, coeff_matrix)
    #                 x0.grad = pgrads.detach()
    #                 return loss

    #             optimizer.step(closure)

    #             err = torch.linalg.norm(x0.grad) / n_atoms
    #             if err <= self.eps:
    #                 self.converge_iter = ii
    #                 # self.grads = x0.grad
    #                 break
    #         x_opt = x0
    #         return x_opt

    #     return solver_fn

    def _optimize_quadratic(self):
        def solver_fn(
            charges: torch.Tensor,
            positions: torch.Tensor,
            box: Optional[torch.Tensor],
            chi: torch.Tensor,
            hardness: torch.Tensor,
            eta: torch.Tensor,
            pairs: torch.Tensor,
            ds: torch.Tensor,
            buffer_scales: torch.Tensor,
            constraint_matrix: torch.Tensor,
            coeff_matrix: torch.Tensor,
        ):
            def line_search(
                x0: torch.Tensor,
                positions: torch.Tensor,
                box: torch.Tensor,
                fk: torch.Tensor,
                gk: torch.Tensor,
                pk: torch.Tensor,
            ):
                history_x = torch.arange(3, dtype=torch.float64, device=x0.device)
                history_f = [fk]

                xk = x0.detach()
                for _ in range(2):
                    if torch.norm(gk) / xk.shape[0] < self.eps:
                        return xk
                    xk = xk + pk
                    xk.detach_()
                    xk.requires_grad = True
                    if xk.grad is not None:
                        xk.grad.detach_()
                        xk.grad.zero_()
                    fk = self.func_energy(
                        xk, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
                    )
                    gk = calc_pgrads(fk, xk, constraint_matrix, coeff_matrix)
                    fk.detach_()
                    history_f.append(fk)

                _coeff_matrix = torch.stack(
                    [history_x**2, history_x, torch.ones_like(history_x)], dim=1
                )
                y = torch.stack(history_f)
                coeff = torch.linalg.solve(_coeff_matrix, y)
                # print(coeff[0])
                x_opt = x0 - coeff[1] / (2 * coeff[0]) * pk
                return x_opt

            self.converge_iter = -1

            with torch.enable_grad():
                xk = charges.detach()
                xk.requires_grad = True
                if xk.grad is not None:
                    xk.grad.detach_()
                    xk.grad.zero_()

                fk = self.func_energy(
                    xk, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
                )
                gk = calc_pgrads(fk, xk, constraint_matrix, coeff_matrix)
                pk = -gk / torch.norm(gk)
                for ii in range(self.max_iter):
                    fk.detach_()
                    pk.detach_()
                    # Selecting the step length
                    x_new = line_search(xk, positions, box, fk, gk, pk)
                    x_new.detach_()
                    x_new.requires_grad = True
                    if x_new.grad is not None:
                        x_new.grad.detach_()
                        x_new.grad.zero_()
                    fk_new = self.func_energy(
                        x_new,
                        positions,
                        box,
                        chi,
                        hardness,
                        eta,
                        pairs,
                        ds,
                        buffer_scales,
                    )
                    gk_new = calc_pgrads(fk_new, x_new, constraint_matrix, coeff_matrix)

                    xk = x_new
                    fk = fk_new

                    norm_grad = torch.norm(gk_new) / xk.shape[0]
                    if norm_grad < self.eps:
                        gk = gk_new
                        self.converge_iter = ii
                        # self.grads = gk
                        break
                    else:
                        pk = update_pr(gk, pk, gk_new)
                        gk = gk_new

            return xk

        return solver_fn


def pgrad_optimize(
    module: QEqForceModule,
    q0: Optional[torch.Tensor],
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    chi: torch.Tensor,
    hardness: torch.Tensor,
    eta: torch.Tensor,
    pairs: torch.Tensor,
    ds: torch.Tensor,
    buffer_scales: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vals: torch.Tensor,
    coeff_matrix: Optional[torch.Tensor] = None,
    reinit_q: bool = False,
    method: str = "lbfgs",
    **kwargs,
):
    """Function to optimize atomic charges with projected gradient method

    Parameters
    ----------
    module : QEqForceModule
        QEq module
    q0 : torch.Tensor
        initial guess for atomic charges, all zeros for None
    positions : torch.Tensor
        atomic positions
    box : torch.Tensor
        simulation box
    chi : torch.Tensor
        eletronegativity in energy / charge unit
    hardness : torch.Tensor
        atomic hardness in energy / charge^2 unit
    eta : torch.Tensor
        Gaussian width in length unit
    pairs : torch.Tensor
        n_pairs * 2 tensor of pairs
    ds : torch.Tensor
        i-j distance tensor
    buffer_scales : torch.Tensor
        buffer scales for each pair, 1 if i < j else 0
    constraint_matrix : torch.Tensor
        n_const * natoms, constraint matrix
    constraint_vals : torch.Tensor
        n_const, constraint values
    coeff_matrix : torch.Tensor
        n_atoms * n_const, coefficient matrix
    reinit_q : bool, optional
        if reinitialize the atomic charges based on constraints, by default False
    method : str, optional
        optimization method, by default "lbfgs"

    Returns
    -------
    energy: torch.Tensor
        energy tensor
    q_opt: torch.Tensor
        optimized atomic charges
    """
    n_atoms = positions.shape[0]
    # n_const = constraint_matrix.shape[0]

    if q0 is None:
        q0 = torch.rand(n_atoms, device=positions.device, dtype=torch.float64)
        reinit_q = True
    assert q0.shape[0] == n_atoms
    # make sure the initial guess satisfy constraints
    if reinit_q:
        q0 = vector_projection(q0, constraint_matrix, constraint_vals)

    if coeff_matrix is None:
        coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

    # choose iterative algorithm
    try:
        solver_fn = globals()[f"_pgrad_optimize_{method}"](
            module.func_energy, module.max_iter, module.eps
        )
    except KeyError:
        raise ValueError(f"Method {method} is not supported.")

    out = custom_root(
        module.optimality,
        argnums=1,
        has_aux=True,
        solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0),
    )(solver_fn)(
        q0,
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
        constraint_matrix,
        coeff_matrix,
    )
    if out[1] == -1:
        Warning("Optimization did not converge.")
    module.converge_iter = out[1]
    q_opt = out[0].detach()
    q_opt.requires_grad = True
    energy = module.func_energy(
        q_opt, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
    )
    # forces = -calc_grads(energy, positions)
    return energy, q_opt


def _pgrad_optimize_lbfgs(func_energy: Callable, max_iter: int, eps: float):
    def solver_fn(
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        constraint_matrix: torch.Tensor,
        coeff_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        n_atoms = charges.shape[0]
        x0 = charges.detach()
        x0.requires_grad = True
        # optimize the energy function based on the projected gradients
        # stop the optimization according to the convergence criteria and the max iteration
        default_opt_setup = {
            "max_iter": 5,
        }
        opt_setup = default_opt_setup.copy()
        # user_opt_setup = self.kwargs.get("opt_setup", {})
        # opt_setup.update(user_opt_setup)
        optimizer = torch.optim.LBFGS([x0], **opt_setup)
        converge_iter = -1
        for ii in range(max_iter):

            def closure():
                x0.grad = None
                loss = func_energy(
                    x0, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
                )
                pgrads = calc_pgrads(loss, x0, constraint_matrix, coeff_matrix)
                x0.grad = pgrads.detach()
                return loss

            optimizer.step(closure)

            err = torch.linalg.norm(x0.grad) / n_atoms
            if err <= eps:
                converge_iter = ii
                break
        x_opt = x0
        return x_opt, converge_iter

    return solver_fn


def _pgrad_optimize_quadratic(func_energy: Callable, max_iter: int, eps: float):
    def solver_fn(
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        chi: torch.Tensor,
        hardness: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        constraint_matrix: torch.Tensor,
        coeff_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        def line_search(
            x0: torch.Tensor,
            positions: torch.Tensor,
            box: torch.Tensor,
            fk: torch.Tensor,
            gk: torch.Tensor,
            pk: torch.Tensor,
        ):
            history_x = torch.arange(3, dtype=torch.float64, device=x0.device)
            history_f = [fk]

            xk = x0.detach()
            for _ in range(2):
                if torch.norm(gk) / xk.shape[0] < eps:
                    return xk
                xk = xk + pk
                xk.detach_()
                xk.requires_grad = True
                if xk.grad is not None:
                    xk.grad.detach_()
                    xk.grad.zero_()
                fk = func_energy(
                    xk, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
                )
                gk = calc_pgrads(fk, xk, constraint_matrix, coeff_matrix)
                fk.detach_()
                history_f.append(fk)

            _coeff_matrix = torch.stack(
                [history_x**2, history_x, torch.ones_like(history_x)], dim=1
            )
            y = torch.stack(history_f)
            coeff = torch.linalg.solve(_coeff_matrix, y)
            # print(coeff[0])
            x_opt = x0 - coeff[1] / (2 * coeff[0]) * pk
            return x_opt

        converge_iter = -1
        with torch.enable_grad():
            xk = charges.detach()
            xk.requires_grad = True
            if xk.grad is not None:
                xk.grad.detach_()
                xk.grad.zero_()

            fk = func_energy(
                xk, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
            )
            gk = calc_pgrads(fk, xk, constraint_matrix, coeff_matrix)
            pk = -gk / torch.norm(gk)
            for ii in range(max_iter):
                fk.detach_()
                pk.detach_()
                # Selecting the step length
                x_new = line_search(xk, positions, box, fk, gk, pk)
                x_new.detach_()
                x_new.requires_grad = True
                if x_new.grad is not None:
                    x_new.grad.detach_()
                    x_new.grad.zero_()
                fk_new = func_energy(
                    x_new,
                    positions,
                    box,
                    chi,
                    hardness,
                    eta,
                    pairs,
                    ds,
                    buffer_scales,
                )
                gk_new = calc_pgrads(fk_new, x_new, constraint_matrix, coeff_matrix)

                xk = x_new
                fk = fk_new

                norm_grad = torch.norm(gk_new) / xk.shape[0]
                if norm_grad < eps:
                    gk = gk_new
                    converge_iter = ii
                    break
                else:
                    pk = update_pr(gk, pk, gk_new)
                    gk = gk_new

        return xk, converge_iter

    return solver_fn


def matinv_optimize(
    module: QEqForceModule,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    chi: torch.Tensor,
    hardness: torch.Tensor,
    eta: torch.Tensor,
    pairs: torch.Tensor,
    ds: torch.Tensor,
    buffer_scales: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vals: torch.Tensor,
    **kwargs,
):
    # calculate hessian
    # n_atoms * n_atoms
    hessian = calc_hessian(
        module.func_energy, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
    )

    # coeff matrix as [[hessian, constraint_matrix.T], [constraint_matrix, 0]]
    # (n_atoms + n_const) * (n_atoms + n_const)
    n_atoms = positions.shape[0]
    if constraint_matrix is None:
        coeff_matrix = hessian
        vector = -chi
    else:
        n_const = constraint_matrix.shape[0]
        coeff_matrix = torch.cat(
            [
                torch.cat([hessian, constraint_matrix.T], dim=1),
                torch.cat([constraint_matrix, torch.zeros(n_const, n_const)], dim=1),
            ],
            dim=0,
        )
        vector = torch.concat([-chi, constraint_vals])

    _q_opt = torch.linalg.solve(coeff_matrix, vector.reshape(-1, 1)).reshape(-1)

    q_opt = _q_opt[:n_atoms].detach()
    q_opt.requires_grad = True
    energy = module.func_energy(
        q_opt, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
    )
    return energy, q_opt


def calc_hessian(
    func_energy: Callable,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    chi: torch.Tensor,
    hardness: torch.Tensor,
    eta: torch.Tensor,
    pairs: torch.Tensor,
    ds: torch.Tensor,
    buffer_scales: torch.Tensor,
):
    n_atoms = positions.shape[0]
    q_tmp = torch.zeros(
        n_atoms, device=positions.device, dtype=torch.float64, requires_grad=True
    )
    y = func_energy(q_tmp, positions, box, chi, hardness, eta, pairs, ds, buffer_scales)
    grad = torch.autograd.grad(y, q_tmp, retain_graph=True, create_graph=True)
    hessian = []
    for anygrad in grad[0]:
        hessian.append(torch.autograd.grad(anygrad, q_tmp, retain_graph=True)[0])
    hessian = torch.stack(hessian)
    return hessian.reshape([n_atoms, n_atoms])
