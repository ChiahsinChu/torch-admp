# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import units
from scipy import constants

from torch_admp.env import DEVICE, NP_PRECISION_DICT, PT_PRECISION_DICT


# @torch.jit.script
def pair_buffer_scales(pairs: torch.Tensor) -> torch.Tensor:
    dp = pairs[:, 0] - pairs[:, 1]
    return torch.where(
        dp < 0,
        torch.tensor(1, dtype=torch.long, device=pairs.device),
        torch.tensor(0, dtype=torch.long, device=pairs.device),
    )


# @torch.jit.script
def regularize_pairs(pairs: torch.Tensor, buffer_scales: torch.Tensor) -> torch.Tensor:
    a = pairs[:, 0] - buffer_scales
    b = pairs[:, 1] - buffer_scales * 2
    return torch.stack((a, b), dim=1)


# @torch.jit.script
def safe_inverse(x: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    """Safe inverse for numerical stability

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    threshold : float (default: 1e-8)
        Threshold for numerical stability

    Returns
    -------
    inv_x : torch.Tensor
        Inverse of x if x.abs() > threshold, otherwise 0
    """
    return torch.where(x.abs() > threshold, 1 / x, torch.zeros_like(x))


@torch.jit.script
def calc_grads(t_out: torch.Tensor, t_in: torch.Tensor):
    """
    Calculate gradients

    Parameters
    ----------
    t_out : torch.Tensor
        Outputs of the differentiated function
    t_in : torch.Tensor
        Inputs w.r.t. which the gradient will be returned

    Returns
    -------
    grad : torch.Tensor
        Gradients
    """
    assert t_in.requires_grad, "Input tensor requires grad"

    faked_grad = torch.ones_like(t_out)
    lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
    grad = torch.autograd.grad(
        [t_out],
        [t_in],
        grad_outputs=lst,
        retain_graph=True,
    )[0]
    assert grad is not None
    return grad


@torch.jit.script
def calc_pgrads(
    t_out: torch.Tensor,
    t_in: torch.Tensor,
    constraint_matrix: torch.Tensor,
    coeff_matrix: torch.Tensor,
):
    """
    Calculate projected gradients for constrained optimization

    Parameters
    ----------
    t_out : torch.Tensor
        Output tensor
    t_in : torch.Tensor
        Input tensor
    constraint_matrix : torch.Tensor
        n_const * natoms, constraint matrix
    coeff_matrix : torch.Tensor
        natoms * n_const, Coefficient matrix for vector projection

    Returns
    -------
    torch.Tensor
        Projected gradients
    """
    raw_grads = calc_grads(t_out, t_in)
    # n_atoms * 1
    raw_grads = raw_grads.reshape(-1, 1)
    # n_const * 1
    residual = -torch.matmul(constraint_matrix, raw_grads)
    # n_atoms * 1
    pgrads = raw_grads + torch.matmul(coeff_matrix, residual)
    return pgrads.reshape(-1)


@torch.jit.script
def vector_projection_coeff_matrix(constraint_matrix: torch.Tensor) -> torch.Tensor:
    """Calculate coefficient matrix for vector projection based on constraint matrix

    Parameters
    ----------
    constraint_matrix : torch.Tensor
        Constraint matrix (n_const * natoms).

    Returns
    -------
    coeff_mat: torch.Tensor
        Coefficient matrix (n_atoms * n_const).
    """
    constraint_matrix_t = torch.transpose(constraint_matrix, 0, 1)
    # n_atoms * n_const
    coeff_mat = torch.matmul(
        constraint_matrix_t,
        torch.linalg.inv(torch.matmul(constraint_matrix, constraint_matrix_t)),
    )
    return coeff_mat


@torch.jit.script
def vector_projection(
    vector_in: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vector projection subject to linear constraints
        P(x) = x + A^T(A A^T)^{-1}(b - Ax)

    Parameters
    ----------
    vector_in : torch.Tensor
        Input vector (n_atoms * 1).
    constraint_matrix : torch.Tensor
        Constraint matrix (n_const * natoms).
    constraint_vector : torch.Tensor, optional
        Constraint vector (n_const * 1).
        All zeros when set as None (default).

    Returns
    -------
    vector_out : torch.Tensor
        n_atoms, projected vector
    """
    if constraint_vector is None:
        constraint_vector = torch.zeros(
            [constraint_matrix.shape[0]], device=vector_in.device
        )

    # n_atoms * n_atoms
    coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)
    # n_atoms * 1
    residual = constraint_vector.reshape(-1, 1) - torch.matmul(
        constraint_matrix, vector_in
    )
    vector_out = vector_in.reshape(-1, 1) + torch.matmul(coeff_matrix, residual)
    return vector_out.reshape(-1)


class TorchConstants(torch.nn.Module):
    def __init__(self, units_dict: Optional[Dict] = None):
        """
        Consistent with ASE: https://wiki.fysik.dtu.dk/ase/ase/units.html
        Example:
        units_dict = {
            "energy": "kJ/mol",
            "length": "nm",
        }
        Electron volts (eV), Ångström (Ang), the atomic mass unit and Kelvin are defined as 1.0.
        """
        torch.nn.Module.__init__(self)
        self.register_buffer("pi", torch.tensor(np.pi, device=DEVICE))
        self.register_buffer("sqrt_pi", torch.tensor(np.sqrt(np.pi), device=DEVICE))

        if units_dict is None:
            units_dict = {}
        user_energy = units_dict.get("energy", "eV")
        user_length = units_dict.get("length", "Ang")

        # from user-defined units to ASE units (eV, Ang)
        try:
            length_coeff = getattr(units, user_length)
        except AttributeError:
            raise ValueError(f"Unknown length unit: {user_length}")
        try:
            if user_energy == "kJ/mol":
                energy_coeff = units.kJ / units.mol
            else:
                energy_coeff = getattr(units, user_energy)
        except AttributeError:
            raise ValueError(f"Unknown energy unit: {user_energy}")

        self.register_buffer(
            "length_coeff",
            torch.tensor(
                length_coeff,
                device=DEVICE,
            ),
        )
        self.register_buffer(
            "energy_coeff",
            torch.tensor(
                energy_coeff,
                device=DEVICE,
            ),
        )

        # self.register_buffer(
        #     "j2ev",
        #     torch.tensor(
        #         constants.physical_constants["joule-electron volt relationship"][0],
        #         device=DEVICE,
        #     ),
        # )
        # # convert energy unit from kJ/mol to eV/particle (DMFF <-> DP)
        # self.register_buffer(
        #     "energy_coeff", self.j2ev * constants.kilo / constants.Avogadro
        # )

        # vacuum electric permittivity in eV^-1 * angstrom^-1
        self.register_buffer(
            "epsilon",
            torch.tensor(
                constants.epsilon_0 / constants.elementary_charge * constants.angstrom,
                device=DEVICE,
            ),
        )
        # qqrd2e = 1 / (4 * np.pi * EPSILON)
        # eV
        self.register_buffer("dielectric", 1 / (4 * self.pi * self.epsilon))
        # kJ/mol
        # DIELECTRIC = torch.tensor(1389.35455846).to(DEVICE)
        # self.dielectric = 1 / (4 * self.pi * self.epsilon) / self.energy_coeff


# adapted from deepmd.pt.utils.utils.to_numpy_array
def to_numpy_array(
    xx,
):
    if xx is None:
        return None
    assert xx is not None
    # Create a reverse mapping of PT_PRECISION_DICT
    reverse_precision_dict = {v: k for k, v in PT_PRECISION_DICT.items()}
    # Use the reverse mapping to find keys with the desired value
    prec = reverse_precision_dict.get(xx.dtype, None)
    prec = NP_PRECISION_DICT.get(prec, None)
    if prec is None:
        raise ValueError(f"unknown precision {xx.dtype}")
    if xx.dtype == torch.bfloat16:
        # https://github.com/pytorch/pytorch/issues/109873
        xx = xx.float()
    return xx.detach().cpu().numpy().astype(prec)
