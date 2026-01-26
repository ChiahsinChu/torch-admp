# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Electrode models and constraints for molecular dynamics simulations.

This module implements polarizable electrode models and constraint handling
for constant potential (CONP) and constant charge (CONQ) electrode simulations.
It provides functionality for charge equilibration (QEq) with electrode constraints,
finite field calculations, and integration with LAMMPS electrode fix implementations.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from torch_admp import env
from torch_admp.qeq import QEqForceModule, matinv_optimize, pgrad_optimize
from torch_admp.utils import calc_grads, to_torch_tensor


class PolarizableElectrode(QEqForceModule):
    """Polarizable Electrode Model

    Parameters
    ----------
    rcut : float
        cutoff radius for short-range interactions
    ethresh : float, optional
        energy threshold for electrostatic interaction, by default 1e-5
    **kwargs : dict
        Additional keyword arguments passed to parent class
    """

    def __init__(self, rcut: float, ethresh: float = 1e-5, **kwargs) -> None:
        """Initialize a PolarizableElectrode instance.

        Parameters
        ----------
        rcut : float
            cutoff radius for short-range interactions
        ethresh : float, optional
            energy threshold for electrostatic interaction, by default 1e-5
        **kwargs : dict
            Additional keyword arguments passed to parent class
        """
        super().__init__(rcut, ethresh, **kwargs)

    @torch.jit.export
    def calc_coulomb_potential(
        self,
        electrode_mask: torch.Tensor | None,
        positions: torch.Tensor,
        box: torch.Tensor,
        eta: torch.Tensor,
        charges: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the vector b and add it in chi
        """
        if electrode_mask is None:
            modified_charges = charges.clone()
        else:
            modified_charges = torch.where(electrode_mask == 0, charges, 0.0)
        modified_charges.requires_grad_(True)
        energy = self.forward(
            positions,
            box,
            pairs,
            ds,
            buffer_scales,
            {
                "charge": modified_charges,
                "eta": eta,
                "hardness": torch.zeros_like(eta),
                "chi": torch.zeros_like(eta),
            },
        )
        # single frame
        assert energy.size(0) == 1
        elec_potential = calc_grads(energy[0], modified_charges)
        return elec_potential, energy

    @torch.jit.export
    def coulomb_calculator(
        self,
        positions: torch.Tensor,
        box: torch.Tensor,
        charges: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        efield: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Coulomb force for the system
        """
        device = positions.device
        dtype = positions.dtype

        _energy = self.forward(
            positions,
            box,
            pairs,
            ds,
            buffer_scales,
            {
                "charge": charges,
                "eta": eta,
                "hardness": torch.zeros_like(eta),
                "chi": torch.zeros_like(eta),
            },
        )
        # single frame
        assert _energy.size(0) == 1
        energy = _energy[0]
        if not positions.requires_grad:
            raise ValueError(
                "positions must require grad to compute forces; call positions.requires_grad_(True)"
            )
        forces = -calc_grads(energy, positions)

        if efield is not None:
            _efield = torch.zeros(3, dtype=dtype, device=device)
            _efield[self.slab_axis] = efield[self.slab_axis]
            forces = forces + charges.unsqueeze(1) * _efield
            energy = energy + torch.sum(
                _efield.reshape(1, 3) * charges.unsqueeze(1) * positions
            )
        return energy, forces


class LAMMPSElectrodeConstraint:
    """
    Register the electrode constraint for LAMMPS

    Parameters
    ----------
    indices : Union[List[int], np.ndarray]
        indices of the atoms in constraint
    mode : str
        conp or conq
    value : float
        value of the constraint (potential or charge)
    eta : float
        eta as used in LAMMPS (in length^-1)
    chi: float
        electronegativity [V]
        default: 0.0 (single element)
    hardness: float
        atomic hardness [V/e]
        default: 0.0
    ffield: bool
        if used as ffield group
    """

    def __init__(
        self,
        indices: Union[List[int], np.ndarray],
        mode: str,
        value: float,
        eta: float,
        chi: float = 0.0,
        hardness: float = 0.0,
        ffield: bool = False,
    ) -> None:
        """Initialize a LAMMPSElectrodeConstraint instance.

        Parameters
        ----------
        indices : Union[List[int], np.ndarray]
            indices of the atoms in constraint
        mode : str
            conp or conq
        value : float
            value of the constraint (potential or charge)
        eta : float
            eta as used in LAMMPS (in length^-1)
        chi : float, optional
            electronegativity [V], by default 0.0 (single element)
        hardness : float, optional
            atomic hardness [V/e], by default 0.0
        ffield : bool, optional
            if used as ffield group, by default False
        """
        self.indices = np.array(indices, dtype=int)
        # assert one dimension array
        assert self.indices.ndim == 1

        self.mode = mode
        assert mode in ["conp", "conq"], f"mode {mode} not supported"

        self.value = value
        self.eta = eta
        self.hardness = hardness
        self.chi = chi
        self.ffield = ffield


def setup_from_lammps(
    n_atoms: int,
    constraint_list: List[LAMMPSElectrodeConstraint],
    symm: bool = False,
):
    """
    Generate input data based on lammps-like constraint definitions
    """
    device = env.DEVICE
    dtype = env.GLOBAL_PT_FLOAT_PRECISION

    mask = np.zeros(n_atoms, dtype=bool)

    eta = np.zeros(n_atoms)
    chi = np.zeros(n_atoms)
    hardness = np.zeros(n_atoms)

    constraint_matrix = []
    constraint_vals = []
    ffield_electrode_mask = []
    ffield_potential = []

    for constraint in constraint_list:
        mask[constraint.indices] = True
        eta[constraint.indices] = 1 / constraint.eta * np.sqrt(2) / 2.0
        chi[constraint.indices] = constraint.chi
        hardness[constraint.indices] = constraint.hardness
        if constraint.mode == "conq":
            if symm:
                raise AttributeError(
                    "symm should be False for conq, user can implement symm by conq"
                )
            if constraint.ffield:
                raise AttributeError("ffield with conq has not been implemented yet")
            constraint_matrix.append(np.zeros((1, n_atoms)))
            constraint_matrix[-1][0, constraint.indices] = 1.0
            constraint_vals.append(constraint.value)
        if constraint.mode == "conp":
            chi[constraint.indices] -= constraint.value
        if constraint.ffield:
            ffield_electrode_mask.append(np.zeros((1, n_atoms)))
            ffield_electrode_mask[-1][0, constraint.indices] = 1.0
            ffield_potential.append(constraint.value)

    if len(ffield_electrode_mask) == 0:
        ffield_electrode_mask = None
        ffield_potential = None
    elif len(ffield_electrode_mask) == 2:
        ffield_electrode_mask = torch.tensor(
            np.concatenate(ffield_electrode_mask, axis=0),
            dtype=torch.bool,
            device=device,
        )
        ffield_potential = to_torch_tensor(np.array(ffield_potential)).to(dtype)
        # if using ffield, electroneutrality should be enforced
        # symm = True
    else:
        raise AttributeError("number of ffield group should be 0 or 2")

    if symm:
        constraint_matrix.append(np.ones((1, n_atoms)))
        constraint_vals.append(0.0)

    if len(constraint_matrix) > 0:
        constraint_matrix = to_torch_tensor(
            np.concatenate(constraint_matrix, axis=0)[:, mask]
        )
        constraint_vals = to_torch_tensor(np.array(constraint_vals))
    else:
        number_electrode = mask.sum()
        constraint_matrix = torch.zeros((0, number_electrode), device=device)
        constraint_vals = torch.zeros(0, device=device)

    return (
        to_torch_tensor(mask),
        to_torch_tensor(eta).to(dtype),
        to_torch_tensor(chi).to(dtype),
        to_torch_tensor(hardness).to(dtype),
        constraint_matrix.to(dtype),
        constraint_vals.to(dtype),
        ffield_electrode_mask,
        ffield_potential,
    )


@torch.jit.script
def finite_field_add_chi(
    positions: torch.Tensor,
    box: torch.Tensor,
    ffield_electrode_mask: torch.Tensor,
    ffield_potential: torch.Tensor,
    slab_axis: int = 2,
):
    """
    Compute the correction term for the finite field

    potential need to be same in the electrode_mask
    potential drop is potential[0] - potential[1]
    """
    assert positions.dim() == 2
    assert box.dim() == 2
    assert ffield_potential.dim() == 1
    assert ffield_electrode_mask.dim() == 2

    assert ffield_electrode_mask.shape[0] == 2
    assert positions.shape[1] == 3

    n_atoms = positions.shape[0]
    assert ffield_electrode_mask.shape[1] == n_atoms
    assert ffield_potential.shape[0] == 2

    first_electrode_mask = ffield_electrode_mask[0]
    second_electrode_mask = ffield_electrode_mask[1]

    potential_drop = ffield_potential[0] - ffield_potential[1]

    ## find max position in slab_axis for left electrode
    max_pos_first = torch.max(positions[first_electrode_mask, slab_axis])
    max_pos_second = torch.max(positions[second_electrode_mask, slab_axis])
    # only valid for orthogonality cell
    lz = box[slab_axis][slab_axis]
    normalized_positions = positions[:, slab_axis] / lz
    ### lammps fix electrode implementation
    ### cos180(-1) or cos0(1) for E(delta_psi/(r1-r2)) and r
    if max_pos_first > max_pos_second:
        zprd_offset = -1 * -1 * normalized_positions
        efield = -1 * potential_drop / lz
    else:
        zprd_offset = -1 * normalized_positions
        efield = potential_drop / lz

    potential = potential_drop * zprd_offset
    mask = first_electrode_mask | second_electrode_mask
    return potential[mask], efield


def infer(
    calculator: PolarizableElectrode,
    positions: torch.Tensor,
    box: torch.Tensor,
    charges: torch.Tensor,
    pairs: torch.Tensor,
    ds: torch.Tensor,
    buffer_scales: torch.Tensor,
    electrode_mask: torch.Tensor,
    eta: torch.Tensor,
    chi: torch.Tensor,
    hardness: torch.Tensor,
    constraint_matrix: Optional[torch.Tensor],
    constraint_vals: Optional[torch.Tensor],
    ffield_electrode_mask: Optional[torch.Tensor],
    ffield_potential: Optional[torch.Tensor],
    method: str = "lbfgs",
):
    """Perform electrode charge optimization and compute energy and forces.

    Parameters
    ----------
    calculator : PolarizableElectrode
        The polarizable electrode calculator instance
    positions : torch.Tensor
        Atomic positions with shape (n_atoms, 3)
    box : torch.Tensor
        Simulation box vectors with shape (3, 3)
    charges : torch.Tensor
        Initial atomic charges with shape (n_atoms,)
    pairs : torch.Tensor
        Neighbor pair list with shape (n_pairs, 2)
    ds : torch.Tensor
        Distances between atom pairs with shape (n_pairs,)
    buffer_scales : torch.Tensor
        Buffer scaling factors with shape (n_pairs,)
    electrode_mask : torch.Tensor
        Boolean mask identifying electrode atoms with shape (n_atoms,)
    eta : torch.Tensor
        Slater-type orbital decay parameters with shape (n_atoms,)
    chi : torch.Tensor
        Electronegativity parameters with shape (n_atoms,)
    hardness : torch.Tensor
        Atomic hardness parameters with shape (n_atoms,)
    constraint_matrix : Optional[torch.Tensor]
        Matrix of constraint equations
    constraint_vals : Optional[torch.Tensor]
        Values of constraint equations
    ffield_electrode_mask : Optional[torch.Tensor]
        Mask for finite field electrode groups
    ffield_potential : Optional[torch.Tensor]
        Applied potential for finite field calculations
    method : str, optional
        Optimization method ('lbfgs' or 'matinv'), by default "lbfgs"

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - energy: Total system energy
        - forces: Forces on all atoms
        - q_opt: Optimized charges for all atoms
    """
    (
        _positions,
        _box,
        _pairs,
        _ds,
        _buffer_scales,
    ) = calculator.standardize_input_tensor(
        positions,
        box,
        pairs,
        ds,
        buffer_scales,
    )

    # single frame
    assert _positions.shape[0] == 1
    assert _box is not None

    _q_opt, efield = charge_optimization(
        calculator,
        _positions[0],
        _box[0],
        charges,
        _pairs[0],
        _ds[0],
        _buffer_scales[0],
        electrode_mask,
        eta,
        chi,
        hardness,
        constraint_matrix,
        constraint_vals,
        ffield_electrode_mask,
        ffield_potential,
        method,
    )

    q_opt = charges.clone()
    q_opt[electrode_mask] = _q_opt

    energy, forces = calculator.coulomb_calculator(
        positions=positions,
        box=box,
        charges=q_opt,
        eta=eta,
        pairs=pairs,
        ds=ds,
        buffer_scales=buffer_scales,
        efield=efield,
    )

    return energy, forces, q_opt


def charge_optimization(
    calculator: PolarizableElectrode,
    positions: torch.Tensor,
    box: torch.Tensor,
    charges: torch.Tensor,
    pairs: torch.Tensor,
    ds: torch.Tensor,
    buffer_scales: torch.Tensor,
    electrode_mask: torch.Tensor,
    eta: torch.Tensor,
    chi: torch.Tensor,
    hardness: torch.Tensor,
    constraint_matrix: Optional[torch.Tensor],
    constraint_vals: Optional[torch.Tensor],
    ffield_electrode_mask: Optional[torch.Tensor],
    ffield_potential: Optional[torch.Tensor],
    method: str = "lbfgs",
):
    """
    Perform QEq charge optimization
    """
    device = positions.device
    dtype = positions.dtype

    if electrode_mask.sum() == 0:
        efield = None
        return charges[electrode_mask], efield
    # ffield mode
    if ffield_electrode_mask is not None and calculator.slab_corr:
        raise ValueError("Slab correction and finite field cannot be used together.")

    # electrode + electrolyte
    chi_chemical = chi
    chi_elec, _energy = calculator.calc_coulomb_potential(
        electrode_mask,
        positions,
        box,
        eta,
        charges,
        pairs,
        ds,
        buffer_scales,
    )

    # electrode
    chi = chi_chemical + chi_elec
    chi = chi[electrode_mask]
    if ffield_electrode_mask is not None:
        chi_ffield, _efield = finite_field_add_chi(
            positions,
            box,
            ffield_electrode_mask,
            ffield_potential,
            calculator.slab_axis,
        )
        chi = chi + chi_ffield

        efield = torch.zeros(3, dtype=dtype, device=device)
        efield[calculator.slab_axis] = _efield
    else:
        efield = None

    pair_mask = electrode_mask[pairs[:, 0]] & electrode_mask[pairs[:, 1]]
    # electrode_indices find the indices of electrode_mask which is True
    electrode_indices = torch.arange(
        electrode_mask.size(0), device=device, dtype=torch.long
    )[electrode_mask]
    mapping = torch.zeros(electrode_mask.size(0), dtype=torch.long, device=device)
    mapping[electrode_indices] = torch.arange(
        electrode_mask.sum().item(), device=device, dtype=torch.long
    )
    pair_i = pairs[pair_mask][:, 0]
    pair_j = pairs[pair_mask][:, 1]
    new_pairs = torch.stack([mapping[pair_i], mapping[pair_j]], dim=1)

    # common var & for matrix inversion
    kwargs = {
        "module": calculator,
        "positions": positions[electrode_mask],
        "box": box,
        "chi": chi,
        "hardness": hardness[electrode_mask],
        "eta": eta[electrode_mask],
        "pairs": new_pairs,
        "ds": ds[pair_mask],
        "buffer_scales": buffer_scales[pair_mask],
        "constraint_matrix": constraint_matrix,
        "constraint_vals": constraint_vals,
    }

    if method == "matinv":
        _energy, _q_opt = matinv_optimize(**kwargs)
    else:
        # projected gradient
        kwargs.update(
            {
                "q0": charges[electrode_mask].reshape(-1, 1),
                "method": method,
                "reinit_q": True,
            }
        )

        _energy, _q_opt = pgrad_optimize(**kwargs)

    return _q_opt, efield
