# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule, matinv_optimize, pgrad_optimize
from torch_admp.utils import calc_grads


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
        super().__init__(rcut, ethresh, **kwargs)

    @torch.jit.export
    def calc_coulomb_potential(
        self,
        electrode_mask: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        eta: torch.Tensor,
        charges: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the vector b and add it in chi
        """
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
        elec_potential = calc_grads(energy, modified_charges)
        return elec_potential

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
        energy = self.forward(
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
        forces = -calc_grads(energy, positions)

        if efield is not None:
            _efield = torch.zeros(3)
            _efield[self.slab_axis] = efield[self.slab_axis]
            forces = forces + charges.unsqueeze(1) * _efield
            energy = energy + torch.sum(
                _efield.reshape(1, 3) * charges.unsqueeze(1) * positions
            )
        return energy, forces


@torch.jit.script
def finite_field_add_chi(
    positions: torch.Tensor,
    box: torch.Tensor,
    electrode_mask: torch.Tensor,
    slab_axis: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the correction term for the finite field

    potential  need to be same in the electrode_mask
    potential drop is potential[0] - potential[1]
    """
    potential = torch.tensor([0.0, 0.0])
    electrode_mask_mid_1 = electrode_mask[electrode_mask != 0]
    if len(electrode_mask_mid_1) == 0:
        raise ValueError("No nonzero electrode values found in electrode_mask.")
    potential[0] = electrode_mask_mid_1[0]
    electrode_mask_mid_2 = electrode_mask_mid_1[
        electrode_mask_mid_1 != electrode_mask_mid_1[0]
    ]
    if len(electrode_mask_mid_2) == 0:
        potential[1] = electrode_mask_mid_1[0]
    else:
        potential[1] = electrode_mask_mid_2[0]

    if not torch.all(electrode_mask_mid_2 == electrode_mask_mid_2[0]):
        raise KeyError("Only two electrodes are supported now")

    first_electrode = torch.zeros_like(electrode_mask)
    second_electrode = torch.zeros_like(electrode_mask)

    first_electrode[electrode_mask == potential[0]] = 1
    second_electrode[electrode_mask == potential[1]] = 1
    potential_drop = potential[0] - potential[1]

    ## find max position in slab_axis for left electrode
    max_pos_first = torch.max(positions[first_electrode == 1, slab_axis])
    max_pos_second = torch.max(positions[second_electrode == 1, slab_axis])
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
    mask = (second_electrode == 1) | (first_electrode == 1)
    return potential[mask], efield


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
        eta in used in LAMMPS (in legth^-1)
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
            np.concatenate(ffield_electrode_mask, axis=0), dtype=bool
        )
        ffield_potential = torch.tensor(np.array(ffield_potential))
        # if using ffield, electroneutrality should be enforced
        # symm = True
    else:
        raise AttributeError("number of ffield group should be 0 or 2")

    if symm:
        constraint_matrix.append(np.ones((1, n_atoms)))
        constraint_vals.append(0.0)

    if len(constraint_matrix) > 0:
        constraint_matrix = torch.tensor(
            np.concatenate(constraint_matrix, axis=0)[:, mask]
        )
        constraint_vals = torch.tensor(np.array(constraint_vals))
    else:
        number_electrode = mask.sum()
        constraint_matrix = torch.zeros((0, number_electrode))
        constraint_vals = torch.zeros(0)

    return (
        torch.tensor(mask),
        torch.tensor(eta),
        torch.tensor(chi),
        torch.tensor(hardness),
        constraint_matrix,
        constraint_vals,
        ffield_electrode_mask,
        ffield_potential,
    )


@torch.jit.script
def finite_field_add_chi_new(
    positions: torch.Tensor,
    box: torch.Tensor,
    ffield_electrode_mask: torch.Tensor,
    ffield_potential: torch.Tensor,
    slab_axis: int = 2,
):
    """
    Compute the correction term for the finite field

    potential  need to be same in the electrode_mask
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
    _q_opt, efield = charge_optimization(
        calculator,
        positions,
        box,
        charges,
        pairs,
        ds,
        buffer_scales,
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
    if electrode_mask.sum() == 0:
        efield = None
        return charges[electrode_mask], efield
    # ffield mode
    if ffield_electrode_mask is not None:
        assert not calculator.slab_corr, KeyError(
            "Slab correction and finite field cannot be used together"
        )

    # electrode + electrolyte
    chi_chemical = chi
    chi_elec = calculator.calc_coulomb_potential(
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
        chi_ffield, _efield = finite_field_add_chi_new(
            positions,
            box,
            ffield_electrode_mask,
            ffield_potential,
            calculator.slab_axis,
        )
        chi = chi + chi_ffield

        efield = torch.zeros(3)
        efield[calculator.slab_axis] = _efield
    else:
        efield = None

    pair_mask = electrode_mask[pairs[:, 0]] & electrode_mask[pairs[:, 1]]
    # electrode_indices find the indices of electrode_mask which is True
    electrode_indices = torch.arange(electrode_mask.size(0))[electrode_mask]
    mapping = torch.zeros(electrode_mask.size(0), dtype=torch.long)
    mapping[electrode_indices] = torch.arange(electrode_mask.sum())
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


# >>>>>>>>>>>>>>>>>>>>>> Deprecated >>>>>>>>>>>>>>>>>>>>>>


def _conp(
    module: PolarizableElectrode,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vals: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
    ffield: Optional[bool] = False,
) -> torch.Tensor:
    """
    Constrained Potential Method implementation
    An instantiation of QEq Module for electrode systems totally

    The electrode_mask not only contains information about which atoms are electrode atoms,
    but also the potential(in volt) of the electrode atoms
    """
    n_atoms = len(electrode_mask)
    box = box if box is not None else torch.zeros(3, 3)

    if "chi" not in params:
        params["chi"] = torch.zeros(n_atoms)
    if "hardness" not in params:
        params["hardness"] = torch.zeros(n_atoms)

    electrode_params = {k: v[electrode_mask != 0] for k, v in params.items()}
    electrode_positions = positions[electrode_mask != 0]
    charge = params["charge"]

    # calculate pairs
    nblist = TorchNeighborList(cutoff=module.rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    chi_elec = module.calc_coulomb_potential(
        electrode_mask,
        positions,
        box,
        params["eta"],
        params["charge"],
        pairs,
        ds,
        buffer_scales,
    )
    chi = params["chi"] + chi_elec
    ##Apply the constant potential condition
    electrode_params["chi"] = (
        chi[electrode_mask != 0] - electrode_mask[electrode_mask != 0]
    )
    ##Apply the finite field condition
    if ffield:
        if module.slab_corr:
            raise KeyError("Slab correction and finite field cannot be used together")
        potential, _efield = finite_field_add_chi(
            positions, box, electrode_mask, module.slab_axis
        )
        electrode_params["chi"] = electrode_params["chi"] + potential
        module.ffield_flag = True

    # Neighbor list calculations
    pairs = nblist(electrode_positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    constraint_matrix = constraint_matrix[:, electrode_mask != 0]
    q0 = charge[electrode_mask != 0]
    args = [
        module,
        q0,
        electrode_positions,
        box,
        electrode_params["chi"],
        electrode_params["hardness"],
        electrode_params["eta"],
        pairs,
        ds,
        buffer_scales,
        constraint_matrix,
        constraint_vals,
        None,
        True,
        method,
    ]
    energy, q_opt = pgrad_optimize(*args)
    charges = params["charge"].clone()
    charges[electrode_mask != 0] = q_opt
    charge_opt = torch.Tensor(charges)
    charge_opt.requires_grad_(True)

    return charge_opt


def conp(
    module: PolarizableElectrode,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
    symm: bool = True,
    ffield: Optional[bool] = False,
) -> torch.Tensor:
    """
    Lammps like implementation for User which is more convenient
    """
    # n_electrode_atoms = len(electrode_mask[electrode_mask != 0])
    n_atoms = len(electrode_mask)
    if symm:
        constraint_matrix = torch.ones([1, n_atoms])
        constraint_vals = torch.zeros(1)
    else:
        constraint_matrix = torch.zeros([0, n_atoms])
        constraint_vals = torch.zeros(0)
    if ffield:
        if not symm:
            raise KeyError("Finite field only support charge neutral condition")

    return _conp(
        module,
        electrode_mask,
        positions,
        constraint_matrix,
        constraint_vals,
        box,
        params,
        method,
        ffield,
    )


def _conq(
    module: PolarizableElectrode,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    constraint_matrix: torch.Tensor,
    constraint_vals: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
    ffield: Optional[bool] = False,
) -> torch.Tensor:
    """
    Constrained Potential Method implementation
    An instantiation of QEq Module for electrode systems totally

    The electrode_mask not only contains information about which atoms are electrode atoms,
    but also the potential(in volt) of the electrode atoms
    """
    n_atoms = len(electrode_mask)
    box = box if box is not None else torch.zeros(3, 3)

    if "chi" not in params:
        params["chi"] = torch.zeros(n_atoms)
    if "hardness" not in params:
        params["hardness"] = torch.zeros(n_atoms)

    electrode_params = {k: v[electrode_mask != 0] for k, v in params.items()}
    electrode_positions = positions[electrode_mask != 0]
    charge = params["charge"]

    # calculate pairs
    nblist = TorchNeighborList(cutoff=module.rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    chi_elec = module.calc_coulomb_potential(
        electrode_mask,
        positions,
        box,
        params["eta"],
        params["charge"],
        pairs,
        ds,
        buffer_scales,
    )
    chi = params["chi"] + chi_elec

    electrode_params["chi"] = chi[electrode_mask != 0]

    ##Apply the finite field condition
    if ffield:
        if module.slab_corr:
            raise KeyError("Slab correction and finite field cannot be used together")

        raise KeyError("conq with finite field has not been implemented")

    # Neighbor list calculations
    pairs = nblist(electrode_positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    constraint_matrix = constraint_matrix[:, electrode_mask != 0]

    q0 = charge[electrode_mask != 0].reshape(-1, 1)

    args = [
        module,
        q0,
        electrode_positions,
        box,
        electrode_params["chi"],
        electrode_params["hardness"],
        electrode_params["eta"],
        pairs,
        ds,
        buffer_scales,
        constraint_matrix,
        constraint_vals,
        None,
        True,
        method,
    ]
    energy, q_opt = pgrad_optimize(*args)
    charges = params["charge"].clone()
    charges[electrode_mask != 0] = q_opt
    charge_opt = torch.Tensor(charges)
    charge_opt.requires_grad_(True)

    return charge_opt


def conq(
    module: PolarizableElectrode,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    charge_constraint_dict: Dict[int, torch.Tensor],
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
    ffield: Optional[bool] = False,
) -> torch.Tensor:
    """
    Lammps like implementation for User which is more convenient
    which also can realize by conp
    charge_constraint_dict: Dict
        key is int data correspond to the electrode mask
        value is the constraint charge value
    """
    n_atoms = len(electrode_mask)
    tolerance = 1e-6
    if len(charge_constraint_dict) > 2:
        raise KeyError("Only one or two electrodes are supported Now")
    if len(charge_constraint_dict) == 1:
        constraint_matrix = torch.ones([1, n_atoms])
        constraint_vals = torch.tensor([list(charge_constraint_dict.values())[0]])
    else:
        key1 = list(charge_constraint_dict.keys())[0]
        key2 = list(charge_constraint_dict.keys())[1]

        row1 = torch.zeros([1, n_atoms])
        row1[0, torch.abs(electrode_mask - key1) < tolerance] = 1
        row2 = torch.zeros([1, n_atoms])
        row2[0, torch.abs(electrode_mask - key2) < tolerance] = 1

        constraint_matrix = torch.cat([row1, row2], dim=0)
        constraint_vals = torch.tensor(
            [
                [list(charge_constraint_dict.values())[0]],
                [list(charge_constraint_dict.values())[1]],
            ]
        )
    if ffield:
        raise KeyError("conq with finite field has not been implemented")

    return _conq(
        module,
        electrode_mask,
        positions,
        constraint_matrix,
        constraint_vals,
        box,
        params,
        method,
        ffield,
    )


def conq_aimd_data(
    module: PolarizableElectrode,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
) -> torch.Tensor:
    charge = params["charge"]
    constraint_vals = torch.sum(charge[electrode_mask == 0]) * -1
    constraint_matrix = torch.ones([1, len(electrode_mask)])

    return _conq(
        module=module,
        electrode_mask=electrode_mask,
        positions=positions,
        constraint_matrix=constraint_matrix,
        constraint_vals=constraint_vals,
        box=box,
        params=params,
        method=method,
        ffield=False,
    )
