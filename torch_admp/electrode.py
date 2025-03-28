# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule, pgrad_optimize
from torch_admp.utils import calc_grads


class PolarisableElectrode(QEqForceModule):
    """Polarizable Electrode Model

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
        ≈
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

    def __init__(self, rcut: float, ethresh: float = 1e-5, **kwargs) -> None:
        super().__init__(rcut, ethresh, **kwargs)

    @torch.jit.export
    def calc_coulomb_potential(
        self,
        positions: torch.Tensor,
        box: torch.Tensor,
        charges: torch.Tensor,
        eta: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        calculate the coulomb potential for the system
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
        elec_potential = calc_grads(energy, charges)
        return elec_potential

    @torch.jit.export
    def coulomb_potential_add_chi(
        self,
        electrode_mask: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        chi: torch.Tensor,
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
        elec_potential = self.calc_coulomb_potential(
            positions, box, modified_charges, eta, pairs, ds, buffer_scales
        )
        return chi + elec_potential

    @torch.jit.export
    def finite_field_add_chi(
        self,
        positions: torch.Tensor,
        box: torch.Tensor,
        electrode_mask: torch.Tensor,
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

        slab_axis = self.slab_axis

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
        efield: torch.Tensor = None,
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

        vector = torch.tensor([0, 0, 0])
        vector[self.slab_axis] = 1

        if efield is not None:
            forces += efield * charges.unsqueeze(1) * vector
            energy += torch.sum(efield * charges * positions[:, self.slab_axis])

        return energy, forces


def _conp(
    module: PolarisableElectrode,
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

    chi = module.coulomb_potential_add_chi(
        electrode_mask,
        positions,
        box,
        params["chi"],
        params["eta"],
        params["charge"],
        pairs,
        ds,
        buffer_scales,
    )
    electrode_params["chi"] = chi[electrode_mask != 0]

    ##Apply the constant potential condition
    electrode_params["chi"] -= electrode_mask[electrode_mask != 0]

    ##Apply the finite field condition
    if ffield:
        if module.slab_corr:
            raise KeyError("Slab correction and finite field cannot be used together")
        potential, efield = module.finite_field_add_chi(positions, box, electrode_mask)
        electrode_params["chi"] += potential
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
    module: PolarisableElectrode,
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
    module: PolarisableElectrode,
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

    chi = module.coulomb_potential_add_chi(
        electrode_mask,
        positions,
        box,
        params["chi"],
        params["eta"],
        params["charge"],
        pairs,
        ds,
        buffer_scales,
    )
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
    module: PolarisableElectrode,
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
    module: PolarisableElectrode,
    electrode_mask: torch.Tensor,
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    params: Dict[str, torch.Tensor],
    method: Optional[str] = "lbfgs",
) -> torch.Tensor:
    charge = params["charge"]
    constraint_vals = torch.sum(charge[electrode_mask == 0]) * -1
    electrode_positions = positions[electrode_mask == 1]
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


class BackupPolarisableElectrode(QEqForceModule):
    """Polarisable electrode model

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
        **kwargs,
    ) -> None:
        super().__init__(
            rcut=rcut,
            ethresh=ethresh,
            **kwargs,
        )

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
                "electrode_mask": t_mask, # mask for QEq calculation, False for fixed charges and 1 for QEq
            }

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        if "hardness" not in params:
            params["hardness"] = torch.zeros_like(params["chi"])

        full_charges = torch.where(params["electrode_mask"], 0.0, params["charge"])
        full_params = copy.deepcopy(params)
        full_params["charge"] = full_charges
        energy = torch.zeros(1, device=positions.device)
        for model in self.submodels.values():
            energy = energy + model(
                positions, box, pairs, ds, buffer_scales, full_params
            )
        elec_potential = calc_grads(energy, full_charges)

        pair_mask = (
            params["electrode_mask"][pairs[:, 0]]
            & params["electrode_mask"][pairs[:, 1]]
        )

        chi = (
            params["chi"][params["electrode_mask"]]
            + elec_potential[params["electrode_mask"]]
        )
        eta = params["eta"][params["electrode_mask"]]
        # charge = params["charge"][params["electrode_mask"]]
        hardness = params["hardness"][params["electrode_mask"]]

        qeq_out = self.solve_matrix_inversion(
            positions[params["electrode_mask"]],
            box,
            chi,
            hardness,
            eta,
            pairs[pair_mask],
            ds[pair_mask],
            buffer_scales[pair_mask],
            params["constraint_matrix"],
            params["constraint_vals"],
        )
        # todo: add choice for pgrads
        opt_charge = qeq_out[1]
        print(opt_charge)

        total_energy = energy + qeq_out[0]
        return total_energy


class LAMMPSElectrodeConstraint:
    def __init__(
        self,
        indices: Union[List[int], np.ndarray],
        mode: str,
        value: float,
        eta: float,
        chi: float = 0.0,
    ) -> None:
        self.indices = np.array(indices, dtype=int)
        # assert one dimension array
        assert self.indices.ndim == 1

        self.mode = mode
        assert mode in ["conp", "conq"], f"mode {mode} not supported"

        self.value = value
        self.eta = eta
        self.chi = chi


def setup_from_lammps(
    n_atoms: int,
    constraint_list: List[LAMMPSElectrodeConstraint],
    symm: bool = True,
):
    mask = np.zeros(n_atoms, dtype=bool)
    chi = np.zeros(n_atoms)
    eta = np.zeros(n_atoms)
    constraint_matrix = []
    constraint_vals = []

    for constraint in constraint_list:
        mask[constraint.indices] = True
        chi[constraint.indices] = constraint.chi
        eta[constraint.indices] = 1 / constraint.eta * np.sqrt(2) / 2.0
        if constraint.mode == "conq":
            constraint_matrix.append(np.zeros((1, n_atoms)))
            constraint_matrix[-1][0, constraint.indices] = 1.0
            constraint_vals.append(constraint.value)

    if symm:
        constraint_matrix.append(np.ones((1, n_atoms)))
        constraint_vals.append(0.0)

    if len(constraint_matrix) > 0:
        constraint_matrix = np.concatenate(constraint_matrix, axis=0)[:, mask]
        constraint_vals = np.array(constraint_vals)
        return (
            torch.tensor(mask),
            torch.tensor(chi),
            torch.tensor(eta),
            torch.tensor(constraint_matrix),
            torch.tensor(constraint_vals),
        )
    else:
        return torch.tensor(mask), torch.tensor(chi), torch.tensor(eta), None, None
