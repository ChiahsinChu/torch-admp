# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Particle Mesh Ewald (PME) implementation for torch-admp.

This module implements the Coulomb energy calculation using the Particle Mesh Ewald
method, which splits the calculation into real-space and reciprocal-space
contributions for improved efficiency in periodic systems. It includes support for
slab corrections and various optimization methods.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import special

from torch_admp.base_force import BaseForceModule
from torch_admp.env import DEVICE, GLOBAL_PT_FLOAT_PRECISION
from torch_admp.recip import bspline, setup_kpts, setup_kpts_integer, spread_charges
from torch_admp.utils import safe_inverse, to_torch_tensor


class CoulombForceModule(BaseForceModule):
    """
    Coulomb energy module with Particle Mesh Ewald (PME).

    This module implements the Coulomb energy calculation using the Particle Mesh Ewald
    method, which splits the calculation into real-space and reciprocal-space
    contributions for improved efficiency in periodic systems.

    Parameters
    ----------
    rcut : float
        Real-space cutoff distance
    ethresh : float, optional
        Energy threshold for PME accuracy, by default 1e-5
    kspace : bool, optional
        Whether to include reciprocal space contribution, by default True
    rspace : bool, optional
        Whether to include real space contribution, by default True
    slab_corr : bool, optional
        Whether to apply slab correction, by default False
    slab_axis : int, optional
        Axis at which the slab correction is applied, by default 2
    units_dict : Optional[Dict], optional
        Dictionary of unit conversions, by default None
    sel : Optional[list[int]], optional
        Selection list for neighbor list, by default None
    kappa : Optional[float], optional
        Inverse screening length [Å^-1], by default None
    spacing : Optional[List[float]], optional
        Grid spacing for reciprocal space, by default None
    """

    def __init__(
        self,
        rcut: float,
        ethresh: float = 1e-5,
        kspace: bool = True,
        rspace: bool = True,
        slab_corr: bool = False,
        slab_axis: int = 2,
        units_dict: Optional[Dict] = None,
        sel: Optional[list[int]] = None,
        kappa: Optional[float] = None,
        spacing: Union[List[float], float, None] = None,
    ) -> None:
        """
        Initialize the CoulombForceModule with PME.

        Parameters
        ----------
        rcut : float
            Real-space cutoff distance
        ethresh : float, optional
            Energy threshold for PME accuracy, by default 1e-5
        kspace : bool, optional
            Whether to include reciprocal space contribution, by default True
        rspace : bool, optional
            Whether to include real space contribution, by default True
        slab_corr : bool, optional
            Whether to apply slab correction, by default False
        slab_axis : int, optional
            Axis at which the slab correction is applied, by default 2
        units_dict : Optional[Dict], optional
            Dictionary of unit conversions, by default None
        sel : Optional[list[int]], optional
            Selection list for neighbor list, by default None
        kappa : Optional[float], optional
            Inverse screening length [Å^-1], by default None
        spacing : Optional[List[float]], optional
            Grid spacing for reciprocal space, by default None
        """
        BaseForceModule.__init__(self, units_dict)

        self.kspace_flag = kspace
        if kappa is not None:
            self.kappa = kappa
        else:
            if self.kspace_flag:
                kappa = math.sqrt(-math.log(2 * ethresh)) / rcut
                self.kappa = kappa / getattr(self.const_lib, "length_coeff")
            else:
                self.kappa = 0.0
        self.ethresh = ethresh
        self.kmesh = torch.ones(3, dtype=torch.long, device=DEVICE)
        if spacing is not None:
            if isinstance(spacing, float):
                spacing = [spacing, spacing, spacing]
            self.spacing = to_torch_tensor(np.array(spacing)).to(GLOBAL_PT_FLOAT_PRECISION)
        else:
            self.spacing = spacing
        self.rspace_flag = rspace
        self.slab_corr_flag = slab_corr
        self.slab_axis = slab_axis

        self.real_energy = to_torch_tensor(np.zeros(1)).to(GLOBAL_PT_FLOAT_PRECISION)
        self.reciprocal_energy = to_torch_tensor(np.zeros(1)).to(GLOBAL_PT_FLOAT_PRECISION)
        self.self_energy = to_torch_tensor(np.zeros(1)).to(GLOBAL_PT_FLOAT_PRECISION)
        self.non_neutral_energy = to_torch_tensor(np.zeros(1)).to(GLOBAL_PT_FLOAT_PRECISION)
        self.slab_corr_energy = to_torch_tensor(np.zeros(1)).to(GLOBAL_PT_FLOAT_PRECISION)

        # Currently only supprots pme_order=6
        # Because only the 6-th order spline function is hard implemented
        self.pme_order: int = 6
        n_mesh = int(self.pme_order**3)

        # global variables for the reciprocal module, all related to pme_order
        bspline_range = torch.arange(
            -self.pme_order // 2, self.pme_order // 2, device=DEVICE
        )
        shift_y, shift_x, shift_z = torch.meshgrid(
            bspline_range, bspline_range, bspline_range, indexing="ij"
        )
        self.pme_shifts = (
            torch.stack((shift_x, shift_y, shift_z))
            .transpose(0, 3)
            .reshape((1, n_mesh, 3))
        )

        self.rcut = rcut
        self.sel = sel

    def get_rcut(self) -> float:
        """
        Get the cutoff radius.

        Returns
        -------
        float
            Cutoff radius
        """
        return self.rcut

    def get_sel(self) -> Optional[list[int]]:
        """
        Get `sel` list of DP model.

        Returns
        -------
        Optional[list[int]]
            The number of selected neighbors for each type of atom.
        """
        return self.sel

    def _forward_impl(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Coulomb energy model with PME algorithm for KSpace

        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions with shape (nframes, natoms, 3). Each frame contains
            x, y, z coordinates of all atoms.
        box : Optional[torch.Tensor]
            Simulation box vectors with shape (nframes, 3, 3) or None if input was None.
            Each frame contains three box vectors. Required for periodic boundary conditions.
        pairs : torch.Tensor
            Tensor of atom pairs with shape (nframes, n_pairs, 2). Each frame contains
            the indices of two atoms that form a pair.
        ds : torch.Tensor
            Distance tensor with shape (nframes, n_pairs). Contains the distances
            between atom pairs specified in the pairs tensor for each frame.
        buffer_scales : torch.Tensor
            Buffer scales for each pair with shape (nframes, n_pairs). Contains values
            of 1 if i < j else 0 for each pair, used for buffer management.
        params : Dict[str, torch.Tensor]
            Dictionary of parameters for the Coulomb model:
            {"charge": t_charges} # atomic charges with shape (nframes, natoms).

        Returns
        -------
        energy: torch.Tensor
            Scalar energy tensor for single system or (nframes,) for batched systems
            representing the total Coulomb energy.
        """
        # nframes, natoms,
        nf = positions.size(0)
        na = positions.size(1)
        charges = params["charge"].reshape(nf, na)

        if box is not None:
            energy = self._forward_pbc(
                charges, positions, box, pairs, ds, buffer_scales
            )
        else:
            energy = self._forward_obc(charges, pairs, ds, buffer_scales)
        return energy / getattr(self.const_lib, "energy_coeff")

    def _forward_pbc(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Coulomb energy for periodic boundary conditions.

        Parameters
        ----------
        charges : torch.Tensor
            Atomic charges
        positions : torch.Tensor
            Atomic positions
        box : torch.Tensor
            Simulation box vectors
        pairs : torch.Tensor
            Tensor of atom pairs
        ds : torch.Tensor
            Distance tensor
        buffer_scales : torch.Tensor
            Buffer scales for each pair

        Returns
        -------
        torch.Tensor
            Total Coulomb energy
        """
        if self.rspace_flag:
            self.real_energy = self._forward_pbc_real(charges, pairs, ds, buffer_scales)
        if self.kspace_flag:
            self.reciprocal_energy = self._forward_pbc_reciprocal(
                charges, positions, box
            )
            self.self_energy = self._forward_pbc_self(charges)
            self.non_neutral_energy = self._forward_pbc_non_neutral(charges, box)
        if self.slab_corr_flag:
            self.slab_corr_energy = self._forward_slab_corr(charges, positions, box, ds)
        coul_energy = (
            self.real_energy
            + self.reciprocal_energy
            + self.self_energy
            + self.non_neutral_energy
            + self.slab_corr_energy
        )
        return coul_energy

    def _forward_pbc_real(
        self,
        charges: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate real-space contribution to Coulomb energy.

        Parameters
        ----------
        charges : torch.Tensor
            Atomic charges
        pairs : torch.Tensor
            Tensor of atom pairs
        ds : torch.Tensor
            Distance tensor
        buffer_scales : torch.Tensor
            Buffer scales for each pair

        Returns
        -------
        torch.Tensor
            Real-space contribution to Coulomb energy
        """
        # qi or qj: nf, np
        qi = torch.gather(charges, 1, pairs[:, :, 0])
        qj = torch.gather(charges, 1, pairs[:, :, 1])

        e_sr = torch.sum(
            torch.erfc(self.kappa * ds)
            * qi
            * qj
            * safe_inverse(ds, threshold=1e-4)
            * buffer_scales,
            dim=-1,
        ) * getattr(self.const_lib, "dielectric")
        return e_sr

    def _forward_pbc_reciprocal(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate reciprocal-space contribution to Coulomb energy.

        Parameters
        ----------
        charges : torch.Tensor
            Atomic charges
        positions : torch.Tensor
            Atomic positions
        box : torch.Tensor
            Simulation box vectors

        Returns
        -------
        torch.Tensor
            Reciprocal-space contribution to Coulomb energy
        """
        device = positions.device
        nf = positions.size(0)

        box_inv = torch.linalg.inv(box)
        volume = torch.det(box)
        box_diag = torch.diagonal(box, dim1=1, dim2=2)
        if self.spacing is not None:
            spacing = torch.as_tensor(
                self.spacing, dtype=box_diag.dtype, device=box_diag.device
            )
            self.kmesh = torch.ceil(box_diag / spacing).to(torch.long)
        else:
            self.kmesh = torch.ceil(
                2 * self.kappa * box_diag / (3.0 * self.ethresh ** (1.0 / 5.0))
            ).to(torch.long)

        # for electrostatic, exclude gamma point
        gamma_flag = False
        coeff_k_func = _coeff_k_1
        all_ener = []
        for ii in range(nf):
            # charges: -1, 1
            _charges = charges[ii].reshape(-1, 1)
            # mapping charges onto mesh
            meshed_charges = spread_charges(
                positions[ii],
                box_inv[ii],
                _charges,
                self.kmesh[ii],
                self.pme_shifts,
                self.pme_order,
            )
            kpts_int = setup_kpts_integer(self.kmesh[ii])
            kpts = setup_kpts(box_inv[ii], kpts_int)
            m = torch.linspace(
                -self.pme_order // 2 + 1,
                self.pme_order // 2 - 1,
                self.pme_order - 1,
                device=device,
            ).reshape(self.pme_order - 1, 1, 1)
            theta_k = torch.prod(
                torch.sum(
                    bspline(m + self.pme_order / 2)
                    * torch.cos(
                        2
                        * torch.pi
                        * m
                        * kpts_int[None]
                        / self.kmesh[ii].float().reshape(1, 1, 3)
                    ),
                    dim=0,
                ),
                dim=1,
            )

            S_k = torch.fft.fftn(meshed_charges).flatten()
            if not gamma_flag:
                coeff_k = coeff_k_func(kpts[3, 1:], self.kappa, volume[ii])
                E_k = coeff_k * (
                    (S_k[1:].real ** 2 + S_k[1:].imag ** 2) / theta_k[1:] ** 2
                )
            else:
                coeff_k = coeff_k_func(kpts[3, :], self.kappa, volume[ii])
                E_k = coeff_k * ((S_k.real**2 + S_k.imag**2) / theta_k**2)
            # return
            all_ener.append(torch.sum(E_k) * getattr(self.const_lib, "dielectric"))
        return torch.stack(all_ener)

    def _forward_pbc_self(
        self,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        -\frac{\alpha}{\sqrt{\pi}} \sum_{i} q_i^2
        """
        if self.kspace_flag:
            coeff = self.kappa / getattr(self.const_lib, "sqrt_pi")
            return -torch.sum(coeff * charges**2, dim=-1) * getattr(
                self.const_lib, "dielectric"
            )
        else:
            return torch.zeros(charges.size(0), device=charges.device)

    def _forward_pbc_non_neutral(
        self,
        charges: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate non-neutral correction for Coulomb energy.

        Parameters
        ----------
        charges : torch.Tensor
            Atomic charges
        box : torch.Tensor
            Simulation box vectors

        Returns
        -------
        torch.Tensor
            Non-neutral correction to Coulomb energy
        """
        volume = torch.det(box)
        # total charge
        Q_tot = torch.sum(charges, dim=-1)

        coeff = (
            -getattr(self.const_lib, "pi")
            / (2 * volume * self.kappa**2)
            * getattr(self.const_lib, "dielectric")
        )
        e_corr_non = coeff * Q_tot**2
        return e_corr_non

    def _forward_slab_corr(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        ds: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Slab correction energy (ref: 10.1063/1.3216473)

        E = \frac{2\pi}{V} \varepsilon \left( M_z^2 - Q_{\text{tot}} \sum_i q_i z_i + \frac{Q_{\text{tot}}^2 L_z^2}{12} \right)
        """
        volume = torch.det(box)
        pre_corr = (
            2
            * getattr(self.const_lib, "pi")
            / volume
            * getattr(self.const_lib, "dielectric")
        )
        # dipole moment in axis direction
        Mz = torch.sum(charges * positions[:, :, self.slab_axis], dim=-1)
        # total charge
        Q_tot = torch.sum(charges, dim=-1)
        # length of the box in axis direction
        Lz = torch.norm(box[:, self.slab_axis], dim=-1)

        e_corr = pre_corr * (
            Mz**2
            - Q_tot
            * (torch.sum(charges * positions[:, :, self.slab_axis] ** 2, dim=-1))
            - torch.pow(Q_tot, 2) * torch.pow(Lz, 2) / 12
        )
        return e_corr

    def _forward_obc(
        self,
        charges: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Coulomb energy for open boundary conditions.

        Parameters
        ----------
        charges : torch.Tensor
            Atomic charges
        pairs : torch.Tensor
            Tensor of atom pairs
        ds : torch.Tensor
            Distance tensor
        buffer_scales : torch.Tensor
            Buffer scales for each pair

        Returns
        -------
        torch.Tensor
            Coulomb energy for open boundary conditions
        """
        # qi or qj: nf, np
        qi = torch.gather(charges, 1, pairs[:, :, 0])
        qj = torch.gather(charges, 1, pairs[:, :, 1])
        ds_inv = safe_inverse(ds)
        E_inter = qi * qj * getattr(self.const_lib, "dielectric") * ds_inv
        coul_energy = torch.sum(E_inter * buffer_scales, dim=-1)
        return coul_energy


def setup_ewald_parameters(
    rcut: float,
    box: Union[torch.Tensor, np.ndarray, None] = None,
    threshold: float = 1e-5,
    spacing: Optional[float] = None,
    method: str = "openmm",
) -> Tuple[float, int, int, int]:
    """
    Given the cutoff distance, and the required precision, determine the parameters used in
    Ewald sum, including: kappa, kx, ky, and kz.

    Parameters
    ----------
    rcut : float
        Cutoff distance
    threshold : float
        Expected average relative errors in force
    box : torch.Tensor or np.ndarray
        Lattice vectors in (3 x 3) matrix
        Keep unit consistent with rcut
    spacing : float, optional
        Fourier spacing to determine K, used in gromacs method
        Keep unit consistent with rcut
    method : str
        Method to determine ewald parameters.
        Valid values: "openmm" or "gromacs".
        If openmm, the algorithm can refer to http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-particle-mesh-ewald
        If gromacs, the algorithm is adapted from gromacs source code

    Returns
    -------
    kappa: float
        Ewald parameter, in 1/lenght unit
    kx, ky, kz: int
        number of the k-points mesh
    """
    if box is None:
        return 0.1, 1, 1, 1

    if isinstance(box, torch.Tensor):
        box = torch.Tensor.numpy(box, force=True)

    # assert orthogonal box
    assert (
        np.inner(box[0], box[1]) == 0.0
    ), "Only orthogonal box is supported currently."
    assert (
        np.inner(box[0], box[2]) == 0.0
    ), "Only orthogonal box is supported currently."
    assert (
        np.inner(box[1], box[2]) == 0.0
    ), "Only orthogonal box is supported currently."

    if method == "openmm":
        kappa = np.sqrt(-np.log(2 * threshold)) / rcut
        kx = np.ceil(2 * kappa * box[0, 0] / (3.0 * threshold ** (1.0 / 5.0))).astype(
            int
        )
        ky = np.ceil(2 * kappa * box[1, 1] / (3.0 * threshold ** (1.0 / 5.0))).astype(
            int
        )
        kz = np.ceil(2 * kappa * box[2, 2] / (3.0 * threshold ** (1.0 / 5.0))).astype(
            int
        )
    elif method == "gromacs":
        assert spacing is not None, "Spacing must be provided for gromacs method."
        # determine kappa
        kappa = 5.0
        i = 0
        while special.erfc(kappa * rcut) > threshold:
            i += 1
            kappa *= 2

        n = i + 60
        low = 0.0
        high = kappa
        for k in range(n):
            kappa = (low + high) / 2
            if special.erfc(kappa * rcut) > threshold:
                low = kappa
            else:
                high = kappa
        # determine K
        kx = np.ceil(box[0, 0] / spacing).astype(int)
        ky = np.ceil(box[1, 1] / spacing).astype(int)
        kz = np.ceil(box[2, 2] / spacing).astype(int)
    else:
        raise ValueError(
            f"Invalid method: {method}." "Valid methods: 'openmm', 'gromacs'"
        )

    return kappa, kx, ky, kz


def _coeff_k_1(
    ksq: torch.Tensor,
    kappa: float,
    volume: torch.Tensor,
):
    """
    Calculate coefficient for k-space contribution.

    Parameters
    ----------
    ksq : torch.Tensor
        Square of k-vectors
    kappa : float
        Eald parameter
    volume : torch.Tensor
        Volume of the simulation box

    Returns
    -------
    torch.Tensor
        Coefficient for k-space contribution
    """
    return 2 * torch.pi / volume / ksq * torch.exp(-ksq / 4 / kappa**2)
