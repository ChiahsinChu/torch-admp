# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import special

from torch_admp.base_force import BaseForceModule
from torch_admp.recip import bspline, setup_kpts, setup_kpts_integer, spread_charges
from torch_admp.utils import safe_inverse


class CoulombForceModule(BaseForceModule):
    """
    Coulomb energy

    Parameters
    ----------
    kappa: float
        inverse screening length [Å^-1]
    kmesh: Tuple[int, int, int]
        number of k-points mesh
    kspace: bool
        whether the reciprocal part is included
    rspace: bool
        whether the real space part is included
    slab_corr: bool
        whether the slab correction is applied
    slab_axis: int
        axis at which the slab correction is applied
    units_dict: Dict
        dictionary of units
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
        sel: list[int] = None,
        kappa: Optional[float] = None,
        spacing: Optional[List[float]] = None,
    ) -> None:
        BaseForceModule.__init__(self, units_dict)

        self.kspace_flag = kspace
        if kappa is not None:
            self.kappa = kappa
        else:
            if self.kspace_flag:
                kappa = math.sqrt(-math.log(2 * ethresh)) / rcut
                self.kappa = kappa / self.const_lib.length_coeff
            else:
                self.kappa = 0.0
        self.ethresh = ethresh
        self.kmesh = torch.ones(3, dtype=torch.long)
        if spacing is not None:
            self.spacing = torch.tensor(np.array(spacing), dtype=torch.float64)
        else:
            self.spacing = spacing
        self.rspace_flag = rspace
        self.slab_corr_flag = slab_corr
        self.slab_axis = slab_axis

        # todo: how to set device
        self.real_energy = torch.tensor(0.0)
        self.reciprocal_energy = torch.tensor(0.0)
        self.self_energy = torch.tensor(0.0)
        self.non_neutral_energy = torch.tensor(0.0)
        self.slab_corr_energy = torch.tensor(0.0)

        # Currently only supprots pme_order=6
        # Because only the 6-th order spline function is hard implemented
        self.pme_order: int = 6
        n_mesh = int(self.pme_order**3)

        # global variables for the reciprocal module, all related to pme_order
        # todo: how to set device
        bspline_range = torch.arange(-self.pme_order // 2, self.pme_order // 2)
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
        """Coulomb energy model with PME algorithm for KSpace

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
            {"charge": t_charges}

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
        positions = positions * self.const_lib.length_coeff
        ds = ds * self.const_lib.length_coeff
        charges = params["charge"]

        if box is not None:
            box = box * self.const_lib.length_coeff
            energy = self._forward_pbc(
                charges, positions, box, pairs, ds, buffer_scales
            )
        else:
            energy = self._forward_obc(charges, pairs, ds, buffer_scales)
        return energy / self.const_lib.energy_coeff

    def _forward_pbc(
        self,
        _charge: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        charges = torch.reshape(_charge, (-1, 1))

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
        q_i = charges[pairs[:, 0]].reshape(-1)
        q_j = charges[pairs[:, 1]].reshape(-1)

        e_sr = (
            torch.sum(
                torch.erfc(self.kappa * ds)
                * q_i
                * q_j
                * safe_inverse(ds, threshold=1e-4)
                * buffer_scales
            )
            * self.const_lib.dielectric
        )
        return e_sr

    def _forward_pbc_reciprocal(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        device = positions.device

        box_inv = torch.linalg.inv(box)
        if self.spacing is not None:
            self.kmesh = torch.ceil(box.diagonal() / self.spacing).to(torch.long)
        else:
            self.kmesh = torch.ceil(
                2 * self.kappa * box.diagonal() / (3.0 * self.ethresh ** (1.0 / 5.0))
            ).to(torch.long)

        # for electrostatic, exclude gamma point
        gamma_flag = False
        coeff_k_func = _coeff_k_1

        # mapping charges onto mesh
        meshed_charges = spread_charges(
            positions,
            box_inv,
            charges,
            self.kmesh,
            self.pme_shifts,
            self.pme_order,
        )
        kpts_int = setup_kpts_integer(self.kmesh)
        kpts = setup_kpts(box_inv, kpts_int)
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
                    2 * torch.pi * m * kpts_int[None] / self.kmesh.reshape(1, 1, 3)
                ),
                dim=0,
            ),
            dim=1,
        )
        volume = torch.linalg.det(box)
        S_k = torch.fft.fftn(meshed_charges).flatten()
        if not gamma_flag:
            coeff_k = coeff_k_func(kpts[3, 1:], self.kappa, volume)
            E_k = coeff_k * ((S_k[1:].real ** 2 + S_k[1:].imag ** 2) / theta_k[1:] ** 2)
            return torch.sum(E_k) * self.const_lib.dielectric
        else:
            coeff_k = coeff_k_func(kpts[3, :], self.kappa, volume)
            E_k = coeff_k * ((S_k.real**2 + S_k.imag**2) / theta_k**2)
            return torch.sum(E_k) * self.const_lib.dielectric

    def _forward_pbc_self(
        self,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        -\frac{\alpha}{\sqrt{\pi}} \sum_{i} q_i^2
        """
        if self.kspace_flag:
            coeff = self.kappa / self.const_lib.sqrt_pi
            return -torch.sum(coeff * charges**2) * self.const_lib.dielectric
        else:
            return torch.zeros(1, device=charges.device)

    def _forward_pbc_non_neutral(
        self,
        charges: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        volume = torch.det(box)
        # total charge
        Q_tot = torch.sum(charges)

        coeff = (
            -self.const_lib.pi
            / (2 * volume * self.kappa**2)
            * self.const_lib.dielectric
        )
        e_corr_non = coeff * Q_tot**2
        return e_corr_non

    def _forward_slab_corr(
        self,
        _charges: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        ds: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Slab correction energy (ref: 10.1063/1.3216473)

        E = \frac{2\pi}{V} \varepsilon \left( M_z^2 - Q_{\text{tot}} \sum_i q_i z_i + \frac{Q_{\text{tot}}^2 L_z^2}{12} \right)
        """
        charges = _charges.reshape(-1)
        positions = positions * self.const_lib.length_coeff
        ds = ds * self.const_lib.length_coeff
        box = box * self.const_lib.length_coeff

        volume = torch.det(box)
        pre_corr = 2 * self.const_lib.pi / volume * self.const_lib.dielectric
        # dipole moment in axis direction
        Mz = torch.sum(charges * positions[:, self.slab_axis])
        # total charge
        Q_tot = torch.sum(charges)
        # length of the box in axis direction
        Lz = torch.linalg.norm(box[self.slab_axis])

        e_corr = pre_corr * (
            Mz**2
            - Q_tot * (torch.sum(charges * positions[:, self.slab_axis] ** 2))
            - torch.pow(Q_tot, 2) * torch.pow(Lz, 2) / 12
        )
        return torch.sum(e_corr) / self.const_lib.energy_coeff

    def _forward_obc(
        self,
        charges: torch.Tensor,
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> torch.Tensor:
        qi = charges[pairs[:, 0]]
        qj = charges[pairs[:, 1]]
        ds_inv = safe_inverse(ds)
        E_inter = qi * qj * self.const_lib.dielectric * ds_inv
        coul_energy = torch.sum(E_inter * buffer_scales)
        return coul_energy / self.const_lib.energy_coeff


def setup_ewald_parameters(
    rcut: float,
    box: Union[torch.Tensor, np.ndarray] = None,
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
    return 2 * torch.pi / volume / ksq * torch.exp(-ksq / 4 / kappa**2)
