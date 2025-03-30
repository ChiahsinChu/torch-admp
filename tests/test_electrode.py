# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import io

from torch_admp.electrode import (
    LAMMPSElectrodeConstraint,
    PolarisableElectrode,
    infer,
    setup_from_lammps,
)
from torch_admp.nblist import TorchNeighborList
from torch_admp.utils import to_numpy_array


class LAMMPSReferenceDataTest:
    def test(self) -> None:
        rcut = 5.0
        ethresh = 1e-6
        kappa = 0.5

        self.calculator = PolarisableElectrode(rcut=rcut, ethresh=ethresh, kappa=kappa)

        self.ref_charges = self.atoms.get_initial_charges()
        self.ref_forces = self.atoms.get_forces()

        self.positions = torch.tensor(self.atoms.get_positions(), requires_grad=True)
        self.box = torch.tensor(self.atoms.cell.array)
        self.charges = torch.tensor(
            self.atoms.get_initial_charges(), requires_grad=True
        )

        nblist = TorchNeighborList(cutoff=rcut)
        self.pairs = nblist(self.positions, self.box)
        self.ds = nblist.get_ds()
        self.buffer_scales = nblist.get_buffer_scales()

        # energy, forces, q_opt
        test_output = infer(
            self.calculator,
            self.positions,
            self.box,
            self.charges,
            self.pairs,
            self.ds,
            self.buffer_scales,
            *self.input_data,
        )

        # force
        # lammps: estimated absolute RMS force accuracy = 6.2850532e-06
        diff = to_numpy_array(test_output[1]) - self.ref_forces
        rmse = np.sqrt(np.mean((diff) ** 2))
        self.assertTrue(rmse < 1e-5)
        # max deviation
        self.assertTrue(
            np.allclose(
                to_numpy_array(test_output[1]),
                self.ref_forces,
                atol=1e-4,
            )
        )


class TestConpSlab3D(LAMMPSReferenceDataTest, unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_slab_3d/dump.lammpstrj"
        )
        self.ref_energy = 2.5921899

        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=20.0,
                    mode="conp",
                    eta=1.6,
                    ffield=True,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=True,
                ),
            ],
        )


class TestConpInterface3DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_3d_pzc/dump.lammpstrj"
        )
        self.ref_energy = -1943.6583

        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=True,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=True,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
