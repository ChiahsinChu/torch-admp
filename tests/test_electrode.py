# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import io

from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule
from torch_admp.utils import calc_grads, to_numpy_array


class TestSlab3D(unittest.TestCase):
    def setUp(self):
        atoms = io.read(Path(__file__).parent / "data/lmp_conp_slab_3d/dump.lammpstrj")
        self.ref_charges = atoms.get_initial_charges()
        self.ref_forces = atoms.get_forces()

        self.rcut = 5.0
        self.calculator = QEqForceModule(rcut=self.rcut, ethresh=1e-6, kappa=0.5)

        self.positions = torch.tensor(atoms.get_positions(), requires_grad=True)
        self.box = torch.tensor(atoms.cell.array)

        self.nblist = TorchNeighborList(cutoff=self.rcut)
        self.pairs = self.nblist(self.positions, self.box)
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()

        # lammps input
        lmp_dv = 20.0
        lmp_eta = 1.6

        self.efield = (
            lmp_dv
            / atoms.get_cell()[self.calculator.slab_axis, self.calculator.slab_axis]
        )
        chi = torch.concat(
            [torch.zeros(len(atoms) // 2), torch.ones(len(atoms) // 2) * lmp_dv]
        )
        self.chi = chi + self.efield * self.positions[:, self.calculator.slab_axis]
        self.hardness = torch.zeros_like(chi)
        # take care of the different meaning of eta in lammps (1/length) and here!
        self.eta = torch.ones_like(chi) * (1 / lmp_eta * np.sqrt(2) / 2)

        self.constraint_matrix = torch.ones((1, len(atoms)))
        self.constraint_vals = torch.tensor([0.0])

    def test(self):
        out = self.calculator.solve_matrix_inversion(
            positions=self.positions,
            box=self.box,
            chi=self.chi,
            hardness=self.hardness,
            eta=self.eta,
            pairs=self.pairs,
            ds=self.ds,
            buffer_scales=self.buffer_scales,
            constraint_matrix=self.constraint_matrix,
            constraint_vals=self.constraint_vals,
        )

        # charge
        self.assertTrue(
            np.allclose(
                to_numpy_array(out[1]),
                self.ref_charges,
                atol=1e-3,
            )
        )

        # force
        # lammps: estimated absolute RMS force accuracy = 6.2850532e-06
        test_forces = -calc_grads(out[0], self.positions)
        diff = to_numpy_array(test_forces) - self.ref_forces
        rmse = np.sqrt(np.mean((diff) ** 2))
        self.assertTrue(rmse < 1e-5)
        # max deviation
        self.assertTrue(
            np.allclose(
                to_numpy_array(test_forces),
                self.ref_forces,
                atol=1e-4,
            )
        )


# class TestSlab2D(unittest.TestCase):
#     def setUp(self):
#         pass


# class TestInterface3D(unittest.TestCase):
#     def setUp(self):
#         pass


# class TestInterface2D(unittest.TestCase):
#     def setUp(self):
#         pass


# class TestLAMMPSRefData(unittest.TestCase):
#     def setUp(self) -> None:
#         self.fnames_2d = glob.glob(
#             str(Path(__file__).parent / "data/lmp_con*_2d*/dump.lammpstrj")
#         )
#         self.fnames_2d.sort()
#         self.fnames_3d = glob.glob(
#             str(Path(__file__).parent / "data/lmp_con*_3d*/dump.lammpstrj")
#         )
#         self.fnames_3d.sort()

#     def test(self):
#         for fname_2d, fname_3d in zip(self.fnames_2d, self.fnames_3d):
#             atoms_2d = io.read(fname_2d)
#             atoms_3d = io.read(fname_3d)

#             charges_2d = atoms_2d.get_initial_charges()
#             charges_3d = atoms_3d.get_initial_charges()
#             diff = charges_2d - charges_3d
#             # rmse
#             rmse = np.sqrt(np.mean(diff ** 2))
#             print(fname_2d, rmse, np.abs(diff).max())
#             # self.assertTrue(np.allclose(charges_2d, charges_3d, atol=1e-4))
#             forces_2d = atoms_2d.get_forces()
#             forces_3d = atoms_3d.get_forces()
#             diff = forces_2d - forces_3d
#             # rmse
#             rmse = np.sqrt(np.mean(diff ** 2))
#             print(fname_2d, rmse, np.abs(diff).max())


if __name__ == "__main__":
    unittest.main()
