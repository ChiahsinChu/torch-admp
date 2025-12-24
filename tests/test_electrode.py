# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for electrode functionality in torch-admp.

This module contains tests to verify the correctness of electrode simulations,
including constant potential (CONP) and constant charge (CONQ) simulations,
with comparisons against LAMMPS reference data.
"""

import csv
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import io

from torch_admp.electrode import (
    LAMMPSElectrodeConstraint,
    PolarizableElectrode,
    infer,
    setup_from_lammps,
)
from torch_admp.nblist import vesin_nblist
from torch_admp.utils import to_numpy_array

"""
class LAMMPSReferenceDataTest:
    def test(self) -> None:
        rcut = 5.0
        ethresh = 1e-6
        kappa = 0.5
        slab_factor = 3.0
        self.calculator = PolarizableElectrode(rcut=rcut, ethresh=ethresh, kappa=kappa,slab_corr=self.slab_corr)

        self.ref_charges = self.atoms.get_initial_charges()
        self.ref_forces = self.atoms.get_forces()

        self.positions = torch.tensor(self.atoms.get_positions(), requires_grad=True)

        cell = self.atoms.cell.array
        if self.slab_corr:
            cell[2, 2] *= slab_factor
        self.box = torch.tensor(cell)

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
        self.slab_corr = False
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_slab_3d/dump.lammpstrj"
        )
        self.ref_energy = 2.5921899
        self.slab_corr = False
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
            True,
        )
"""


class LAMMPSReferenceDataTest:
    """Test class for comparing torch-admp electrode results with LAMMPS reference data.

    This class provides a generic test method to compare forces computed by
    torch-admp with reference forces from LAMMPS simulations.
    """

    def test(self) -> None:
        """Test electrode simulation against LAMMPS reference data.

        Compares forces computed by torch-admp with reference forces from LAMMPS,
        ensuring that the implementation produces physically correct results.
        """
        rcut = 5.0
        ethresh = 1e-6
        kappa = 0.5
        slab_factor = 3.0
        self.calculator = PolarizableElectrode(
            rcut=rcut, ethresh=ethresh, kappa=kappa, slab_corr=self.slab_corr
        )

        self.ref_charges = self.atoms.get_initial_charges()
        self.ref_forces = self.atoms.get_forces()

        self.positions = torch.tensor(self.atoms.get_positions(), requires_grad=True)

        cell = self.atoms.cell.array
        if self.slab_corr:
            cell[2, 2] *= slab_factor
        self.box = torch.tensor(cell)
        print(self.box)

        self.charges = torch.tensor(
            self.atoms.get_initial_charges(), requires_grad=True
        )

        self.pairs, self.ds, self.buffer_scales = vesin_nblist(
            self.positions, self.box, rcut=rcut
        )

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
        if rmse >= 1e-5:
            print("RMSE exceeds threshold:", rmse)
            print("Differences in forces:\n", diff)
            self._write_csv(diff, "force_differences.csv")
        self.assertTrue(
            rmse < 1e-5,
            f"RMSE exceeds threshold: {rmse}\nDifferences in forces:\n{diff}",
        )

        # max deviation
        if not np.allclose(to_numpy_array(test_output[1]), self.ref_forces, atol=1e-4):
            print("Forces do not match within tolerance of 1e-4")
            print("Calculated forces:\n", to_numpy_array(test_output[1]))
            print("Reference forces:\n", self.ref_forces)
            self._write_csv(diff, "force_differences.csv")
        self.assertTrue(
            np.allclose(
                to_numpy_array(test_output[1]),
                self.ref_forces,
                atol=1e-4,
            ),
            "Forces do not match within tolerance of 1e-4",
        )

    def _write_csv(self, data, filename):
        """Write the differences to a CSV file.

        Parameters
        ----------
        data : np.ndarray
            Array of differences to write
        filename : str
            Name of the CSV file to write
        """
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Difference"])
            for i, diff in enumerate(data):
                writer.writerow([i, diff])


class TestConpSlab2D(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 2D slab with slab correction.

    Tests constant potential electrode simulation for a 2D slab system
    with slab correction enabled.
    """

    def setUp(self) -> None:
        """Set up test data for 2D slab constant potential simulation.

        Loads atomic positions and sets up electrode constraints for a 2D slab
        system with slab correction.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_slab_2d/dump.lammpstrj"
        )
        self.ref_energy = 9.1593921
        self.slab_corr = True
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=20.0,
                    mode="conp",
                    eta=1.6,
                    ffield=False,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=False,
                ),
            ],
            True,
        )


class TestConpInterface3DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 3D interface at zero charge.

    Tests constant potential electrode simulation for a 3D interface system
    at zero charge condition.
    """

    def setUp(self) -> None:
        """Set up test data for 3D interface constant potential simulation.

        Loads atomic positions and sets up electrode constraints for a 3D interface
        system at zero charge condition.
        """
        self.slab_corr = False
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_3d_pzc/dump.lammpstrj"
        )
        self.ref_energy = -1943.6583
        self.slab_corr = False
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
            True,
        )


class TestConpInterface3DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 3D interface with bias.

    Tests constant potential electrode simulation for a 3D interface system
    with applied bias potential.
    """

    def setUp(self) -> None:
        """Set up test data for 3D interface constant potential simulation.

        Loads atomic positions and sets up electrode constraints for a 3D interface
        system with applied bias potential.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_3d_bias/dump.lammpstrj"
        )
        self.ref_energy = -1941.0678
        self.slab_corr = False
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
            True,
        )


class TestConpInterface2DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 2D interface with bias.

    Tests constant potential electrode simulation for a 2D interface system
    with applied bias potential and slab correction.
    """

    def setUp(self) -> None:
        """Set up test data for 2D interface constant potential simulation.

        Loads atomic positions and sets up electrode constraints for a 2D interface
        system with applied bias potential and slab correction.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_2d_bias/dump.lammpstrj"
        )
        self.ref_energy = -1934.5002
        self.slab_corr = True
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=20.0,
                    mode="conp",
                    eta=1.6,
                    ffield=False,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=False,
                ),
            ],
            True,
        )


class TestConpInterface2DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 2D interface at zero charge.

    Tests constant potential electrode simulation for a 2D interface system
    at zero charge condition with slab correction.
    """

    def setUp(self) -> None:
        """Set up test data for 2D interface constant potential simulation.

        Loads atomic positions and sets up electrode constraints for a 2D interface
        system at zero charge condition with slab correction.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_2d_pzc/dump.lammpstrj"
        )
        self.ref_energy = -1943.6576
        self.slab_corr = True
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=False,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=0.0,
                    mode="conp",
                    eta=1.6,
                    ffield=False,
                ),
            ],
            True,
        )


class TestConqInterface2DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 2D interface at zero charge.

    Tests constant charge electrode simulation for a 2D interface system
    at zero charge condition with slab correction.
    """

    def setUp(self) -> None:
        """Set up test data for 2D interface constant charge simulation.

        Loads atomic positions and sets up electrode constraints for a 2D interface
        system at zero charge condition with slab correction.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_2d_pzc/dump.lammpstrj"
        )
        self.ref_energy = -1943.6576
        self.slab_corr = True
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(216),
                    value=0.0,
                    mode="conq",
                    eta=1.6,
                    ffield=False,
                ),
            ],
        )


class TestConqInterface2DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 2D interface with bias.

    Tests constant charge electrode simulation for a 2D interface system
    with applied bias potential and slab correction.
    """

    def setUp(self) -> None:
        """Set up test data for 2D interface constant charge simulation.

        Loads atomic positions and sets up electrode constraints for a 2D interface
        system with applied bias potential and slab correction.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_2d_bias/dump.lammpstrj"
        )
        self.ref_energy = -900.46651
        self.slab_corr = True
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=-10.0,
                    mode="conq",
                    eta=1.6,
                    ffield=False,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=10.0,
                    mode="conq",
                    eta=1.6,
                    ffield=False,
                ),
            ],
        )


class TestConqInterface2DEDL(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 2D interface with EDL.

    Tests constant charge electrode simulation for a 2D interface system
    with electrical double layer (EDL) formation.
    """

    def setUp(self) -> None:
        """Set up test data for 2D interface constant charge simulation.

        Loads atomic positions and sets up electrode constraints for a 2D interface
        system with electrical double layer (EDL) formation.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_2d_edl/dump.lammpstrj"
        )
        self.ref_energy = -1114.9378
        self.slab_corr = True
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(216),
                    value=-10.0,
                    mode="conq",
                    eta=1.6,
                    ffield=False,
                ),
            ],
        )


class TestConqInterface3DEDL(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 3D interface with EDL.

    Tests constant charge electrode simulation for a 3D interface system
    with electrical double layer (EDL) formation.
    """

    def setUp(self) -> None:
        """Set up test data for 3D interface constant charge simulation.

        Loads atomic positions and sets up electrode constraints for a 3D interface
        system with electrical double layer (EDL) formation.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_3d_edl/dump.lammpstrj"
        )
        self.ref_energy = -1114.9377
        self.slab_corr = False
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(216),
                    value=-10.0,
                    mode="conq",
                    eta=1.6,
                    ffield=False,
                ),
            ],
        )


class TestConqInterface3DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 3D interface with bias.

    Tests constant charge electrode simulation for a 3D interface system
    with applied bias potential.
    """

    def setUp(self) -> None:
        """Set up test data for 3D interface constant charge simulation.

        Loads atomic positions and sets up electrode constraints for a 3D interface
        system with applied bias potential.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_3d_bias/dump.lammpstrj"
        )
        self.ref_energy = -1648.7002
        self.slab_corr = False
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108),
                    value=-10.0,
                    mode="conq",
                    eta=1.6,
                    ffield=True,
                ),
                LAMMPSElectrodeConstraint(
                    indices=np.arange(108, 216),
                    value=10.0,
                    mode="conq",
                    eta=1.6,
                    ffield=True,
                ),
            ],
        )


class TestConqInterface3DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 3D interface at zero charge.

    Tests constant charge electrode simulation for a 3D interface system
    at zero charge condition.
    """

    def setUp(self) -> None:
        """Set up test data for 3D interface constant charge simulation.

        Loads atomic positions and sets up electrode constraints for a 3D interface
        system at zero charge condition.
        """
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_3d_pzc/dump.lammpstrj"
        )
        self.ref_energy = -1943.6583
        self.slab_corr = False
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        self.input_data = setup_from_lammps(
            len(self.atoms),
            [
                LAMMPSElectrodeConstraint(
                    indices=np.arange(216),
                    value=0.0,
                    mode="conq",
                    eta=1.6,
                    ffield=False,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
