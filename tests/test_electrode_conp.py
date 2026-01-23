# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for polarizable electrode functionality in torch-admp.

This module contains tests to verify the correctness of calculations
with polarizable electrode under constant potential (CONP) conditions
with comparisons against LAMMPS reference data.

2D:
    - boundary: p p f
    - slab correction: True
    - ffield: False
3D:
    - boundary: p p p
    - slab correction: False
    - ffield: True
"""

import unittest
from pathlib import Path

import numpy as np
from ase import io

from torch_admp import env
from torch_admp.electrode import (
    LAMMPSElectrodeConstraint,
    PolarizableElectrode,
    infer,
    setup_from_lammps,
)
from torch_admp.nblist import TorchNeighborList
from torch_admp.utils import to_numpy_array, to_torch_tensor


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
        kappa = 0.5
        slab_factor = 3.0

        self.calculator = PolarizableElectrode(
            rcut=rcut,
            kappa=kappa,
            slab_corr=self.slab_corr,
            eps=1e-6,
            ls_eps=1e-6,
            max_iter=100,
            ethresh=1e-6,
        )

        self.ref_charges = self.atoms.get_initial_charges()
        self.ref_forces = self.atoms.get_forces()

        self.positions = to_torch_tensor(self.atoms.get_positions()).to(
            env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.positions.requires_grad_(True)
        cell = self.atoms.cell.array
        if self.slab_corr:
            cell[2, 2] *= slab_factor
        self.box = to_torch_tensor(cell).to(env.GLOBAL_PT_FLOAT_PRECISION)
        self.charges = to_torch_tensor(self.atoms.get_initial_charges()).to(
            env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.charges.requires_grad_(True)

        self.nblist = TorchNeighborList(cutoff=rcut)
        self.pairs = self.nblist(
            self.positions,
            self.box,
        )
        self.ds = self.nblist.get_ds()
        self.buffer_scales = self.nblist.get_buffer_scales()

        for method in ["lbfgs", "matinv"]:
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
                method=method,
            )

            # force [eV/A]
            np.testing.assert_allclose(
                to_numpy_array(test_output[1]),
                self.ref_forces,
                atol=self.tol,
                rtol=self.tol,
            )


class TestConpSlab2D(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 2D slab."""

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_slab_2d/dump.lammpstrj"
        )

        self.slab_corr = True
        self.tol = 5e-5
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

        self.ref_energy = 9.1593921


class TestConpSlab3D(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 3D slab."""

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_slab_3d/dump.lammpstrj"
        )

        self.slab_corr = False
        self.tol = 5e-5
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

        self.ref_energy = 2.5921899


class TestConpInterface2DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 2D interface system
    at the potential of zero charge (PZC).
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_2d_pzc/dump.lammpstrj"
        )

        self.slab_corr = True
        self.tol = 5e-4
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

        self.ref_energy = -1943.6576


class TestConpInterface3DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 3D interface system
    at the potential of zero charge (PZC).
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_3d_pzc/dump.lammpstrj"
        )

        self.slab_corr = False
        self.tol = 5e-4
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
        self.ref_energy = -1943.6583


class TestConpInterface2DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 2D interface system
    with applied bias potential.
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_2d_bias/dump.lammpstrj"
        )

        self.slab_corr = True
        self.tol = 5e-4
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

        self.ref_energy = -1934.5002


class TestConpInterface3DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant potential simulation for 3D interface system
    with applied bias potential.
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conp_interface_3d_bias/dump.lammpstrj"
        )

        self.slab_corr = False
        self.tol = 5e-4
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

        self.ref_energy = -1941.0678
