# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for polarizable electrode functionality in torch-admp.

This module contains tests to verify the correctness of calculations
with polarizable electrode under constant charge (CONQ) conditions
with comparisons against LAMMPS reference data.

2D:
    - boundary: p p f
    - slab correction: True
    - ffield: False
3D:
    - boundary: p p p
    - slab correction: False
    - ffield: True/False
"""

import unittest
from pathlib import Path

import numpy as np
from ase import io

from torch_admp.electrode import LAMMPSElectrodeConstraint, setup_from_lammps

from .test_electrode_conp import LAMMPSReferenceDataTest


class TestConqInterface2DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 2D interface system
    at the potential of zero charge (PZC).
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_2d_pzc/dump.lammpstrj"
        )

        self.slab_corr = True
        self.tol = 5e-4
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

        self.ref_energy = -1943.6576


class TestConqInterface3DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 3D interface system
    at the potential of zero charge (PZC).
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_3d_pzc/dump.lammpstrj"
        )

        self.slab_corr = False
        self.tol = 5e-4
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

        self.ref_energy = -1943.6583


class TestConqInterface2DEDL(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 2D interface system
    with electrical double layer (EDL) formation.
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_2d_edl/dump.lammpstrj"
        )

        self.slab_corr = True
        self.tol = 5e-4
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

        self.ref_energy = -1114.9378


class TestConqInterface3DEDL(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 3D interface system
    with electrical double layer (EDL) formation.
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_3d_edl/dump.lammpstrj"
        )

        self.slab_corr = False
        self.tol = 5e-4
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

        self.ref_energy = -1114.9377


class TestConqInterface2DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 2D interface system
    with applied bias potential.
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_2d_bias/dump.lammpstrj"
        )

        self.slab_corr = True
        self.tol = 5e-4
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

        self.ref_energy = -900.46651


class TestConqInterface3DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
    """Test constant charge simulation for 3D interface system
    with applied bias potential.
    """

    def setUp(self) -> None:
        self.atoms = io.read(
            Path(__file__).parent / "data/lmp_conq_interface_3d_bias/dump.lammpstrj"
        )

        self.slab_corr = False
        self.tol = 5e-4
        # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
        with self.assertRaises(AttributeError) as context:
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

        self.assertIn(
            "ffield with conq has not been implemented yet",
            str(context.exception),
        )

        self.ref_energy = -1648.7002

    def test(self):
        pass
