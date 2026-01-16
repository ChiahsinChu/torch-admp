# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import openmm as mm
import torch
from ase import io
from openmm import app
from openmm.unit import angstrom
from scipy import constants

from torch_admp.env import DEVICE
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads, to_numpy_array

from . import SEED

# torch.set_default_dtype(torch.float64)

# kJ/mol to eV/particle
energy_coeff = (
    constants.physical_constants["joule-electron volt relationship"][0]
    * constants.kilo
    / constants.Avogadro
)
# kJ/(mol nm) to eV/particle/A
force_coeff = energy_coeff * constants.angstrom / constants.nano
# Set random generators with SEED for reproducibility
np_rng = np.random.default_rng(SEED)
torch_rng = torch.Generator(device=DEVICE).manual_seed(SEED)


class TestOpenMMSimulation:
    def __init__(self) -> None:
        self.rcut = 5.0
        self.l_box = 20.0
        self.ethresh = 5e-6
        self.n_atoms = 100

        self.charges = np_rng.uniform(-1.0, 1.0, (self.n_atoms))
        self.charges -= self.charges.mean()
        self.positions = np_rng.random((self.n_atoms, 3)) * self.l_box

    def setup(self, real_space=True):
        self.system = mm.System()
        self.system.setDefaultPeriodicBoxVectors(
            (self.l_box * angstrom, 0, 0),
            (0, self.l_box * angstrom, 0),
            (0, 0, self.l_box * angstrom),
        )
        # NonbondedForce, Particle Mesh Ewald
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
        nonbonded.setCutoffDistance(self.rcut * angstrom)
        nonbonded.setEwaldErrorTolerance(self.ethresh)
        nonbonded.setIncludeDirectSpace(real_space)

        self.system.addForce(nonbonded)
        # add ions to the system
        for ii in range(self.n_atoms):
            self.system.addParticle(1.0)  # assume the mass is 1 for simplicity
            nonbonded.addParticle(self.charges[ii], 0, 0)

        dummy_integrator = mm.CustomIntegrator(0)
        # platform = mm.Platform.getPlatformByName("CUDA")
        # create simulation
        self.simulation = app.Simulation(
            topology=None,
            system=self.system,
            integrator=dummy_integrator,
            # platform=platform,
        )
        self.simulation.context.setPositions(self.positions * angstrom)

    def run(self):
        """
        return energy [eV], forces [eV/A]
        """
        state = self.simulation.context.getState(getEnergy=True, getForces=True)
        forces = state.getForces(asNumpy=True)
        energy = state.getPotentialEnergy()
        return (
            np.atleast_1d(energy._value)[0] * energy_coeff,
            forces._value.reshape(-1, 3) * force_coeff,
        )


class TestOBCCoulombForceModule(unittest.TestCase):
    """
    Coulomb interaction under open boundary condition
    """

    def setUp(self) -> None:
        atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_obc/system.data"),
            format="lammps-data",
        )
        positions = atoms.get_positions()
        self.box = None
        charges = atoms.get_initial_charges()

        _positions = torch.tensor(positions, requires_grad=True)
        self.charges = torch.tensor(charges)
        self.positions = _positions.unsqueeze(0)

        self.nblist = TorchNeighborList(cutoff=4.0)
        self.pairs = self.nblist(
            self.positions.squeeze(0),
            self.box,
        ).unsqueeze(0)
        self.ds = self.nblist.get_ds().unsqueeze(0)
        self.buffer_scales = self.nblist.get_buffer_scales().unsqueeze(0)

        self.module = CoulombForceModule(rcut=5.0, ethresh=1e-5)
        # test jit-able
        self.jit_module = torch.jit.script(self.module)

    def test_numerical(self):
        ref_energy = np.loadtxt(
            str(Path(__file__).parent / "data/lmp_coul_obc/thermo.out")
        ).reshape(-1)[1]
        ref_atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_obc/dump.lammpstrj")
        )
        ref_forces = ref_atoms.get_forces()

        energy = self.module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        jit_energy = self.jit_module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        forces = -calc_grads(energy, self.positions)
        jit_forces = -calc_grads(jit_energy, self.positions)

        # energy [eV]
        for e in [energy, jit_energy]:
            np.testing.assert_allclose(
                to_numpy_array(e),
                [ref_energy],
                atol=1e-6,
                rtol=1e-6,
            )
        # force [eV/A]
        for f in [forces, jit_forces]:
            np.testing.assert_allclose(
                to_numpy_array(f).reshape(-1, 3),
                ref_forces,
                atol=1e-6,
                rtol=1e-6,
            )


class TestPBCCoulombForceModule(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_system = TestOpenMMSimulation()
        self.ref_system.setup(real_space=True)

        _positions = torch.tensor(self.ref_system.positions, requires_grad=True)
        self.charges = torch.tensor(self.ref_system.charges).unsqueeze(0)
        _box = torch.tensor(
            np.diag(
                [self.ref_system.l_box, self.ref_system.l_box, self.ref_system.l_box]
            )
        )
        self.positions = _positions.unsqueeze(0)
        self.box = _box.unsqueeze(0)

        self.nblist = TorchNeighborList(cutoff=self.ref_system.rcut)
        self.pairs = self.nblist(
            self.positions.squeeze(0),
            self.box.squeeze(0),
        ).unsqueeze(0)
        self.ds = self.nblist.get_ds().unsqueeze(0)
        self.buffer_scales = self.nblist.get_buffer_scales().unsqueeze(0)

        self.module = CoulombForceModule(
            rcut=self.ref_system.rcut,
            ethresh=self.ref_system.ethresh,
        )
        # test jit-able
        self.jit_module = torch.jit.script(self.module)

    def test_numerical(self):
        energy = self.module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        jit_energy = self.jit_module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        forces = -calc_grads(energy, self.positions)
        jit_forces = -calc_grads(jit_energy, self.positions)

        nonbonded = self.ref_system.system.getForce(0)
        # A^-1 to nm^-1 for kappa
        nonbonded.setPMEParameters(
            self.module.kappa * 10.0,
            self.module.kmesh[0, 0].item(),
            self.module.kmesh[0, 1].item(),
            self.module.kmesh[0, 2].item(),
        )
        # simulation = self.ref_system.simulation
        # ewald_params = nonbonded.getPMEParametersInContext(simulation.context)
        # self.kappa = ewald_params[0] / 10.0
        # self.kmesh = tuple(ewald_params[1:])
        ref_energy, ref_forces = self.ref_system.run()

        # energy [eV]
        for e in [energy, jit_energy]:
            np.testing.assert_allclose(
                to_numpy_array(e),
                [ref_energy],
                atol=1e-5,
                rtol=1e-5,
            )
        # force [eV/A]
        for f in [forces, jit_forces]:
            np.testing.assert_allclose(
                to_numpy_array(f).reshape(-1, 3),
                ref_forces,
                atol=1e-5,
                rtol=1e-5,
            )


class TestPBCSlabCorrCoulombForceModule(unittest.TestCase):
    def setUp(self) -> None:
        atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_pbc/system.data"),
            format="lammps-data",
        )
        positions = atoms.get_positions()
        box = atoms.get_cell().array
        charges = atoms.get_initial_charges()

        _positions = torch.tensor(positions, requires_grad=True)
        _box = torch.tensor(box)
        self.charges = torch.tensor(charges).unsqueeze(0)
        self.positions = _positions.unsqueeze(0)
        self.box = _box.unsqueeze(0)

        self.nblist = TorchNeighborList(cutoff=4.0)
        self.pairs = self.nblist(
            self.positions.squeeze(0), self.box.squeeze(0)
        ).unsqueeze(0)
        self.ds = self.nblist.get_ds().unsqueeze(0)
        self.buffer_scales = self.nblist.get_buffer_scales().unsqueeze(0)

    def lammps_ref_data(self):
        e1 = np.loadtxt(
            str(Path(__file__).parent / "data/lmp_coul_pbc/thermo.out")
        ).reshape(-1)[1]
        e2 = np.loadtxt(
            str(Path(__file__).parent / "data/lmp_coul_pbc_slab_corr/thermo.out")
        ).reshape(-1)[1]
        ref_energy = e2 - e1

        atoms = io.read(str(Path(__file__).parent / "data/lmp_coul_pbc/dump.lammpstrj"))
        f1 = atoms.get_forces()
        atoms = io.read(
            str(Path(__file__).parent / "data/lmp_coul_pbc_slab_corr/dump.lammpstrj")
        )
        f2 = atoms.get_forces()
        ref_force = f2 - f1
        return ref_energy, ref_force

    def test_numerical(self):
        # rcut and ethresh are not used in slab correction calculation
        module = CoulombForceModule(
            rcut=self.nblist.cutoff,
            ethresh=1e-3,
            kspace=False,
            rspace=False,
            slab_corr=True,
            slab_axis=2,
        )
        jit_module = torch.jit.script(module)

        energy = module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        jit_energy = jit_module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        forces = -calc_grads(energy, self.positions)
        jit_forces = -calc_grads(jit_energy, self.positions)

        # ref_energy = self.make_ref_data(axis=2)
        ref_energy, ref_forces = self.lammps_ref_data()

        # energy [eV]
        for e in [energy, jit_energy]:
            np.testing.assert_allclose(
                to_numpy_array(e),
                [ref_energy],
                atol=1e-6,
                rtol=1e-6,
            )
        # force [eV/A]
        for f in [forces, jit_forces]:
            np.testing.assert_allclose(
                to_numpy_array(f).reshape(-1, 3),
                ref_forces,
                atol=1e-6,
                rtol=1e-6,
            )


class TestCoulombForceModule(unittest.TestCase):
    """Test cases to improve coverage of CoulombForceModule"""

    def setUp(self) -> None:
        # Setup for getter tests
        self.module = CoulombForceModule(rcut=5.0, ethresh=1e-5, sel=[10, 20])

        # Setup for edge case tests
        self.n_atoms = 10
        self.positions = torch.rand(1, self.n_atoms, 3, generator=torch_rng) * 10.0
        self.box = torch.diag(torch.tensor([10.0, 10.0, 10.0])).unsqueeze(0)
        self.charges = torch.randn(1, self.n_atoms, generator=torch_rng)

        self.nblist = TorchNeighborList(cutoff=4.0)
        self.pairs = self.nblist(
            self.positions.squeeze(0), self.box.squeeze(0)
        ).unsqueeze(0)
        self.ds = self.nblist.get_ds().unsqueeze(0)
        self.buffer_scales = self.nblist.get_buffer_scales().unsqueeze(0)

    def test_get_rcut(self):
        """Test get_rcut method (line 98)"""
        self.assertEqual(self.module.get_rcut(), 5.0)

    def test_get_sel(self):
        """Test get_sel method (line 101)"""
        self.assertEqual(self.module.get_sel(), [10, 20])

    def test_kspace_false(self):
        """Test _forward_pbc_self when kspace_flag is False (line 288)"""
        module = CoulombForceModule(rcut=5.0, ethresh=1e-5, kspace=False)
        module(
            self.positions,
            self.box,
            self.pairs,
            self.ds,
            self.buffer_scales,
            {"charge": self.charges},
        )
        # Verify that self_energy is zero when kspace_flag is False
        self.assertEqual(module.self_energy.item(), 0.0)
        self.assertEqual(module.reciprocal_energy.item(), 0.0)

    def test_setup_ewald_parameters_openmm(self):
        """Test setup_ewald_parameters with openmm method (lines 388-443)"""
        from torch_admp.pme import setup_ewald_parameters

        box = np.diag([10.0, 10.0, 10.0])
        kappa, kx, ky, kz = setup_ewald_parameters(
            rcut=5.0, box=box, threshold=1e-5, method="openmm"
        )

        # Verify that parameters are reasonable
        self.assertGreater(kappa, 0)
        self.assertGreaterEqual(kx, 1)
        self.assertGreaterEqual(ky, 1)
        self.assertGreaterEqual(kz, 1)

    def test_setup_ewald_parameters_gromacs(self):
        """Test setup_ewald_parameters with gromacs method (lines 388-443)"""
        from torch_admp.pme import setup_ewald_parameters

        box = np.diag([10.0, 10.0, 10.0])
        kappa, kx, ky, kz = setup_ewald_parameters(
            rcut=5.0, box=box, threshold=1e-5, spacing=1.0, method="gromacs"
        )

        # Verify that parameters are reasonable
        self.assertGreater(kappa, 0)
        self.assertGreaterEqual(kx, 1)
        self.assertGreaterEqual(ky, 1)
        self.assertGreaterEqual(kz, 1)

    def test_setup_ewald_parameters_no_box(self):
        """Test setup_ewald_parameters with no box (lines 388-443)"""
        from torch_admp.pme import setup_ewald_parameters

        kappa, kx, ky, kz = setup_ewald_parameters(rcut=5.0, box=None)

        # Should return default values
        self.assertEqual(kappa, 0.1)
        self.assertEqual(kx, 1)
        self.assertEqual(ky, 1)
        self.assertEqual(kz, 1)

    def test_setup_ewald_parameters_invalid_method(self):
        """Test setup_ewald_parameters with invalid method (lines 388-443)"""
        from torch_admp.pme import setup_ewald_parameters

        box = np.diag([10.0, 10.0, 10.0])

        with self.assertRaises(ValueError):
            setup_ewald_parameters(rcut=5.0, box=box, threshold=1e-5, method="invalid")

    def test_setup_ewald_parameters_gromacs_no_spacing(self):
        """Test setup_ewald_parameters with gromacs method but no spacing (lines 388-443)"""
        from torch_admp.pme import setup_ewald_parameters

        box = np.diag([10.0, 10.0, 10.0])

        with self.assertRaises(AssertionError):
            setup_ewald_parameters(rcut=5.0, box=box, threshold=1e-5, method="gromacs")

    def test_setup_ewald_parameters_non_orthogonal_box(self):
        """Test setup_ewald_parameters with non-orthogonal box (lines 388-443)"""
        from torch_admp.pme import setup_ewald_parameters

        # Create a non-orthogonal box
        box = np.array([[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        with self.assertRaises(AssertionError):
            setup_ewald_parameters(rcut=5.0, box=box, threshold=1e-5, method="openmm")
