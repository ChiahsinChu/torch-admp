# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch
from scipy import constants

try:
    import jax
    import jax.numpy as jnp
    from dmff.admp.qeq import E_site3, E_sr3

    DMFF_AVAILABLE = True
except ImportError:
    DMFF_AVAILABLE = False

from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import (
    GaussianDampingForceModule,
    QEqForceModule,
    SiteForceModule,
    pgrad_optimize,
)
from torch_admp.utils import (
    calc_grads,
    calc_pgrads,
    to_numpy_array,
    vector_projection_coeff_matrix,
)

from . import SEED

# Generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=SEED)


class JaxTestData:
    def __init__(self):
        self.rcut = 5.0
        self.l_box = 20.0
        self.n_atoms = 100

        charges = rng.uniform(-1.0, 1.0, (self.n_atoms))
        self.charges = charges - charges.mean()
        self.positions = rng.random((self.n_atoms, 3)) * self.l_box
        self.box = np.diag([self.l_box, self.l_box, self.l_box])
        self.chi = np.ones(self.n_atoms)
        self.hardness = np.zeros(self.n_atoms)
        self.eta = np.ones(self.n_atoms) * 0.5

        # kJ/mol to eV
        j2ev = constants.physical_constants["joule-electron volt relationship"][0]
        # kJ/mol to eV/particle
        self.energy_coeff = j2ev * constants.kilo / constants.Avogadro


@unittest.skipIf(not DMFF_AVAILABLE, "dmff package not installed")
class TestGaussianDampingForceModule(unittest.TestCase, JaxTestData):
    def setUp(self) -> None:
        JaxTestData.__init__(self)

        self.module = GaussianDampingForceModule()
        self.jit_module = torch.jit.script(self.module)

    def test_consistent(self):
        positions = torch.tensor(self.positions, requires_grad=True)
        box = torch.tensor(self.box)
        charges = torch.tensor(self.charges)
        eta = torch.tensor(self.eta)

        nblist = TorchNeighborList(cutoff=self.rcut)
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        ener_pt = self.module(
            positions,
            box,
            pairs,
            ds,
            buffer_scales,
            {"charge": charges, "eta": eta},
        )
        force_pt = -calc_grads(ener_pt, positions)

        # in DMFF they use eta as sqrt(2) * Gaussian width
        jax_out = jax.value_and_grad(E_sr3, argnums=0)(
            jnp.array(self.positions),
            jnp.array(self.box),
            jnp.array(to_numpy_array(pairs)),
            jnp.array(self.charges),
            jnp.array(self.eta * np.sqrt(2.0)),
            jnp.array(to_numpy_array(buffer_scales)),
            True,
        )
        ener_jax = jax_out[0] * self.energy_coeff
        force_jax = -jax_out[1] * self.energy_coeff

        # energy [eV]
        np.testing.assert_allclose(
            to_numpy_array(ener_pt),
            ener_jax,
            atol=1e-6,
            rtol=1e-6,
        )
        # force [eV/A]
        np.testing.assert_allclose(
            to_numpy_array(force_pt).reshape(-1, 3),
            force_jax.reshape(-1, 3),
            atol=1e-6,
            rtol=1e-6,
        )


@unittest.skipIf(not DMFF_AVAILABLE, "dmff package not installed")
class TestSiteForceModule(unittest.TestCase, JaxTestData):
    def setUp(self) -> None:
        JaxTestData.__init__(self)

        self.module = SiteForceModule()
        self.jit_module = torch.jit.script(self.module)

    def test_consistent(self):
        positions = torch.tensor(self.positions, requires_grad=True)
        box = torch.tensor(self.box)
        charges = torch.tensor(self.charges)
        chi = torch.tensor(self.chi)
        hardness = torch.tensor(self.hardness)

        nblist = TorchNeighborList(cutoff=self.rcut)
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        ener_pt = self.module(
            positions,
            box,
            pairs,
            ds,
            buffer_scales,
            {"charge": charges, "chi": chi, "hardness": hardness},
        )

        ener_jax = E_site3(
            jnp.array(self.chi),
            jnp.array(self.hardness),
            jnp.array(self.charges),
        )

        # energy [eV]
        np.testing.assert_allclose(
            to_numpy_array(ener_pt),
            ener_jax,
            atol=1e-6,
            rtol=1e-6,
        )


@unittest.skipIf(not DMFF_AVAILABLE, "dmff package not installed")
class TestQEqForceModule(unittest.TestCase):
    """
    self consistent test (matrix inversion vs pgrad)
    """

    def setUp(self) -> None:
        self.rcut = 5.0
        self.l_box = 20.0
        self.ethresh = 1e-6
        self.n_atoms = 100

        positions = rng.random((self.n_atoms, 3)) * self.l_box
        self.positions = torch.tensor(positions, requires_grad=True)
        self.box = torch.tensor(np.diag([self.l_box, self.l_box, self.l_box]))
        charges = rng.uniform(-1.0, 1.0, (self.n_atoms))
        charges -= charges.mean()
        self.charges = torch.tensor(charges, requires_grad=True)

        chi = rng.random((self.n_atoms,))
        self.chi = torch.tensor(chi)
        self.hardness = torch.zeros(self.n_atoms)
        self.eta = torch.ones(self.n_atoms) * 0.5

        self.constraint_matrix = torch.ones((1, self.n_atoms))
        self.constraint_vals = torch.zeros(1)
        self.coeff_matrix = vector_projection_coeff_matrix(self.constraint_matrix)

        self.module_matinv = QEqForceModule(self.rcut, self.ethresh)
        self.module_pgrad = QEqForceModule(
            self.rcut, self.ethresh, eps=1e-6, ls_eps=1e-6, max_iter=100
        )
        self.jit_module = torch.jit.script(self.module_pgrad)

    def test_consistent(self):
        n_atoms = self.positions.shape[0]

        nblist = TorchNeighborList(cutoff=self.rcut)
        pairs = nblist(self.positions, self.box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # energy, q_opt
        out_matinv = self.module_matinv.solve_matrix_inversion(
            self.positions,
            self.box,
            self.chi,
            self.hardness,
            self.eta,
            pairs,
            ds,
            buffer_scales,
            self.constraint_matrix,
            self.constraint_vals,
        )
        forces_matinv = -calc_grads(out_matinv[0], self.positions)

        for method in ["lbfgs", "quadratic"]:
            out_pgrad = self.module_pgrad.solve_pgrad(
                self.charges,
                self.positions,
                self.box,
                self.chi,
                self.hardness,
                self.eta,
                pairs,
                ds,
                buffer_scales,
                self.constraint_matrix,
                self.constraint_vals,
                self.coeff_matrix,
                reinit_q=True,
                method=method,
            )
            forces_pgrad = -calc_grads(out_pgrad[0], self.positions)

            # convergence check
            assert self.module_pgrad.converge_iter >= 0

            pgrad = calc_pgrads(
                out_pgrad[0], out_pgrad[1], self.constraint_matrix, self.coeff_matrix
            )
            assert (pgrad.norm() / n_atoms).item() < self.module_pgrad.eps

            # energy [eV]
            np.testing.assert_allclose(
                to_numpy_array(out_matinv[0]),
                to_numpy_array(out_pgrad[0]),
                atol=1e-6,
                rtol=1e-6,
            )
            # force [eV/A]
            np.testing.assert_allclose(
                to_numpy_array(forces_matinv),
                to_numpy_array(forces_pgrad),
                atol=5e-6,
                rtol=5e-6,
            )
            # charge [e]
            np.testing.assert_allclose(
                to_numpy_array(out_matinv[1]),
                to_numpy_array(out_pgrad[1]),
                atol=5e-6,
                rtol=5e-6,
            )

        for method in ["lbfgs", "quadratic"]:
            out_jit = pgrad_optimize(
                self.jit_module,
                self.charges,
                self.positions,
                self.box,
                self.chi,
                self.hardness,
                self.eta,
                pairs,
                ds,
                buffer_scales,
                self.constraint_matrix,
                self.constraint_vals,
                self.coeff_matrix,
                reinit_q=True,
                method=method,
            )
            forces_jit = -calc_grads(out_jit[0], self.positions)

            # convergence check
            assert self.jit_module.converge_iter >= 0

            pgrad = calc_pgrads(
                out_jit[0], out_jit[1], self.constraint_matrix, self.coeff_matrix
            )
            assert (pgrad.norm() / n_atoms).item() < self.jit_module.eps

            # energy [eV]
            np.testing.assert_allclose(
                to_numpy_array(out_matinv[0]),
                to_numpy_array(out_jit[0]),
                atol=1e-6,
                rtol=1e-6,
            )
            # force [eV/A]
            np.testing.assert_allclose(
                to_numpy_array(forces_matinv),
                to_numpy_array(forces_jit),
                atol=5e-6,
                rtol=5e-6,
            )
            # charge [e]
            np.testing.assert_allclose(
                to_numpy_array(out_matinv[1]),
                to_numpy_array(out_jit[1]),
                atol=5e-6,
                rtol=5e-6,
            )

    def test_hessian(self):
        charges = rng.uniform(-1.0, 1.0, (self.n_atoms))
        charges -= charges.mean()
        charges = torch.tensor(charges)

        params = {
            "charge": charges,
            "eta": self.eta,
        }

        nblist = TorchNeighborList(cutoff=self.rcut)
        pairs = nblist(self.positions, self.box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        hessian = self.module_matinv.calc_hessian(
            self.positions,
            self.box,
            self.chi,
            torch.zeros_like(self.chi),
            self.eta,
            pairs,
            ds,
            buffer_scales,
        )

        e1 = self.module_matinv.submodels["coulomb"](
            self.positions, self.box, pairs, ds, buffer_scales, params
        )
        e2 = self.module_matinv.submodels["damping"](
            self.positions, self.box, pairs, ds, buffer_scales, params
        )

        hessian = to_numpy_array(hessian)
        charges = to_numpy_array(charges)
        np.testing.assert_allclose(
            0.5 * np.inner(np.matmul(charges, hessian), charges),
            to_numpy_array(e1 + e2),
        )
