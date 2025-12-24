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


class JaxTestData:
    def __init__(self):
        self.rcut = 5.0
        self.l_box = 20.0
        self.n_atoms = 100

        charges = np.random.uniform(-1.0, 1.0, (self.n_atoms))
        self.charges = charges - charges.mean()
        self.positions = np.random.rand(self.n_atoms, 3) * self.l_box
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
        self.assertAlmostEqual(ener_pt.item(), ener_jax, places=5)
        # force [eV/A]
        diff = to_numpy_array(force_pt).reshape(-1, 3) - force_jax.reshape(-1, 3)
        self.assertTrue(np.abs(diff).max() < 5e-4)


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
        self.assertAlmostEqual(ener_pt.item(), ener_jax, places=5)


@unittest.skipIf(not DMFF_AVAILABLE, "dmff package not installed")
class TestQEqForceModule(unittest.TestCase):
    """
    self consistent test (matrix inversion vs pgrad)
    """

    def setUp(self) -> None:
        self.rcut = 5.0
        self.l_box = 20.0
        self.ethresh = 1e-5
        self.n_atoms = 100

        self.positions = torch.rand(self.n_atoms, 3, requires_grad=True) * self.l_box
        self.box = torch.tensor(np.diag([self.l_box, self.l_box, self.l_box]))
        charges = np.random.uniform(-1.0, 1.0, (self.n_atoms))
        charges -= charges.mean()
        self.charges = torch.tensor(charges, requires_grad=True)

        self.chi = torch.rand(self.n_atoms)
        self.hardness = torch.zeros(self.n_atoms)
        self.eta = torch.ones(self.n_atoms) * 0.5

        self.constraint_matrix = torch.ones((1, self.n_atoms))
        self.constraint_vals = torch.zeros(1)
        self.coeff_matrix = vector_projection_coeff_matrix(self.constraint_matrix)

        self.module_matinv = QEqForceModule(self.rcut, self.ethresh)
        self.module_pgrad = QEqForceModule(
            self.rcut, self.ethresh, eps=1e-5, max_iter=100
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
            self.assertTrue(np.abs(out_matinv[0].item() - out_pgrad[0].item()) < 1e-5)
            # force [eV/A]
            diff = forces_matinv - forces_pgrad
            self.assertTrue(diff.abs().max().item() < 1e-4)
            # rmse = torch.sqrt(torch.mean((diff) ** 2))
            # print(rmse)
            # print(diff.max(), diff.min())
            # self.assertTrue(
            #     np.allclose(
            #         torch.Tensor.numpy(forces_matinv, force=True).reshape(-1, 3),
            #         torch.Tensor.numpy(forces_pgrad, force=True).reshape(-1, 3),
            #         atol=1e-4,
            #     )
            # )
            # charge [e]
            diff = out_matinv[1] - out_pgrad[1]
            self.assertTrue(diff.abs().max().item() < 5e-4)
            # rmse = torch.sqrt(torch.mean((diff) ** 2))
            # print(rmse)
            # print(diff.max(), diff.min())
            # self.assertTrue(
            #     np.allclose(
            #         torch.Tensor.numpy(out_matinv[1], force=True),
            #         torch.Tensor.numpy(out_pgrad[1], force=True),
            #         atol=1e-3,
            #     )
            # )

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
            self.assertTrue(np.abs(out_matinv[0].item() - out_jit[0].item()) < 1e-5)
            # force [eV/A]
            diff = forces_matinv - forces_jit
            self.assertTrue(diff.abs().max().item() < 1e-4)
            # charge [e]
            diff = out_matinv[1] - out_jit[1]
            self.assertTrue(diff.abs().max().item() < 1e-4)

    def test_hessian(self):
        n_atoms = self.positions.shape[0]
        charges = torch.rand(n_atoms)
        charges -= charges.mean()

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

        hessian = hessian.detach().cpu().numpy()
        charges = charges.detach().cpu().numpy()
        self.assertAlmostEqual(
            0.5 * np.inner(np.matmul(charges, hessian), charges), (e1 + e2).item()
        )


if __name__ == "__main__":
    unittest.main()
