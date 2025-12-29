# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for automatic differentiation functionality in torch-admp.

This module contains tests to verify the correctness of gradient calculations
using finite difference methods compared to automatic differentiation.
"""

import unittest

import numpy as np
import torch

from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

from .deepmd.pt.utils import env
from .deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor

SEED = 1

dtype = torch.float64


def finite_difference(f, x, delta=1e-6):
    """Calculate finite difference gradient of a function.

    This function computes the gradient of function f at point x using
    the central finite difference method.

    Parameters
    ----------
    f : callable
        Function to differentiate
    x : np.ndarray
        Input tensor at which to compute gradient
    delta : float, optional
        Step size for finite difference approximation, by default 1e-6

    Returns
    -------
    np.ndarray
        Gradient computed using finite differences
    """
    in_shape = x.shape
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1p = f(x + diff)
        y1n = f(x - diff)
        res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
    return res


def data_generator(natoms: int, l_box: float):
    """Generate test data for autodiff testing.

    Creates random atomic positions and box vectors for testing
    gradient calculations.

    Parameters
    ----------
    natoms : int
        Number of atoms to generate
    l_box : float
        Length of the simulation box

    Returns
    -------
    tuple
        (generator, input_dict) containing the random generator
        and input dictionary with positions, box, and placeholder data
    """
    generator = torch.Generator(device=env.DEVICE).manual_seed(SEED)
    box = torch.rand([3, 3], device=env.DEVICE, dtype=dtype, generator=generator)
    box = (box + box.T) + l_box * torch.eye(3)
    positions = torch.rand(
        [natoms, 3], device=env.DEVICE, dtype=dtype, generator=generator
    )
    positions = torch.matmul(positions, box)

    positions.requires_grad_(True)
    box.requires_grad_(True)

    placeholder_pairs = torch.ones(1, 2).to(torch.long)
    placeholder_ds = torch.ones(1)
    placeholder_buffer_scales = torch.zeros(1)

    input_dict = {
        "positions": positions,
        "box": box,
        "pairs": placeholder_pairs,
        "ds": placeholder_ds,
        "buffer_scales": placeholder_buffer_scales,
    }

    return generator, input_dict


class FDTest:
    """Test class for finite difference gradient verification.

    This class provides a generic test method to compare gradients
    computed using finite differences with those from automatic differentiation.
    """

    def test(
        self,
    ) -> None:
        """Test gradient accuracy using finite difference comparison.

        Compares gradients computed by automatic differentiation with
        those computed using finite difference approximation for both
        positions and box parameters.
        """
        places = 5
        delta = 1e-5

        for test_kw in ["positions", "box"]:

            def ff(v):
                input_dict = self.input_dict.copy()
                input_dict[test_kw] = to_torch_tensor(v)
                return to_numpy_array(self.calculator(**input_dict))

            fd_grad = finite_difference(
                ff, to_numpy_array(self.input_dict[test_kw]), delta=delta
            )
            rf_grad = calc_grads(
                self.calculator(**self.input_dict), self.input_dict[test_kw]
            )
            rf_grad = to_numpy_array(rf_grad)
            np.testing.assert_almost_equal(fd_grad, rf_grad, decimal=places)


class TestGradCoulombForceModule(unittest.TestCase, FDTest):
    """Test gradient calculations for CoulombForceModule.

    This test class verifies that gradients computed by the CoulombForceModule
    match finite difference approximations, ensuring correct implementation
    of automatic differentiation for electrostatic interactions.
    """

    def setUp(self):
        """Set up test data and CoulombForceModule for gradient testing.

        Initializes random atomic positions, box vectors, and charges,
        and creates a CoulombForceModule instance for testing.
        """
        natoms = 100
        l_box = 10.0

        generator, input_dict = data_generator(natoms, l_box)
        self.input_dict = input_dict

        rcut = 5.0
        ewald_h = 0.4
        ewald_beta = 0.5

        self.calculator = CoulombForceModule(
            rcut=rcut,
            rspace=False,
            kappa=ewald_beta,
            spacing=ewald_h,
        )
        charges = torch.rand(
            [natoms], device=env.DEVICE, dtype=dtype, generator=generator
        )
        self.input_dict["params"] = {"charge": charges}
