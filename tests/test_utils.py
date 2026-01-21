# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from torch_admp import env, utils

from . import SEED
from .common import run_test_without_packages

# Generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=SEED)


class TestArrayConversion(unittest.TestCase):
    def setUp(self):
        self.ref_np_array = rng.random((3, 4)).astype(env.GLOBAL_NP_FLOAT_PRECISION)
        self.ref_torch_tensor = torch.tensor(
            self.ref_np_array,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )

    def test_to_numpy_array(self):
        self._test_to_numpy_array()
        run_test_without_packages(self._test_to_numpy_array, "deepmd*", utils)

    def _test_to_numpy_array(self):
        np_array = utils.to_numpy_array(self.ref_torch_tensor)
        np.testing.assert_array_equal(np_array, self.ref_np_array)

    def test_to_torch_tensor(self):
        self._test_to_torch_tensor()
        run_test_without_packages(self._test_to_torch_tensor, "deepmd*", utils)

    def _test_to_torch_tensor(self):
        torch_tensor = utils.to_torch_tensor(self.ref_np_array)
        torch.testing.assert_close(torch_tensor, self.ref_torch_tensor)
