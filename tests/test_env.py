# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import os
import unittest
from unittest.mock import patch

from deepmd import env as dp_env
from deepmd.dpmodel import common
from deepmd.pt.utils import env as pt_env

from torch_admp import env

from .common import run_test_without_packages


class TestClass(unittest.TestCase):
    def setUp(self):
        self.ref_data = {}
        for prec in ["low", "high"]:
            with patch.dict(os.environ, {"DP_INTERFACE_PREC": prec}):
                importlib.reload(dp_env)
                importlib.reload(common)
                importlib.reload(pt_env)
                importlib.reload(env)
                self.ref_data[prec] = {
                    "device": env.DEVICE,
                    "global_pt_float_precision": env.GLOBAL_PT_FLOAT_PRECISION,
                    "global_pt_ener_float_precision": env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                    "global_float_precision": env.GLOBAL_NP_FLOAT_PRECISION,
                    "global_float_ener_precision": env.GLOBAL_ENER_FLOAT_PRECISION,
                    "np_precision_dict": env.NP_PRECISION_DICT,
                    "pt_precision_dict": env.PT_PRECISION_DICT,
                }

        importlib.reload(dp_env)
        importlib.reload(common)
        importlib.reload(pt_env)
        importlib.reload(env)

    def tearDown(self):
        os.environ.pop("DP_INTERFACE_PREC", None)
        importlib.reload(env)

    def _test(self, prec: str):
        os.environ["DP_INTERFACE_PREC"] = prec
        importlib.reload(env)

        self.assertEqual(self.ref_data[prec]["device"], env.DEVICE)
        self.assertEqual(
            self.ref_data[prec]["global_pt_float_precision"],
            env.GLOBAL_PT_FLOAT_PRECISION,
        )
        self.assertEqual(
            self.ref_data[prec]["global_pt_ener_float_precision"],
            env.GLOBAL_PT_ENER_FLOAT_PRECISION,
        )
        self.assertEqual(
            self.ref_data[prec]["global_float_precision"], env.GLOBAL_NP_FLOAT_PRECISION
        )
        self.assertEqual(
            self.ref_data[prec]["global_float_ener_precision"],
            env.GLOBAL_ENER_FLOAT_PRECISION,
        )
        self.assertEqual(
            self.ref_data[prec]["np_precision_dict"], env.NP_PRECISION_DICT
        )
        self.assertEqual(
            self.ref_data[prec]["pt_precision_dict"], env.PT_PRECISION_DICT
        )

    def test(self):
        for prec in ["low", "high"]:
            run_test_without_packages(
                self._test,
                "deepmd*",
                env,
                prec=prec,
            )
        os.environ.pop("DP_INTERFACE_PREC", None)
