#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
BaseForceModule
"""

import unittest
from typing import Dict, Optional

import numpy as np
import torch

from torch_admp.base_force import BaseForceModule
from torch_admp.env import DEVICE, GLOBAL_PT_FLOAT_PRECISION
from torch_admp.utils import to_torch_tensor

from . import SEED


class ForceModuleTester(BaseForceModule):
    """
    Test implementation of BaseForceModule for unit testing.

    This class provides a minimal implementation of the abstract _forward_impl
    method to enable testing of the BaseForceModule functionality.
    """

    def _forward_impl(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Simple dummy implementation for testing
        return torch.tensor(0.0, device=positions.device, dtype=positions.dtype)


class TestBaseForceModule(unittest.TestCase):
    """
    Test suite for the BaseForceModule class.

    This test suite verifies the functionality of the BaseForceModule abstract class,
    including input tensor standardization, forward method behavior, and initialization.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Set random seed for reproducibility
        self.rng = np.random.default_rng(SEED)
        self.tester = ForceModuleTester()

    def test_standardize_input_tensor_single_system(self):
        """Test standardize_input_tensor with single system inputs."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)  # 10 atoms, 3 coordinates
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)  # 3x3 box
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 20 buffer scales

        # This should not raise an exception
        self.tester.standardize_input_tensor(positions, box, pairs, ds, buffer_scales)

    def test_standardize_input_tensor_batched_system(self):
        """Test standardize_input_tensor with batched system inputs."""
        positions = to_torch_tensor(
            self.rng.random((5, 10, 3))
        ).to(GLOBAL_PT_FLOAT_PRECISION)  # 5 frames, 10 atoms, 3 coordinates
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )  # 5 frames, 3x3 box
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(
            torch.long
        )  # 5 frames, 20 pairs
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)  # 5 frames, 20 distances
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 5 frames, 20 buffer scales

        # This should not raise an exception
        self.tester.standardize_input_tensor(positions, box, pairs, ds, buffer_scales)

    def test_standardize_input_tensor_invalid_positions(self):
        """Test standardize_input_tensor with invalid positions dimensions."""
        positions = to_torch_tensor(
            self.rng.random((10, 2))
        ).to(GLOBAL_PT_FLOAT_PRECISION)  # Should be 3 for coordinates
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box(self):
        """Test standardize_input_tensor with invalid box dimensions."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(
            2, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # Should be 3x3
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs(self):
        """Test standardize_input_tensor with invalid pairs dimensions."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 3))).to(
            torch.long
        )  # Should have shape (n_pairs, 2)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_positions_3d(self):
        """Test standardize_input_tensor with invalid 3D positions dimensions."""
        positions = to_torch_tensor(
            self.rng.random((5, 10, 2))
        ).to(GLOBAL_PT_FLOAT_PRECISION)  # Last dim should be 3 for coordinates
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 213)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_positions_ndim(self):
        """Test standardize_input_tensor with invalid positions ndim."""
        positions = to_torch_tensor(self.rng.random(10)).to(GLOBAL_PT_FLOAT_PRECISION)  # Should be 2D or 3D
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError (line 225)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_3d(self):
        """Test standardize_input_tensor with invalid 3D box dimensions."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 2)
        )  # Wrong shape
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 234)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_frame_mismatch(self):
        """Test standardize_input_tensor with box frame count mismatch."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(3, 1, 1)
        )  # 3 frames vs 5 in positions
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 238)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_2d(self):
        """Test standardize_input_tensor with invalid 2D box dimensions."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(
            2, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # Should be 3x3
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError (line 247)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs_3d(self):
        """Test standardize_input_tensor with invalid 3D pairs dimensions."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 3))).to(
            torch.long
        )  # Last dim should be 2
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 253)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs_frame_mismatch(self):
        """Test standardize_input_tensor with pairs frame count mismatch."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (3, 20, 2))).to(
            torch.long
        )  # 3 frames vs 5 in positions
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 257)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs_ndim(self):
        """Test standardize_input_tensor with invalid pairs ndim."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20,))).to(
            torch.long
        )  # Should be 2D or 3D
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError (line 268)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_frame_mismatch(self):
        """Test standardize_input_tensor with ds frame count mismatch."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(
            self.rng.normal(size=(3, 20))
        ).to(GLOBAL_PT_FLOAT_PRECISION)  # 3 frames vs 5 in positions
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 275)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_pair_mismatch(self):
        """Test standardize_input_tensor with ds pair count mismatch."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 15))).to(GLOBAL_PT_FLOAT_PRECISION)  # 15 pairs vs 20 in pairs
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )

        # This should raise a ValueError (line 279)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_1d(self):
        """Test standardize_input_tensor with invalid 1D ds dimensions."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(15,))).to(GLOBAL_PT_FLOAT_PRECISION)  # 15 pairs vs 20 in pairs
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError (line 285)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_ndim(self):
        """Test standardize_input_tensor with invalid ds ndim."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 20, 1))).to(GLOBAL_PT_FLOAT_PRECISION)  # Should be 1D or 2D
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError (line 290)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_frame_mismatch(self):
        """Test standardize_input_tensor with buffer_scales frame count mismatch."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            3, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 3 frames vs 5 in positions

        # This should raise a ValueError (line 296)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_pair_mismatch(self):
        """Test standardize_input_tensor with buffer_scales pair count mismatch."""
        positions = to_torch_tensor(self.rng.random((5, 10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 15, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 15 pairs vs 20 in pairs

        # This should raise a ValueError (line 300)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_1d(self):
        """Test standardize_input_tensor with invalid 1D buffer_scales dimensions."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            15, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 15 pairs vs 20 in pairs

        # This should raise a ValueError (line 306)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_ndim(self):
        """Test standardize_input_tensor with invalid buffer_scales ndim."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(
            5, 20, 1, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # Should be 1D or 2D

        # This should raise a ValueError (line 311)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_ndim(self):
        """Test standardize_input_tensor with invalid box dimensions."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)
        box = to_torch_tensor(
            self.rng.random((3, 3, 3, 3))
        ).to(GLOBAL_PT_FLOAT_PRECISION)  # 4D tensor, should be 2D or 3D
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(torch.long)
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)
        buffer_scales = torch.ones(20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)

        # This should raise a ValueError (line 247)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_forward_single_system(self):
        """Test forward method with single system inputs."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)  # 10 atoms, 3 coordinates
        box = torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)  # 3x3 box
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(
            torch.long
        )  # 20 pairs
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)  # 20 distances
        buffer_scales = torch.ones(
            20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 20 buffer scales
        params = {"charges": to_torch_tensor(self.rng.normal(size=(10,))).to(GLOBAL_PT_FLOAT_PRECISION)}

        # This should not raise an exception
        result = self.tester.forward(positions, box, pairs, ds, buffer_scales, params)
        self.assertEqual(
            result, torch.tensor(0.0, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        )

    def test_forward_batched_system(self):
        """Test forward method with batched system inputs."""
        positions = to_torch_tensor(
            self.rng.random((5, 10, 3))
        ).to(GLOBAL_PT_FLOAT_PRECISION)  # 5 frames, 10 atoms, 3 coordinates
        box = (
            torch.eye(3, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
            .unsqueeze(0)
            .repeat(5, 1, 1)
        )  # 5 frames, 3x3 box
        pairs = to_torch_tensor(self.rng.integers(0, 10, (5, 20, 2))).to(
            torch.long
        )  # 5 frames, 20 pairs
        ds = to_torch_tensor(self.rng.normal(size=(5, 20))).to(GLOBAL_PT_FLOAT_PRECISION)  # 5 frames, 20 distances
        buffer_scales = torch.ones(
            5, 20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 5 frames, 20 buffer scales
        params = {"charges": to_torch_tensor(self.rng.normal(size=(5, 10))).to(GLOBAL_PT_FLOAT_PRECISION)}

        # This should not raise an exception
        result = self.tester.forward(positions, box, pairs, ds, buffer_scales, params)
        self.assertEqual(
            result, torch.tensor(0.0, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        )

    def test_forward_with_none_box(self):
        """Test forward method with None box."""
        positions = to_torch_tensor(self.rng.random((10, 3))).to(GLOBAL_PT_FLOAT_PRECISION)  # 10 atoms, 3 coordinates
        box = None  # No box
        pairs = to_torch_tensor(self.rng.integers(0, 10, (20, 2))).to(
            torch.long
        )  # 20 pairs
        ds = to_torch_tensor(self.rng.normal(size=(20,))).to(GLOBAL_PT_FLOAT_PRECISION)  # 20 distances
        buffer_scales = torch.ones(
            20, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION
        )  # 20 buffer scales
        params = {"charges": to_torch_tensor(self.rng.normal(size=(10,))).to(GLOBAL_PT_FLOAT_PRECISION)}

        # This should not raise an exception
        result = self.tester.forward(positions, box, pairs, ds, buffer_scales, params)
        self.assertEqual(
            result, torch.tensor(0.0, device=DEVICE, dtype=GLOBAL_PT_FLOAT_PRECISION)
        )

    def test_initialization_with_custom_units(self):
        """Test initialization with custom units_dict."""
        units_dict = {"length": "nm"}
        tester_with_units = ForceModuleTester(units_dict=units_dict)
        # tester_with_units.const_lib.length_coeff: factor from nm to ang
        self.assertEqual(
            tester_with_units.const_lib.length_coeff,
            10.0,
        )

    def test_initialization_without_units(self):
        """Test initialization without units_dict."""
        tester_without_units = ForceModuleTester()

        # Check that default units are used
        self.assertIsNotNone(tester_without_units.const_lib.length_coeff)
