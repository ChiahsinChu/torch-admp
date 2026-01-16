#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
BaseForceModule
"""

import unittest
from typing import Dict, Optional

import torch

from torch_admp.base_force import BaseForceModule
from torch_admp.env import DEVICE

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
        return torch.tensor(0.0)


class TestBaseForceModule(unittest.TestCase):
    """
    Test suite for the BaseForceModule class.

    This test suite verifies the functionality of the BaseForceModule abstract class,
    including input tensor standardization, forward method behavior, and initialization.
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Set random seed for reproducibility
        self.rng = torch.Generator(device=DEVICE).manual_seed(SEED)
        self.tester = ForceModuleTester()

    def test_standardize_input_tensor_single_system(self):
        """Test standardize_input_tensor with single system inputs."""
        positions = torch.randn(10, 3, generator=self.rng)  # 10 atoms, 3 coordinates
        box = torch.eye(3)  # 3x3 box
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)  # 20 pairs
        ds = torch.randn(20, generator=self.rng)  # 20 distances
        buffer_scales = torch.ones(20)  # 20 buffer scales

        # This should not raise an exception
        self.tester.standardize_input_tensor(positions, box, pairs, ds, buffer_scales)

    def test_standardize_input_tensor_batched_system(self):
        """Test standardize_input_tensor with batched system inputs."""
        positions = torch.randn(
            5, 10, 3, generator=self.rng
        )  # 5 frames, 10 atoms, 3 coordinates
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)  # 5 frames, 3x3 box
        pairs = torch.randint(
            0, 10, (5, 20, 2), generator=self.rng
        )  # 5 frames, 20 pairs
        ds = torch.randn(5, 20, generator=self.rng)  # 5 frames, 20 distances
        buffer_scales = torch.ones(5, 20)  # 5 frames, 20 buffer scales

        # This should not raise an exception
        self.tester.standardize_input_tensor(positions, box, pairs, ds, buffer_scales)

    def test_standardize_input_tensor_invalid_positions(self):
        """Test standardize_input_tensor with invalid positions dimensions."""
        positions = torch.randn(
            10, 2, generator=self.rng
        )  # Should be 3 for coordinates
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box(self):
        """Test standardize_input_tensor with invalid box dimensions."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(2)  # Should be 3x3
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs(self):
        """Test standardize_input_tensor with invalid pairs dimensions."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(3)
        pairs = torch.randint(
            0, 10, (20, 3), generator=self.rng
        )  # Should have shape (n_pairs, 2)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_positions_3d(self):
        """Test standardize_input_tensor with invalid 3D positions dimensions."""
        positions = torch.randn(
            5, 10, 2, generator=self.rng
        )  # Last dim should be 3 for coordinates
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 213)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_positions_ndim(self):
        """Test standardize_input_tensor with invalid positions ndim."""
        positions = torch.randn(10, generator=self.rng)  # Should be 2D or 3D
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError (line 225)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_3d(self):
        """Test standardize_input_tensor with invalid 3D box dimensions."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 2)  # Wrong shape
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 234)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_frame_mismatch(self):
        """Test standardize_input_tensor with box frame count mismatch."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)  # 3 frames vs 5 in positions
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 238)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_2d(self):
        """Test standardize_input_tensor with invalid 2D box dimensions."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(2)  # Should be 3x3
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError (line 247)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs_3d(self):
        """Test standardize_input_tensor with invalid 3D pairs dimensions."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(
            0, 10, (5, 20, 3), generator=self.rng
        )  # Last dim should be 2
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 253)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs_frame_mismatch(self):
        """Test standardize_input_tensor with pairs frame count mismatch."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(
            0, 10, (3, 20, 2), generator=self.rng
        )  # 3 frames vs 5 in positions
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 257)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs_ndim(self):
        """Test standardize_input_tensor with invalid pairs ndim."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20,), generator=self.rng)  # Should be 2D or 3D
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError (line 268)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_frame_mismatch(self):
        """Test standardize_input_tensor with ds frame count mismatch."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(3, 20, generator=self.rng)  # 3 frames vs 5 in positions
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 275)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_pair_mismatch(self):
        """Test standardize_input_tensor with ds pair count mismatch."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(5, 15, generator=self.rng)  # 15 pairs vs 20 in pairs
        buffer_scales = torch.ones(5, 20)

        # This should raise a ValueError (line 279)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_1d(self):
        """Test standardize_input_tensor with invalid 1D ds dimensions."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(15, generator=self.rng)  # 15 pairs vs 20 in pairs
        buffer_scales = torch.ones(20)

        # This should raise a ValueError (line 285)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_ds_ndim(self):
        """Test standardize_input_tensor with invalid ds ndim."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(5, 20, 1, generator=self.rng)  # Should be 1D or 2D
        buffer_scales = torch.ones(20)

        # This should raise a ValueError (line 290)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_frame_mismatch(self):
        """Test standardize_input_tensor with buffer_scales frame count mismatch."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(3, 20)  # 3 frames vs 5 in positions

        # This should raise a ValueError (line 296)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_pair_mismatch(self):
        """Test standardize_input_tensor with buffer_scales pair count mismatch."""
        positions = torch.randn(5, 10, 3, generator=self.rng)
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)
        pairs = torch.randint(0, 10, (5, 20, 2), generator=self.rng)
        ds = torch.randn(5, 20, generator=self.rng)
        buffer_scales = torch.ones(5, 15)  # 15 pairs vs 20 in pairs

        # This should raise a ValueError (line 300)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_1d(self):
        """Test standardize_input_tensor with invalid 1D buffer_scales dimensions."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(15)  # 15 pairs vs 20 in pairs

        # This should raise a ValueError (line 306)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_buffer_scales_ndim(self):
        """Test standardize_input_tensor with invalid buffer_scales ndim."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(5, 20, 1)  # Should be 1D or 2D

        # This should raise a ValueError (line 311)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box_ndim(self):
        """Test standardize_input_tensor with invalid box dimensions."""
        positions = torch.randn(10, 3, generator=self.rng)
        box = torch.randn(
            3, 3, 3, 3, generator=self.rng
        )  # 4D tensor, should be 2D or 3D
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)
        ds = torch.randn(20, generator=self.rng)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError (line 247)
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_forward_single_system(self):
        """Test forward method with single system inputs."""
        positions = torch.randn(10, 3, generator=self.rng)  # 10 atoms, 3 coordinates
        box = torch.eye(3)  # 3x3 box
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)  # 20 pairs
        ds = torch.randn(20, generator=self.rng)  # 20 distances
        buffer_scales = torch.ones(20)  # 20 buffer scales
        params = {"charges": torch.randn(10, generator=self.rng)}

        # This should not raise an exception
        result = self.tester.forward(positions, box, pairs, ds, buffer_scales, params)
        self.assertEqual(result, torch.tensor(0.0))

    def test_forward_batched_system(self):
        """Test forward method with batched system inputs."""
        positions = torch.randn(
            5, 10, 3, generator=self.rng
        )  # 5 frames, 10 atoms, 3 coordinates
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)  # 5 frames, 3x3 box
        pairs = torch.randint(
            0, 10, (5, 20, 2), generator=self.rng
        )  # 5 frames, 20 pairs
        ds = torch.randn(5, 20, generator=self.rng)  # 5 frames, 20 distances
        buffer_scales = torch.ones(5, 20)  # 5 frames, 20 buffer scales
        params = {"charges": torch.randn(5, 10, generator=self.rng)}

        # This should not raise an exception
        result = self.tester.forward(positions, box, pairs, ds, buffer_scales, params)
        self.assertEqual(result, torch.tensor(0.0))

    def test_forward_with_none_box(self):
        """Test forward method with None box."""
        positions = torch.randn(10, 3, generator=self.rng)  # 10 atoms, 3 coordinates
        box = None  # No box
        pairs = torch.randint(0, 10, (20, 2), generator=self.rng)  # 20 pairs
        ds = torch.randn(20, generator=self.rng)  # 20 distances
        buffer_scales = torch.ones(20)  # 20 buffer scales
        params = {"charges": torch.randn(10, generator=self.rng)}

        # This should not raise an exception
        result = self.tester.forward(positions, box, pairs, ds, buffer_scales, params)
        self.assertEqual(result, torch.tensor(0.0))

    def test_initialization_with_custom_units(self):
        """Test initialization with custom units_dict."""
        units_dict = {"length": "nm"}
        tester_with_units = ForceModuleTester(units_dict=units_dict)
        # tester_with_units.const_lib.length_coeff: factor from nm to ang
        self.assertEqual(
            tester_with_units.const_lib.length_coeff,
            torch.tensor(10.0).to(tester_with_units.const_lib.length_coeff.device),
        )

    def test_initialization_without_units(self):
        """Test initialization without units_dict."""
        tester_without_units = ForceModuleTester()

        # Check that default units are used
        self.assertIsNotNone(tester_without_units.const_lib.length_coeff)
