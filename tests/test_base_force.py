#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
BaseForceModule
"""

import unittest
from typing import Dict, Optional

import torch

from torch_admp.base_force import BaseForceModule


class ForceModuleTester(BaseForceModule):
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
    Coulomb interaction under open boundary condition
    """

    def setUp(self) -> None:
        self.tester = ForceModuleTester()

    def test_standardize_input_tensor_single_system(self):
        """Test _standardize_input_tensor with single system (non-batched) inputs."""
        positions = torch.randn(10, 3)  # 10 atoms, 3 coordinates
        box = torch.eye(3)  # 3x3 box
        pairs = torch.randint(0, 10, (20, 2))  # 20 pairs
        ds = torch.randn(20)  # 20 distances
        buffer_scales = torch.ones(20)  # 20 buffer scales

        # This should not raise an exception
        self.tester.standardize_input_tensor(positions, box, pairs, ds, buffer_scales)

    def test_standardize_input_tensor_batched_system(self):
        """Test _standardize_input_tensor with batched system inputs."""
        positions = torch.randn(5, 10, 3)  # 5 frames, 10 atoms, 3 coordinates
        box = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)  # 5 frames, 3x3 box
        pairs = torch.randint(0, 10, (5, 20, 2))  # 5 frames, 20 pairs
        ds = torch.randn(5, 20)  # 5 frames, 20 distances
        buffer_scales = torch.ones(5, 20)  # 5 frames, 20 buffer scales

        # This should not raise an exception
        self.tester.standardize_input_tensor(positions, box, pairs, ds, buffer_scales)

    def test_standardize_input_tensor_invalid_positions(self):
        """Test _standardize_input_tensor with invalid positions dimensions."""
        positions = torch.randn(10, 2)  # Should be 3 for coordinates
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 2))
        ds = torch.randn(20)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_box(self):
        """Test _standardize_input_tensor with invalid box dimensions."""
        positions = torch.randn(10, 3)
        box = torch.eye(2)  # Should be 3x3
        pairs = torch.randint(0, 10, (20, 2))
        ds = torch.randn(20)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )

    def test_standardize_input_tensor_invalid_pairs(self):
        """Test _standardize_input_tensor with invalid pairs dimensions."""
        positions = torch.randn(10, 3)
        box = torch.eye(3)
        pairs = torch.randint(0, 10, (20, 3))  # Should have shape (n_pairs, 2)
        ds = torch.randn(20)
        buffer_scales = torch.ones(20)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.tester.standardize_input_tensor(
                positions, box, pairs, ds, buffer_scales
            )
