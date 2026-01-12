# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from torch_admp.utils import TorchConstants


class BaseForceModule(torch.nn.Module, ABC):
    """
    - take positions and box as input, return energy (compatible with openmm-torch)
    - set const_lib as a class attribute for necessary constants
    """

    def __init__(self, units_dict: Optional[Dict] = None, *args, **kwargs) -> None:
        torch.nn.Module.__init__(self)
        self.const_lib = TorchConstants(units_dict)

    @abstractmethod
    def forward(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Advanced PES model

        Parameters
        ----------
        positions : torch.Tensor
            atomic positions
        box : torch.Tensor
            simulation box
        pairs : torch.Tensor
            n_pairs * 2 tensor of pairs
        ds : torch.Tensor
            i-j distance tensor
        buffer_scales : torch.Tensor
            buffer scales for each pair, 1 if i < j else 0
        params : Dict[str, torch.Tensor]
            dictionary of parameters for PES model (e.g., atomic charges)

        Returns
        -------
        energy: torch.Tensor
            energy tensor
        """
