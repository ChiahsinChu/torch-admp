# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from torch_admp.utils import TorchConstants


class BaseForceModule(torch.nn.Module, ABC):
    """
    Abstract base class for force modules in torch-admp.

    This class provides a common interface for force modules that take atomic
    positions and simulation box as input and return energy values. It is designed
    to be compatible with OpenMM-torch and sets up a constants library as a
    class attribute for necessary physical constants.

    Notes
    -----
    All subclasses must implement the forward method to define specific force
    calculations.
    """

    def __init__(self, units_dict: Optional[Dict] = None, *args, **kwargs) -> None:
        """
        Initialize the BaseForceModule.

        Parameters
        ----------
        units_dict : Optional[Dict], default=None
            Dictionary containing unit conversion factors. If None, default units
            will be used.
        *args
            Additional positional arguments passed to parent classes.
        **kwargs
            Additional keyword arguments passed to parent classes.

        Attributes
        ----------
        const_lib : TorchConstants
            Library containing physical constants and unit conversions.
        """
        torch.nn.Module.__init__(self)
        self.const_lib = TorchConstants(units_dict)

    def forward(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the potential energy for the given atomic configuration.

        This method validates input dimensions and then calls the abstract
        _forward_impl method which must be implemented by subclasses.

        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions with shape (natoms, 3). Each row contains the
            x, y, z coordinates of an atom.
        box : Optional[torch.Tensor]
            Simulation box vectors with shape (3, 3). Each row represents a
            box vector. Required for periodic boundary conditions.
        pairs : torch.Tensor
            Tensor of atom pairs with shape (n_pairs, 2). Each row contains
            the indices of two atoms that form a pair.
        ds : torch.Tensor
            Distance tensor with shape (n_pairs,). Contains the distances
            between atom pairs specified in the pairs tensor.
        buffer_scales : torch.Tensor
            Buffer scales for each pair with shape (n_pairs,). Contains values
            of 1 if i < j else 0 for each pair, used for buffer management.
        params : Dict[str, torch.Tensor]
            Dictionary of parameters for the PES model. Common parameters include
            atomic charges, Lennard-Jones parameters, etc.

        Returns
        -------
        torch.Tensor
            Scalar energy tensor representing the total potential energy of the
            system.
        """
        # Validate input dimensions
        self._check_input_dim(positions, box, pairs, ds, buffer_scales)

        # Call the implementation in subclasses
        return self._forward_impl(positions, box, pairs, ds, buffer_scales, params)

    @abstractmethod
    @torch.jit.export
    def _forward_impl(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Implementation of the potential energy calculation.

        This abstract method must be implemented by subclasses to define the
        specific potential energy surface (PES) model.

        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions with shape (natoms, 3). Each row contains the
            x, y, z coordinates of an atom.
        box : Optional[torch.Tensor]
            Simulation box vectors with shape (3, 3). Each row represents a
            box vector. Required for periodic boundary conditions.
        pairs : torch.Tensor
            Tensor of atom pairs with shape (n_pairs, 2). Each row contains
            the indices of two atoms that form a pair.
        ds : torch.Tensor
            Distance tensor with shape (n_pairs,). Contains the distances
            between atom pairs specified in the pairs tensor.
        buffer_scales : torch.Tensor
            Buffer scales for each pair with shape (n_pairs,). Contains values
            of 1 if i < j else 0 for each pair, used for buffer management.
        params : Dict[str, torch.Tensor]
            Dictionary of parameters for the PES model. Common parameters include
            atomic charges, Lennard-Jones parameters, etc.

        Returns
        -------
        torch.Tensor
            Scalar energy tensor representing the total potential energy of the
            system.

        Raises
        ------
        NotImplementedError
            If called directly on the base class. Must be implemented by
            subclasses.
        """

    @torch.jit.export
    def _check_input_dim(
        self,
        positions: torch.Tensor,
        box: Optional[torch.Tensor],
        pairs: torch.Tensor,
        ds: torch.Tensor,
        buffer_scales: torch.Tensor,
    ) -> None:
        """
        Verify the shape of input tensors.

        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions with shape (natoms, 3) or (nframes, natoms, 3)
        box : Optional[torch.Tensor]
            Simulation box vectors with shape (3, 3) or (nframes, 3, 3)
        pairs : torch.Tensor
            Tensor of atom pairs with shape (n_pairs, 2) or (nframes, n_pairs, 2)
        ds : torch.Tensor
            Distance tensor with shape (n_pairs,) or (nframes, n_pairs)
        buffer_scales : torch.Tensor
            Buffer scales with shape (n_pairs,) or (nframes, n_pairs)
        charges : torch.Tensor
            Atomic charges with shape (natoms,) or (nframes, natoms)

        Raises
        ------
        ValueError
            If any tensor has incorrect dimensions
        """
        nframes = 1
        # Check positions dimensions
        if positions.dim() == 3:
            # Batched input: (nframes, natoms, 3)
            if positions.size(2) != 3:
                raise ValueError(
                    f"positions must have shape (nframes, natoms, 3), got {positions.shape}"
                )
            nframes = positions.size(0)
        elif positions.dim() == 2:
            # Single system: (natoms, 3)
            if positions.size(1) != 3:
                raise ValueError(
                    f"positions must have shape (natoms, 3), got {positions.shape}"
                )
        else:
            raise ValueError(
                f"positions must be 2D or 3D tensor, got {positions.dim()}D"
            )

        # Check box dimensions if provided
        if box is not None:
            if box.dim() == 3:
                # Batched input: (nframes, 3, 3)
                if box.shape[1:] != (3, 3):
                    raise ValueError(
                        f"box must have shape (nframes, 3, 3), got {box.shape}"
                    )
                if box.size(0) != nframes:
                    raise ValueError(
                        f"box is expected to have {nframes} frame(s), got {box.size(0)}"
                    )
            elif box.dim() == 2:
                # Single system: (3, 3)
                if box.shape != (3, 3):
                    raise ValueError(f"box must have shape (3, 3), got {box.shape}")
            else:
                raise ValueError(f"box must be 2D or 3D tensor, got {box.dim()}D")

        # Check pairs dimensions
        if pairs.dim() == 3:
            # Batched input: (nframes, n_pairs, 2)
            if pairs.size(2) != 2:
                raise ValueError(
                    f"pairs must have shape (nframes, n_pairs, 2), got {pairs.shape}"
                )
            if pairs.size(0) != nframes:
                raise ValueError(
                    f"pairs is expected to have {nframes} frame(s), got {pairs.size(0)}"
                )
        elif pairs.dim() == 2:
            # Single system: (n_pairs, 2)
            if pairs.size(1) != 2:
                raise ValueError(
                    f"pairs must have shape (n_pairs, 2), got {pairs.shape}"
                )
        else:
            raise ValueError(f"pairs must be 2D or 3D tensor, got {pairs.dim()}D")
        n_pairs = pairs.size(-2)

        # Check ds dimensions
        if ds.dim() == 2:
            # Batched input: (nframes, n_pairs)
            if ds.size(0) != nframes:
                raise ValueError(
                    f"ds is expected to have {nframes} frame(s), got {ds.size(0)}"
                )
            if ds.size(1) != n_pairs:
                raise ValueError(
                    f"ds is expected to have {n_pairs} pairs(s), got {ds.size(1)}"
                )
        elif ds.dim() == 1:
            # Single system: (n_pairs,)
            if ds.size(0) != n_pairs:
                raise ValueError(
                    f"ds is expected to have {n_pairs} pairs(s), got {ds.size(0)}"
                )
        else:
            raise ValueError(f"ds must be 1D or 2D tensor, got {ds.dim()}D")

        # Check buffer_scales dimensions
        if buffer_scales.dim() == 2:
            # Batched input: (nframes, n_pairs)
            if buffer_scales.size(0) != nframes:
                raise ValueError(
                    f"buffer_scales is expected to have {nframes} frame(s), got {buffer_scales.size(0)}"
                )
            if buffer_scales.size(1) != n_pairs:
                raise ValueError(
                    f"buffer_scales is expected to have {n_pairs} pairs(s), got {buffer_scales.size(1)}"
                )
        elif buffer_scales.dim() == 1:
            # Single system: (n_pairs,)
            if buffer_scales.size(0) != n_pairs:
                raise ValueError(
                    f"buffer_scales is expected to have {n_pairs} pairs(s), got {buffer_scales.size(0)}"
                )
        else:
            raise ValueError(
                f"buffer_scales must be 1D or 2D tensor, got {buffer_scales.dim()}D"
            )
