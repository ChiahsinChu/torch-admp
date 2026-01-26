# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Basic PME Example for torch-admp

This example demonstrates fundamental usage of PME for periodic systems:
1. Setting up a system with random positions and charges
2. Creating a neighbor list for efficient pair calculations
3. Using the CoulombForceModule to calculate energy and forces
4. Computing forces using automatic differentiation
"""

import numpy as np
import torch

from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def main():
    """
    Basic PME example with periodic boundary conditions.
    """
    print("\n" + "="*60)
    print("BASIC PME EXAMPLE")
    print("="*60)
    
    # System parameters
    rcut = 6.0  # Cutoff distance in Angstroms
    n_atoms = 100  # Number of atoms
    ethresh = 1e-5  # Ewald precision threshold
    l_box = 20.0  # Box length in Angstroms

    # Generate random system
    np.random.seed(42)  # For reproducibility
    positions = np.random.rand(n_atoms, 3) * l_box
    box = np.diag([l_box, l_box, l_box])
    charges = np.random.uniform(-1.0, 1.0, (n_atoms))
    charges -= charges.mean()  # Make system charge-neutral

    # Convert to PyTorch tensors
    positions = torch.tensor(positions, requires_grad=True)
    box = torch.tensor(box, requires_grad=False)
    charges = torch.tensor(charges, requires_grad=False)

    # Create neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Calculate PME energy and forces
    module = CoulombForceModule(rcut=rcut, ethresh=ethresh)
    energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    forces = -calc_grads(energy, positions)

    print(f"System: {n_atoms} atoms in {l_box}×{l_box}×{l_box} Å³ box")
    print(f"Cutoff: {rcut} Å, Ewald threshold: {ethresh}")
    print(f"Total PME energy: {energy.item():.6f} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Max force magnitude: {torch.max(torch.norm(forces, dim=1)).item():.6f} eV/Å")
    
    # Print some additional information
    print(f"\nAdditional information:")
    print(f"Number of atom pairs: {pairs.shape[0]}")
    print(f"Average distance: {torch.mean(ds).item():.6f} Å")
    print(f"Min distance: {torch.min(ds).item():.6f} Å")
    print(f"Max distance: {torch.max(ds).item():.6f} Å")
    
    return module, energy, forces


if __name__ == "__main__":
    main()