# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Advanced PME Parameters Example for torch-admp

This example demonstrates advanced PME parameters:
1. Custom kappa (inverse screening length)
2. Custom grid spacing
3. Custom kmesh (grid points)
4. Slab correction for 2D systems
"""

import numpy as np
import torch

from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule

# Set default device and precision
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def main():
    """
    Demonstrate advanced PME parameters including slab correction,
    custom kappa, spacing, and kmesh.
    """
    print("\n" + "="*60)
    print("ADVANCED PARAMETERS EXAMPLE")
    print("="*60)
    
    # System parameters
    rcut = 6.0
    n_atoms = 100
    ethresh = 1e-5
    l_box = 20.0
    
    # Generate system (slab geometry)
    np.random.seed(42)
    positions = np.random.rand(n_atoms, 3) * l_box
    # Create a slab by confining atoms to z < l_box/3
    positions[:, 2] = np.random.rand(n_atoms) * l_box / 3
    box = np.diag([l_box, l_box, l_box])
    charges = np.random.uniform(-1.0, 1.0, (n_atoms))
    charges -= charges.mean()
    
    # Convert to PyTorch tensors
    positions = torch.tensor(positions, requires_grad=True)
    box = torch.tensor(box, requires_grad=False)
    charges = torch.tensor(charges, requires_grad=False)
    
    # Create neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()
    
    # Example 1: Custom kappa (inverse screening length)
    print("\n1. Custom kappa parameter:")
    custom_kappa = 0.3  # Å^-1
    module_kappa = CoulombForceModule(rcut=rcut, ethresh=ethresh, kappa=custom_kappa)
    energy_kappa = module_kappa(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Custom kappa: {custom_kappa} Å^-1")
    print(f"   Computed kappa: {module_kappa.kappa:.6f} Å^-1")
    print(f"   Energy: {energy_kappa.item():.6f} eV")
    
    # Example 2: Custom spacing
    print("\n2. Custom grid spacing:")
    custom_spacing = 1.0  # Å
    module_spacing = CoulombForceModule(rcut=rcut, ethresh=ethresh, spacing=custom_spacing)
    energy_spacing = module_spacing(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Custom spacing: {custom_spacing} Å")
    print(f"   Actual kmesh used: {module_spacing._kmesh.tolist()}")
    print(f"   Energy: {energy_spacing.item():.6f} eV")
    
    # Example 3: Custom kmesh
    print("\n3. Custom kmesh:")
    custom_kmesh = [24, 24, 24]
    module_kmesh = CoulombForceModule(rcut=rcut, ethresh=ethresh, kmesh=custom_kmesh)
    energy_kmesh = module_kmesh(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Custom kmesh: {custom_kmesh}")
    print(f"   Actual kmesh used: {module_kmesh._kmesh.tolist()}")
    print(f"   Energy: {energy_kmesh.item():.6f} eV")
    
    # Example 4: Slab correction
    print("\n4. Slab correction:")
    module_slab = CoulombForceModule(
        rcut=rcut, 
        ethresh=ethresh, 
        slab_corr=True, 
        slab_axis=2  # Apply correction along z-axis
    )
    energy_slab = module_slab(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Slab correction axis: z")
    print(f"   Energy with slab correction: {energy_slab.item():.6f} eV")
    print(f"   Slab correction energy: {module_slab.slab_corr_energy.item():.6f} eV")
    
    # Example 5: Different slab axes
    print("\n5. Testing different slab axes:")
    for axis in [0, 1, 2]:
        module_axis = CoulombForceModule(
            rcut=rcut, 
            ethresh=ethresh, 
            slab_corr=True, 
            slab_axis=axis
        )
        energy_axis = module_axis(positions, box, pairs, ds, buffer_scales, {"charge": charges})
        axis_name = ['x', 'y', 'z'][axis]
        print(f"   {axis_name}-axis slab correction: {energy_axis.item():.6f} eV")
        print(f"   {axis_name}-axis correction term: {module_axis.slab_corr_energy.item():.6f} eV")
    
    # Example 6: Disable kspace or rspace
    print("\n6. Disabling kspace or rspace:")
    
    # Only real space
    module_rspace_only = CoulombForceModule(rcut=rcut, ethresh=ethresh, kspace=False, rspace=True)
    energy_rspace_only = module_rspace_only(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Real space only: {energy_rspace_only.item():.6f} eV")
    print(f"   Real energy: {module_rspace_only.real_energy.item():.6f} eV")
    print(f"   Reciprocal energy: {module_rspace_only.reciprocal_energy.item():.6f} eV")
    
    # Only reciprocal space
    module_kspace_only = CoulombForceModule(rcut=rcut, ethresh=ethresh, kspace=True, rspace=False)
    energy_kspace_only = module_kspace_only(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Reciprocal space only: {energy_kspace_only.item():.6f} eV")
    print(f"   Real energy: {module_kspace_only.real_energy.item():.6f} eV")
    print(f"   Reciprocal energy: {module_kspace_only.reciprocal_energy.item():.6f} eV")
    
    # Example 7: Combined parameters
    print("\n7. Combined advanced parameters:")
    module_combined = CoulombForceModule(
        rcut=rcut,
        ethresh=ethresh,
        kappa=0.25,
        spacing=1.2,
        slab_corr=True,
        slab_axis=2
    )
    energy_combined = module_combined(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    print(f"   Combined parameters energy: {energy_combined.item():.6f} eV")
    print(f"   Used kappa: {module_combined.kappa:.6f} Å^-1")
    print(f"   Used kmesh: {module_combined._kmesh.tolist()}")
    print(f"   Slab correction: {module_combined.slab_corr_energy.item():.6f} eV")
    
    return module_slab, energy_slab


if __name__ == "__main__":
    main()