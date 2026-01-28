# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Energy Components Example for torch-admp

This example demonstrates how to access individual energy components:
1. Real-space energy
2. Reciprocal-space energy
3. Self energy
4. Non-neutral correction
5. Slab correction energy
"""

import numpy as np
import torch

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def main():
    """
    Demonstrate how to access individual energy components.
    """
    print("\n" + "=" * 60)
    print("ENERGY COMPONENTS EXAMPLE")
    print("=" * 60)

    # System parameters
    rcut = 6.0
    n_atoms = 100
    ethresh = 1e-5
    l_box = 20.0

    # Generate system
    np.random.seed(42)
    positions = np.random.rand(n_atoms, 3) * l_box
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

    # Create PME module
    module = CoulombForceModule(rcut=rcut, ethresh=ethresh)

    # Calculate energy
    energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})

    # Access individual energy components
    real_energy = module.real_energy
    reciprocal_energy = module.reciprocal_energy
    self_energy = module.self_energy
    non_neutral_energy = module.non_neutral_energy

    print("Energy components breakdown:")
    print(f"  Real-space energy:     {real_energy.item():.6f} eV")
    print(f"  Reciprocal energy:     {reciprocal_energy.item():.6f} eV")
    print(f"  Self energy:            {self_energy.item():.6f} eV")
    print(f"  Non-neutral correction: {non_neutral_energy.item():.6f} eV")
    print(f"  Total energy:           {energy.item():.6f} eV")
    print(
        f"  Sum of components:      {(real_energy + reciprocal_energy + self_energy + non_neutral_energy).item():.6f} eV"
    )

    # Verify that sum matches total
    torch.testing.assert_close(
        energy,
        real_energy + reciprocal_energy + self_energy + non_neutral_energy,
        rtol=1e-6,
        atol=1e-8 if energy.dtype == torch.float32 else 1e-10,
    )
    print("  ✓ Verification passed: sum of components equals total energy")

    # Example with slab correction
    print("\nEnergy components with slab correction:")
    module_slab = CoulombForceModule(
        rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
    )
    energy_slab = module_slab(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )

    real_energy_slab = module_slab.real_energy
    reciprocal_energy_slab = module_slab.reciprocal_energy
    self_energy_slab = module_slab.self_energy
    non_neutral_energy_slab = module_slab.non_neutral_energy
    slab_corr_energy_slab = module_slab.slab_corr_energy

    print("  Energy components with slab correction:")
    print(f"    Real-space energy:     {real_energy_slab.item():.6f} eV")
    print(f"    Reciprocal energy:     {reciprocal_energy_slab.item():.6f} eV")
    print(f"    Self energy:            {self_energy_slab.item():.6f} eV")
    print(f"    Non-neutral correction: {non_neutral_energy_slab.item():.6f} eV")
    print(f"    Slab correction:       {slab_corr_energy_slab.item():.6f} eV")
    print(f"    Total energy:           {energy_slab.item():.6f} eV")
    print(
        f"    Sum of components:      {(real_energy_slab + reciprocal_energy_slab + self_energy_slab + non_neutral_energy_slab + slab_corr_energy_slab).item():.6f} eV"
    )

    # Verify that sum matches total
    torch.testing.assert_close(
        energy_slab,
        real_energy_slab
        + reciprocal_energy_slab
        + self_energy_slab
        + non_neutral_energy_slab
        + slab_corr_energy_slab,
        rtol=1e-6,
        atol=1e-8 if energy_slab.dtype == torch.float32 else 1e-10,
    )
    print("    ✓ Verification passed: sum of components equals total energy")

    # Example with different parameters
    print("\nEnergy components with different parameters:")

    # Higher ethresh (less accurate)
    module_high_ethresh = CoulombForceModule(rcut=rcut, ethresh=1e-3)
    energy_high_ethresh = module_high_ethresh(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    print(f"  Higher ethresh (1e-3):")
    print(f"    Total energy: {energy_high_ethresh.item():.6f} eV")
    print(f"    Real-space: {module_high_ethresh.real_energy.item():.6f} eV")
    print(f"    Reciprocal: {module_high_ethresh.reciprocal_energy.item():.6f} eV")
    print(f"    Self energy: {module_high_ethresh.self_energy.item():.6f} eV")
    print(f"    Non-neutral: {module_high_ethresh.non_neutral_energy.item():.6f} eV")
    print(f"    Kappa: {module_high_ethresh.kappa:.6f} Å^-1")

    # Lower ethresh (more accurate)
    module_low_ethresh = CoulombForceModule(rcut=rcut, ethresh=1e-7)
    energy_low_ethresh = module_low_ethresh(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    print(f"  Lower ethresh (1e-7):")
    print(f"    Total energy: {energy_low_ethresh.item():.6f} eV")
    print(f"    Real-space: {module_low_ethresh.real_energy.item():.6f} eV")
    print(f"    Reciprocal: {module_low_ethresh.reciprocal_energy.item():.6f} eV")
    print(f"    Self energy: {module_low_ethresh.self_energy.item():.6f} eV")
    print(f"    Non-neutral: {module_low_ethresh.non_neutral_energy.item():.6f} eV")
    print(f"    Kappa: {module_low_ethresh.kappa:.6f} Å^-1")

    # Custom kappa
    module_custom_kappa = CoulombForceModule(rcut=rcut, ethresh=ethresh, kappa=0.2)
    energy_custom_kappa = module_custom_kappa(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    print(f"  Custom kappa (0.2 Å^-1):")
    print(f"    Total energy: {energy_custom_kappa.item():.6f} eV")
    print(f"    Real-space: {module_custom_kappa.real_energy.item():.6f} eV")
    print(f"    Reciprocal: {module_custom_kappa.reciprocal_energy.item():.6f} eV")
    print(f"    Self energy: {module_custom_kappa.self_energy.item():.6f} eV")
    print(f"    Non-neutral: {module_custom_kappa.non_neutral_energy.item():.6f} eV")
    print(f"    Kappa: {module_custom_kappa.kappa:.6f} Å^-1")

    # Disable kspace or rspace
    print("\nEnergy components with kspace/rspace disabled:")

    # Only real space
    module_rspace_only = CoulombForceModule(
        rcut=rcut, ethresh=ethresh, kspace=False, rspace=True
    )
    energy_rspace_only = module_rspace_only(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    print(f"  Real space only:")
    print(f"    Total energy: {energy_rspace_only.item():.6f} eV")
    print(f"    Real-space: {module_rspace_only.real_energy.item():.6f} eV")
    print(f"    Reciprocal: {module_rspace_only.reciprocal_energy.item():.6f} eV")
    print(f"    Self energy: {module_rspace_only.self_energy.item():.6f} eV")

    # Only reciprocal space
    module_kspace_only = CoulombForceModule(
        rcut=rcut, ethresh=ethresh, kspace=True, rspace=False
    )
    energy_kspace_only = module_kspace_only(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    print(f"  Reciprocal space only:")
    print(f"    Total energy: {energy_kspace_only.item():.6f} eV")
    print(f"    Real-space: {module_kspace_only.real_energy.item():.6f} eV")
    print(f"    Reciprocal: {module_kspace_only.reciprocal_energy.item():.6f} eV")
    print(f"    Self energy: {module_kspace_only.self_energy.item():.6f} eV")

    return module, energy


if __name__ == "__main__":
    main()
