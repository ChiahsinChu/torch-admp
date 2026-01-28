# SPDX-License-Identifier: LGPL-3.0-or-later
"""
3D PBC vs 2D Slab Correction Example for torch-admp

This example compares 3D periodic boundary conditions with 2D slab correction:
1. Setting up slab geometries
2. Comparing energy calculations
3. Testing different slab axes
"""

import numpy as np
import torch

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def main():
    """
    Compare 3D PBC and 2D slab correction cases.
    """
    print("\n" + "=" * 60)
    print("3D PBC vs 2D SLAB CORRECTION COMPARISON")
    print("=" * 60)

    # System parameters
    rcut = 6.0
    n_atoms = 100
    ethresh = 1e-5
    l_xy = 20.0  # Box dimensions in x and y
    l_z = 40.0  # Larger box dimension in z for slab geometry

    # Generate slab system (atoms confined to z < l_z/3)
    np.random.seed(42)
    positions = np.random.rand(n_atoms, 3) * [l_xy, l_xy, l_z / 3]
    box_3d = np.diag([l_xy, l_xy, l_z])
    box_2d = np.diag([l_xy, l_xy, l_z])
    charges = np.random.uniform(-1.0, 1.0, (n_atoms))
    charges -= charges.mean()

    # Convert to PyTorch tensors
    positions = torch.tensor(positions, requires_grad=True)
    box_3d = torch.tensor(box_3d, requires_grad=False)
    box_2d = torch.tensor(box_2d, requires_grad=False)
    charges = torch.tensor(charges, requires_grad=False)

    # Create neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box_3d)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    print(f"System: {n_atoms} atoms in {l_xy}×{l_xy}×{l_z} Å³ box")
    print(f"Atoms confined to z < {l_z/3:.1f} Å (slab geometry)")
    print(f"Total charge: {charges.sum().item():.6f} e")

    # 3D PBC calculation
    module_3d = CoulombForceModule(rcut=rcut, ethresh=ethresh, slab_corr=False)
    energy_3d = module_3d(
        positions, box_3d, pairs, ds, buffer_scales, {"charge": charges}
    )

    # 2D slab correction calculation (z-axis)
    module_2d = CoulombForceModule(
        rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
    )
    energy_2d = module_2d(
        positions, box_2d, pairs, ds, buffer_scales, {"charge": charges}
    )

    print(f"\nEnergy comparison:")
    print(f"3D PBC energy:           {energy_3d.item():.6f} eV")
    print(f"2D slab correction energy: {energy_2d.item():.6f} eV")
    print(f"Slab correction term:     {module_2d.slab_corr_energy.item():.6f} eV")
    print(f"Difference:               {(energy_2d - energy_3d).item():.6f} eV")

    # Test different slab axes
    print(f"\nTesting different slab axes:")
    for axis in [0, 1, 2]:
        module_axis = CoulombForceModule(
            rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=axis
        )
        energy_axis = module_axis(
            positions, box_2d, pairs, ds, buffer_scales, {"charge": charges}
        )
        axis_name = ["x", "y", "z"][axis]
        print(f"  {axis_name}-axis slab correction: {energy_axis.item():.6f} eV")
        print(
            f"  {axis_name}-axis correction term: {module_axis.slab_corr_energy.item():.6f} eV"
        )

    # Energy component breakdown
    print(f"\nEnergy component breakdown:")
    print(f"3D PBC:")
    print(f"  Real-space energy:     {module_3d.real_energy.item():.6f} eV")
    print(f"  Reciprocal energy:     {module_3d.reciprocal_energy.item():.6f} eV")
    print(f"  Self energy:            {module_3d.self_energy.item():.6f} eV")
    print(f"  Non-neutral correction: {module_3d.non_neutral_energy.item():.6f} eV")

    print(f"2D slab correction (z-axis):")
    print(f"  Real-space energy:     {module_2d.real_energy.item():.6f} eV")
    print(f"  Reciprocal energy:     {module_2d.reciprocal_energy.item():.6f} eV")
    print(f"  Self energy:            {module_2d.self_energy.item():.6f} eV")
    print(f"  Non-neutral correction: {module_2d.non_neutral_energy.item():.6f} eV")
    print(f"  Slab correction:       {module_2d.slab_corr_energy.item():.6f} eV")

    # Test with different slab thicknesses
    print(f"\nTesting with different slab thicknesses:")
    thickness_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    for ratio in thickness_ratios:
        # Generate slab with different thickness
        positions_test = np.random.rand(n_atoms, 3) * [l_xy, l_xy, l_z * ratio]
        positions_test = torch.tensor(positions_test, requires_grad=True)
        charges_test = torch.tensor(charges, requires_grad=False)

        # Create neighbor list
        pairs_test = nblist(positions_test, box_2d)
        ds_test = nblist.get_ds()
        buffer_scales_test = nblist.get_buffer_scales()

        # Calculate energy with slab correction
        module_test = CoulombForceModule(
            rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
        )
        energy_test = module_test(
            positions_test,
            box_2d,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )

        print(
            f"  Thickness ratio {ratio:.1f} (z < {l_z*ratio:.1f} Å): {energy_test.item():.6f} eV"
        )
        print(f"    Slab correction term: {module_test.slab_corr_energy.item():.6f} eV")

    # Test with different system sizes
    print(f"\nTesting with different system sizes:")
    system_sizes = [50, 100, 200]

    for size in system_sizes:
        # Generate slab system
        positions_test = np.random.rand(size, 3) * [l_xy, l_xy, l_z / 3]
        positions_test = torch.tensor(positions_test, requires_grad=True)
        charges_test = np.random.uniform(-1.0, 1.0, (size))
        charges_test -= charges_test.mean()
        charges_test = torch.tensor(charges_test, requires_grad=False)

        # Create neighbor list
        pairs_test = nblist(positions_test, box_2d)
        ds_test = nblist.get_ds()
        buffer_scales_test = nblist.get_buffer_scales()

        # 3D PBC
        module_3d_test = CoulombForceModule(rcut=rcut, ethresh=ethresh, slab_corr=False)
        energy_3d_test = module_3d_test(
            positions_test,
            box_2d,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )

        # 2D slab correction
        module_2d_test = CoulombForceModule(
            rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
        )
        energy_2d_test = module_2d_test(
            positions_test,
            box_2d,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )

        print(f"  {size:3d} atoms:")
        print(f"    3D PBC:     {energy_3d_test.item():.6f} eV")
        print(f"    2D slab:     {energy_2d_test.item():.6f} eV")
        print(f"    Difference:   {(energy_2d_test - energy_3d_test).item():.6f} eV")
        print(f"    Slab term:   {module_2d_test.slab_corr_energy.item():.6f} eV")

    # Test with different box aspect ratios
    print(f"\nTesting with different box aspect ratios:")
    aspect_ratios = [
        (1.0, 1.0, 2.0),  # Standard slab
        (1.0, 1.0, 5.0),  # Tall slab
        (2.0, 2.0, 1.0),  # Wide box
        (1.0, 2.0, 3.0),  # Rectangular
    ]

    for i, (lx, ly, lz) in enumerate(aspect_ratios):
        # Scale box dimensions
        box_test = np.diag([l_xy * lx, l_xy * ly, l_z * lz])
        box_test = torch.tensor(box_test, requires_grad=False)

        # Generate slab system (confine to 1/3 of shortest dimension)
        min_dim = min(l_xy * lx, l_xy * ly, l_z * lz)
        positions_test = np.random.rand(n_atoms, 3) * [
            l_xy * lx,
            l_xy * ly,
            min_dim / 3,
        ]
        positions_test = torch.tensor(positions_test, requires_grad=True)
        charges_test = torch.tensor(charges, requires_grad=False)

        # Create neighbor list
        pairs_test = nblist(positions_test, box_test)
        ds_test = nblist.get_ds()
        buffer_scales_test = nblist.get_buffer_scales()

        # 3D PBC
        module_3d_test = CoulombForceModule(rcut=rcut, ethresh=ethresh, slab_corr=False)
        energy_3d_test = module_3d_test(
            positions_test,
            box_test,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )

        # 2D slab correction (z-axis)
        module_2d_test = CoulombForceModule(
            rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
        )
        energy_2d_test = module_2d_test(
            positions_test,
            box_test,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )

        print(f"  Aspect ratio {i+1} ({lx:.1f}×{ly:.1f}×{lz:.1f}):")
        print(f"    3D PBC:     {energy_3d_test.item():.6f} eV")
        print(f"    2D slab:     {energy_2d_test.item():.6f} eV")
        print(f"    Difference:   {(energy_2d_test - energy_3d_test).item():.6f} eV")
        print(f"    Slab term:   {module_2d_test.slab_corr_energy.item():.6f} eV")

    # Force comparison
    print(f"\nForce comparison:")
    forces_3d = -calc_grads(energy_3d, positions)
    forces_2d = -calc_grads(energy_2d, positions)

    print(f"3D PBC forces:")
    print(
        f"  Max force magnitude: {torch.max(torch.norm(forces_3d, dim=1)).item():.6f} eV/Å"
    )
    print(
        f"  Mean force magnitude: {torch.mean(torch.norm(forces_3d, dim=1)).item():.6f} eV/Å"
    )

    print(f"2D slab correction forces:")
    print(
        f"  Max force magnitude: {torch.max(torch.norm(forces_2d, dim=1)).item():.6f} eV/Å"
    )
    print(
        f"  Mean force magnitude: {torch.mean(torch.norm(forces_2d, dim=1)).item():.6f} eV/Å"
    )

    force_diff = torch.max(torch.norm(forces_2d - forces_3d, dim=1))
    print(f"Max force difference: {force_diff.item():.6f} eV/Å")

    return module_3d, module_2d


if __name__ == "__main__":
    main()
