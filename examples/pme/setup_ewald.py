# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Setup Ewald Parameters Example for torch-admp

This example demonstrates the setup_ewald_parameters utility function:
1. OpenMM method for parameter calculation
2. Gromacs method for parameter calculation
3. Custom parameter configuration
"""

import numpy as np
import torch

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule, setup_ewald_parameters
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def main():
    """
    Demonstrate setup_ewald_parameters utility function.
    """
    print("\n" + "=" * 60)
    print("SETUP EWALD PARAMETERS EXAMPLE")
    print("=" * 60)

    # Example box
    box = np.diag([10.0, 10.0, 10.0])
    rcut = 4.0
    ethresh = 1e-5

    print(f"Example system:")
    print(f"  Box dimensions: {box.diagonal()} Å")
    print(f"  Cutoff: {rcut} Å")
    print(f"  Energy threshold: {ethresh}")

    # OpenMM method
    print(f"\n1. OpenMM method:")
    kappa_omm, kx_omm, ky_omm, kz_omm = setup_ewald_parameters(
        rcut=rcut, box=box, threshold=ethresh, method="openmm"
    )
    print(f"   kappa: {kappa_omm:.6f} Å^-1")
    print(f"   kmesh: [{kx_omm}, {ky_omm}, {kz_omm}]")

    # Gromacs method
    print(f"\n2. Gromacs method:")
    spacing = 1.0
    kappa_gmx, kx_gmx, ky_gmx, kz_gmx = setup_ewald_parameters(
        rcut=rcut, box=box, threshold=ethresh, spacing=spacing, method="gromacs"
    )
    print(f"   kappa: {kappa_gmx:.6f} Å^-1")
    print(f"   kmesh: [{kx_gmx}, {ky_gmx}, {kz_gmx}]")
    print(f"   spacing: {spacing} Å")

    # No box (should return defaults)
    print(f"\n3. No box (default values):")
    kappa_def, kx_def, ky_def, kz_def = setup_ewald_parameters(rcut=rcut, box=None)
    print(f"   kappa: {kappa_def:.6f} Å^-1")
    print(f"   kmesh: [{kx_def}, {ky_def}, {kz_def}]")

    # Test with different box sizes
    print(f"\n4. Testing with different box sizes:")
    box_sizes = [
        [10.0, 10.0, 10.0],  # Cubic
        [15.0, 15.0, 15.0],  # Larger cubic
        [20.0, 20.0, 10.0],  # Rectangular
        [30.0, 30.0, 5.0],  # Flat slab
    ]

    for i, box_dims in enumerate(box_sizes):
        test_box = np.diag(box_dims)
        kappa_test, kx_test, ky_test, kz_test = setup_ewald_parameters(
            rcut=rcut, box=test_box, threshold=ethresh, method="openmm"
        )
        print(f"   Box {i+1} {box_dims}:")
        print(f"     kappa: {kappa_test:.6f} Å^-1")
        print(f"     kmesh: [{kx_test}, {ky_test}, {kz_test}]")
        print(f"     volume: {np.prod(box_dims):.1f} Å³")

    # Test with different cutoffs
    print(f"\n5. Testing with different cutoffs:")
    cutoffs = [4.0, 6.0, 8.0, 10.0, 12.0]

    for rcut_test in cutoffs:
        kappa_test, kx_test, ky_test, kz_test = setup_ewald_parameters(
            rcut=rcut_test, box=box, threshold=ethresh, method="openmm"
        )
        print(f"   rcut {rcut_test:4.1f} Å:")
        print(f"     kappa: {kappa_test:.6f} Å^-1")
        print(f"     kmesh: [{kx_test}, {ky_test}, {kz_test}]")

    # Test with different thresholds
    print(f"\n6. Testing with different thresholds:")
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    for ethresh_test in thresholds:
        kappa_test, kx_test, ky_test, kz_test = setup_ewald_parameters(
            rcut=rcut, box=box, threshold=ethresh_test, method="openmm"
        )
        print(f"   ethresh {ethresh_test:.1e}:")
        print(f"     kappa: {kappa_test:.6f} Å^-1")
        print(f"     kmesh: [{kx_test}, {ky_test}, {kz_test}]")

    # Test with different spacings (Gromacs method)
    print(f"\n7. Testing with different spacings (Gromacs method):")
    spacings = [0.5, 1.0, 1.5, 2.0]

    for spacing_test in spacings:
        kappa_test, kx_test, ky_test, kz_test = setup_ewald_parameters(
            rcut=rcut,
            box=box,
            threshold=ethresh,
            spacing=spacing_test,
            method="gromacs",
        )
        print(f"   spacing {spacing_test:.1f} Å:")
        print(f"     kappa: {kappa_test:.6f} Å^-1")
        print(f"     kmesh: [{kx_test}, {ky_test}, {kz_test}]")

    # Compare OpenMM and Gromacs methods
    print(f"\n8. Comparing OpenMM and Gromacs methods:")

    # Same target accuracy
    kappa_omm, kx_omm, ky_omm, kz_omm = setup_ewald_parameters(
        rcut=rcut, box=box, threshold=ethresh, method="openmm"
    )
    kappa_gmx, kx_gmx, ky_gmx, kz_gmx = setup_ewald_parameters(
        rcut=rcut, box=box, threshold=ethresh, spacing=1.0, method="gromacs"
    )

    print(f"   OpenMM method:")
    print(f"     kappa: {kappa_omm:.6f} Å^-1")
    print(f"     kmesh: [{kx_omm}, {ky_omm}, {kz_omm}]")
    print(f"     Total grid points: {kx_omm * ky_omm * kz_omm}")

    print(f"   Gromacs method:")
    print(f"     kappa: {kappa_gmx:.6f} Å^-1")
    print(f"     kmesh: [{kx_gmx}, {ky_gmx}, {kz_gmx}]")
    print(f"     Total grid points: {kx_gmx * ky_gmx * kz_gmx}")

    # Test with actual PME calculation
    print(f"\n9. Testing parameters with actual PME calculation:")

    # Generate a simple system
    n_atoms = 50
    np.random.seed(42)
    positions = np.random.rand(n_atoms, 3) * 8.0  # Keep atoms away from box edges
    charges = np.random.uniform(-1.0, 1.0, (n_atoms))
    charges -= charges.mean()

    positions = torch.tensor(positions, requires_grad=True)
    box_tensor = torch.tensor(box, requires_grad=False)
    charges = torch.tensor(charges, requires_grad=False)

    # Create neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box_tensor)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Test with OpenMM parameters
    module_omm = CoulombForceModule(rcut=rcut, ethresh=ethresh, kappa=kappa_omm)
    energy_omm = module_omm(
        positions, box_tensor, pairs, ds, buffer_scales, {"charge": charges}
    )

    # Test with Gromacs parameters
    module_gmx = CoulombForceModule(rcut=rcut, ethresh=ethresh, kappa=kappa_gmx)
    energy_gmx = module_gmx(
        positions, box_tensor, pairs, ds, buffer_scales, {"charge": charges}
    )

    # Test with automatic parameters
    module_auto = CoulombForceModule(rcut=rcut, ethresh=ethresh)
    energy_auto = module_auto(
        positions, box_tensor, pairs, ds, buffer_scales, {"charge": charges}
    )

    print(f"   OpenMM parameters:   {energy_omm.item():.6f} eV")
    print(f"   Gromacs parameters:  {energy_gmx.item():.6f} eV")
    print(f"   Auto parameters:     {energy_auto.item():.6f} eV")
    print(f"   Auto kappa:         {module_auto.kappa:.6f} Å^-1")
    print(f"   Auto kmesh:         {module_auto._kmesh.tolist()}")

    # Test parameter optimization
    print(f"\n10. Parameter optimization for performance:")

    # Test different combinations
    combinations = [
        {"rcut": 6.0, "threshold": 1e-5, "method": "openmm"},
        {"rcut": 6.0, "threshold": 1e-5, "method": "gromacs", "spacing": 1.0},
        {"rcut": 8.0, "threshold": 1e-5, "method": "openmm"},
        {"rcut": 6.0, "threshold": 1e-6, "method": "openmm"},
    ]

    for i, combo in enumerate(combinations):
        kappa_test, kx_test, ky_test, kz_test = setup_ewald_parameters(**combo)
        total_grid_points = kx_test * ky_test * kz_test

        print(f"   Combo {i+1}:")
        print(f"     Parameters: {combo}")
        print(f"     kappa: {kappa_test:.6f} Å^-1")
        print(f"     kmesh: [{kx_test}, {ky_test}, {kz_test}]")
        print(f"     Total grid points: {total_grid_points}")
        print(f"     Grid points per atom: {total_grid_points/n_atoms:.1f}")

    print(f"\nAll setup_ewald_parameters tests completed successfully!")


if __name__ == "__main__":
    main()
