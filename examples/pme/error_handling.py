# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Error Handling Example for torch-admp

This example demonstrates proper error handling and validation:
1. Handling invalid parameters
2. Validating input configurations
3. Catching and reporting errors appropriately
"""

import numpy as np
import torch

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule, setup_ewald_parameters

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def main():
    """
    Demonstrate error handling and validation.
    """
    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)

    # Test 1: Invalid slab_axis
    print("\n1. Testing invalid slab_axis:")
    try:
        module = CoulombForceModule(rcut=6.0, ethresh=1e-5, slab_corr=True, slab_axis=3)
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError, IndexError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 2: Negative slab_axis
    print("\n2. Testing negative slab_axis:")
    try:
        module = CoulombForceModule(
            rcut=6.0, ethresh=1e-5, slab_corr=True, slab_axis=-1
        )
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError, IndexError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 3: Invalid ethresh
    print("\n3. Testing invalid ethresh:")
    try:
        module = CoulombForceModule(rcut=6.0, ethresh=-1e-5)
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 4: Zero ethresh
    print("\n4. Testing zero ethresh:")
    try:
        module = CoulombForceModule(rcut=6.0, ethresh=0.0)
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 5: Negative rcut
    print("\n5. Testing negative rcut:")
    try:
        module = CoulombForceModule(rcut=-6.0, ethresh=1e-5)
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 6: Zero rcut
    print("\n6. Testing zero rcut:")
    try:
        module = CoulombForceModule(rcut=0.0, ethresh=1e-5)
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 7: Non-orthogonal box with setup_ewald_parameters
    print("\n7. Testing non-orthogonal box with setup_ewald_parameters:")
    try:
        # Create a non-orthogonal box
        non_orthogonal_box = np.array(
            [[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        kappa, kx, ky, kz = setup_ewald_parameters(rcut=6.0, box=non_orthogonal_box)
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 8: Invalid method for setup_ewald_parameters
    print("\n8. Testing invalid method for setup_ewald_parameters:")
    try:
        box = np.diag([10.0, 10.0, 10.0])
        kappa, kx, ky, kz = setup_ewald_parameters(rcut=6.0, box=box, method="invalid")
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 9: Gromacs method without spacing
    print("\n9. Testing gromacs method without spacing:")
    try:
        box = np.diag([10.0, 10.0, 10.0])
        kappa, kx, ky, kz = setup_ewald_parameters(rcut=6.0, box=box, method="gromacs")
        print("   ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"   ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 10: Invalid spacing value
    print("\n10. Testing invalid spacing value:")
    try:
        module = CoulombForceModule(rcut=6.0, ethresh=1e-5, spacing=-1.0)
        print("    ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"    ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 11: Invalid kmesh value
    print("\n11. Testing invalid kmesh value:")
    try:
        module = CoulombForceModule(rcut=6.0, ethresh=1e-5, kmesh=[0, 24, 24])
        print("    ERROR: Should have raised an exception!")
    except (ValueError, AssertionError) as e:
        print(f"    ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 12: Mismatched tensor dimensions
    print("\n12. Testing mismatched tensor dimensions:")
    try:
        # Create valid system
        positions = torch.randn(100, 3, requires_grad=True)
        box = torch.eye(3)
        charges = torch.randn(100)

        # Create neighbor list
        nblist = TorchNeighborList(cutoff=6.0)
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Create module
        module = CoulombForceModule(rcut=6.0, ethresh=1e-5)

        # Try with mismatched charges
        wrong_charges = torch.randn(50)  # Wrong size
        energy = module(
            positions, box, pairs, ds, buffer_scales, {"charge": wrong_charges}
        )
        print("    ERROR: Should have raised an exception!")
    except (RuntimeError, ValueError, AssertionError) as e:
        print(f"    ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 13: Invalid box tensor
    print("\n13. Testing invalid box tensor:")
    try:
        # Create valid system
        positions = torch.randn(100, 3, requires_grad=True)
        wrong_box = torch.randn(3, 2)  # Wrong shape
        charges = torch.randn(100)

        # Create neighbor list
        nblist = TorchNeighborList(cutoff=6.0)
        pairs = nblist(positions, wrong_box)
        print("    ERROR: Should have raised an exception!")
    except (RuntimeError, ValueError, AssertionError) as e:
        print(f"    ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 14: Zero volume box
    print("\n14. Testing zero volume box:")
    try:
        # Create system with zero volume box
        positions = torch.randn(100, 3, requires_grad=True)
        zero_volume_box = torch.diag(
            torch.tensor([10.0, 10.0, 0.0])
        )  # Zero in z dimension
        charges = torch.randn(100)

        # Create neighbor list
        nblist = TorchNeighborList(cutoff=6.0)
        pairs = nblist(positions, zero_volume_box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Create module
        module = CoulombForceModule(rcut=6.0, ethresh=1e-5)

        # Try to calculate energy
        energy = module(
            positions, zero_volume_box, pairs, ds, buffer_scales, {"charge": charges}
        )
        print("    ERROR: Should have raised an exception!")
    except (RuntimeError, ValueError, AssertionError) as e:
        print(f"    ✓ Correctly caught error: {type(e).__name__}: {e}")

    # Test 15: Valid parameter validation
    print("\n15. Testing valid parameter validation:")

    # Valid parameters
    valid_params = [
        {"rcut": 6.0, "ethresh": 1e-5},
        {"rcut": 6.0, "ethresh": 1e-5, "kappa": 0.3},
        {"rcut": 6.0, "ethresh": 1e-5, "spacing": 1.0},
        {"rcut": 6.0, "ethresh": 1e-5, "kmesh": [24, 24, 24]},
        {"rcut": 6.0, "ethresh": 1e-5, "slab_corr": True, "slab_axis": 2},
        {"rcut": 6.0, "ethresh": 1e-5, "kspace": False, "rspace": True},
        {"rcut": 6.0, "ethresh": 1e-5, "kspace": True, "rspace": False},
        {"rcut": 6.0, "ethresh": 1e-5, "sel": [10, 20, 30]},
    ]

    for i, params in enumerate(valid_params):
        try:
            module = CoulombForceModule(**params)
            print(f"    ✓ Valid parameter set {i+1}: {params}")
        except Exception as e:
            print(
                f"    ✗ Unexpected error with valid params {i+1}: {type(e).__name__}: {e}"
            )

    print("\nAll error handling tests completed successfully!")


if __name__ == "__main__":
    main()
