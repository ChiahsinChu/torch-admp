# SPDX-License-Identifier: LGPL-3.0-or-later
"""
JIT Compilation Example for torch-admp

This example demonstrates JIT compilation for performance optimization:
1. Creating JIT-compiled modules
2. Performance comparison between regular and JIT execution
3. Saving and loading JIT modules
"""

import time
from pathlib import Path

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
    Demonstrate JIT compilation for performance optimization.
    """
    print("\n" + "=" * 60)
    print("JIT COMPILATION EXAMPLE")
    print("=" * 60)

    # System parameters
    rcut = 6.0
    n_atoms = 200
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

    # Create regular module
    module = CoulombForceModule(rcut=rcut, ethresh=ethresh)

    # Time regular execution
    start_time = time.time()
    energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    for _ in range(9):  # Already computed once above
        energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
    regular_time = time.time() - start_time

    # Create JIT-compiled module
    print("\nCompiling module with TorchScript JIT...")
    jit_module = torch.jit.script(module)

    # Time JIT execution
    start_time = time.time()
    jit_energy = jit_module(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    for _ in range(9):  # Already computed once above
        jit_energy = jit_module(
            positions, box, pairs, ds, buffer_scales, {"charge": charges}
        )
    jit_time = time.time() - start_time

    # Verify results are identical
    assert torch.abs(energy - jit_energy) < 1e-10

    print(f"Regular execution time: {regular_time:.4f} seconds")
    print(f"JIT execution time: {jit_time:.4f} seconds")
    print(f"Speedup: {regular_time/jit_time:.2f}x")
    print(f"Energy: {energy.item():.6f} eV")

    # Save JIT module
    jit_path = Path("pme_module_jit.pt")
    torch.jit.save(jit_module, jit_path)
    print(f"JIT module saved to: {jit_path}")

    # Load JIT module
    print("\nLoading JIT module from file...")
    loaded_jit_module = torch.jit.load(jit_path)

    # Verify loaded module works
    loaded_energy = loaded_jit_module(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    assert torch.abs(energy - loaded_energy) < 1e-10
    print(f"Loaded module energy: {loaded_energy.item():.6f} eV")
    print("✓ Loaded module produces identical results")

    # Test JIT with different configurations
    print("\nTesting JIT with different configurations:")

    # Test with slab correction
    module_slab = CoulombForceModule(
        rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
    )
    jit_module_slab = torch.jit.script(module_slab)

    energy_slab = module_slab(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    jit_energy_slab = jit_module_slab(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )

    assert torch.abs(energy_slab - jit_energy_slab) < 1e-10
    print(
        f"  Slab correction - Regular: {energy_slab.item():.6f} eV, JIT: {jit_energy_slab.item():.6f} eV"
    )

    # Test with custom kappa
    module_kappa = CoulombForceModule(rcut=rcut, ethresh=ethresh, kappa=0.3)
    jit_module_kappa = torch.jit.script(module_kappa)

    energy_kappa = module_kappa(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )
    jit_energy_kappa = jit_module_kappa(
        positions, box, pairs, ds, buffer_scales, {"charge": charges}
    )

    assert torch.abs(energy_kappa - jit_energy_kappa) < 1e-10
    print(
        f"  Custom kappa - Regular: {energy_kappa.item():.6f} eV, JIT: {jit_energy_kappa.item():.6f} eV"
    )

    # Performance test with different system sizes
    print("\nPerformance comparison with different system sizes:")
    system_sizes = [50, 100, 200, 400]

    for size in system_sizes:
        # Generate system
        positions_test = np.random.rand(size, 3) * l_box
        charges_test = np.random.uniform(-1.0, 1.0, (size))
        charges_test -= charges_test.mean()

        positions_test = torch.tensor(positions_test, requires_grad=True)
        charges_test = torch.tensor(charges_test, requires_grad=False)

        # Create neighbor list
        nblist_test = TorchNeighborList(cutoff=rcut)
        pairs_test = nblist_test(positions_test, box)
        ds_test = nblist_test.get_ds()
        buffer_scales_test = nblist_test.get_buffer_scales()

        # Create modules
        module_test = CoulombForceModule(rcut=rcut, ethresh=ethresh)
        jit_module_test = torch.jit.script(module_test)

        # Time regular execution
        start_time = time.time()
        energy_test = module_test(
            positions_test,
            box,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )
        for _ in range(4):  # Already computed once above
            energy_test = module_test(
                positions_test,
                box,
                pairs_test,
                ds_test,
                buffer_scales_test,
                {"charge": charges_test},
            )
        regular_time_test = time.time() - start_time

        # Time JIT execution
        start_time = time.time()
        jit_energy_test = jit_module_test(
            positions_test,
            box,
            pairs_test,
            ds_test,
            buffer_scales_test,
            {"charge": charges_test},
        )
        for _ in range(4):  # Already computed once above
            jit_energy_test = jit_module_test(
                positions_test,
                box,
                pairs_test,
                ds_test,
                buffer_scales_test,
                {"charge": charges_test},
            )
        jit_time_test = time.time() - start_time

        assert torch.abs(energy_test - jit_energy_test) < 1e-10

        print(
            f"  {size:3d} atoms: Regular {regular_time_test:.4f}s, JIT {jit_time_test:.4f}s, Speedup {regular_time_test/jit_time_test:.2f}x"
        )

    # Test JIT with gradient computation
    print("\nTesting JIT with gradient computation:")

    # Regular module
    forces = -calc_grads(energy, positions)

    # JIT module
    jit_forces = -calc_grads(jit_energy, positions)

    # Compare forces
    force_diff = torch.max(torch.abs(forces - jit_forces))
    print(f"  Max force difference: {force_diff.item():.2e} eV/Å")
    assert force_diff < 1e-10
    print("  ✓ Forces match between regular and JIT modules")

    # Clean up
    if jit_path.exists():
        jit_path.unlink()
        print(f"\nCleaned up: {jit_path}")

    return module, jit_module


if __name__ == "__main__":
    main()
