# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Batch Processing Example for torch-admp

This example demonstrates batch processing for multiple configurations:
1. Setting up batched systems
2. Calculating energies for multiple frames simultaneously
3. Efficient processing of multiple configurations
"""

import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def main():
    """
    Demonstrate batch processing for multiple configurations.
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    # System parameters
    rcut = 6.0
    n_atoms = 50
    n_frames = 5
    ethresh = 1e-5
    l_box = 20.0

    # Generate batch of systems
    np.random.seed(42)
    batch_positions = []
    batch_charges = []

    print(f"\nGenerating {n_frames} frames with {n_atoms} atoms each...")
    for frame in range(n_frames):
        positions = np.random.rand(n_atoms, 3) * l_box
        charges = np.random.uniform(-1.0, 1.0, (n_atoms))
        charges -= charges.mean()

        batch_positions.append(positions)
        batch_charges.append(charges)
        print(
            f"  Frame {frame+1}: generated {n_atoms} atoms with total charge {charges.sum():.6f}"
        )

    # Convert to PyTorch tensors (batch dimension first)
    batch_positions = torch.tensor(
        batch_positions, requires_grad=True
    )  # (n_frames, n_atoms, 3)
    batch_box = (
        torch.tensor(np.diag([l_box, l_box, l_box])).unsqueeze(0).repeat(n_frames, 1, 1)
    )  # (n_frames, 3, 3)
    batch_charges = torch.tensor(batch_charges)  # (n_frames, n_atoms)

    print(f"\nBatch tensor shapes:")
    print(f"  Positions: {batch_positions.shape}")
    print(f"  Box: {batch_box.shape}")
    print(f"  Charges: {batch_charges.shape}")

    # Create neighbor list for each frame
    print(f"\nBuilding neighbor lists for each frame...")
    nblist = TorchNeighborList(cutoff=rcut)
    batch_pairs = []
    batch_ds = []
    batch_buffer_scales = []

    for frame in range(n_frames):
        pairs = nblist(batch_positions[frame], batch_box[frame])
        batch_pairs.append(pairs)
        batch_ds.append(nblist.get_ds())
        batch_buffer_scales.append(nblist.get_buffer_scales())
        print(f"  Frame {frame+1}: {pairs.shape[0]} pairs")

    # Stack batch data
    batch_pairs = pad_sequence(batch_pairs, batch_first=True, padding_value=0)
    batch_ds = pad_sequence(batch_ds, batch_first=True, padding_value=0)
    batch_buffer_scales = pad_sequence(
        batch_buffer_scales, batch_first=True, padding_value=0
    )

    print(f"\nBatch neighbor list shapes:")
    print(f"  Pairs: {batch_pairs.shape}")
    print(f"  Distances: {batch_ds.shape}")
    print(f"  Buffer scales: {batch_buffer_scales.shape}")

    # Create PME module
    module = CoulombForceModule(rcut=rcut, ethresh=ethresh)

    # Calculate batch energies
    print(f"\nCalculating batch energies...")
    start_time = time.time()
    batch_energies = module(
        batch_positions,
        batch_box,
        batch_pairs,
        batch_ds,
        batch_buffer_scales,
        {"charge": batch_charges},
    )
    batch_time = time.time() - start_time

    print(f"Batch calculation completed in {batch_time:.4f} seconds")
    print(f"\nBatch processing results:")
    print(f"Energies shape: {batch_energies.shape}")
    for frame, energy in enumerate(batch_energies):
        print(f"  Frame {frame+1}: {energy.item():.6f} eV")

    # Calculate forces for the first frame as an example
    print(f"\nCalculating forces...")
    forces = -calc_grads(batch_energies, batch_positions)
    print(f"Forces shape: {forces.shape}")
    print(
        f"Max force magnitude: {torch.max(torch.norm(forces, dim=-1)).item():.6f} eV/Å"
    )

    # Performance comparison: batch vs individual
    print(f"\nPerformance comparison: batch vs individual processing...")

    # Individual processing
    start_time = time.time()
    individual_energies = []
    for frame in range(n_frames):
        energy = module(
            batch_positions[frame : frame + 1],
            batch_box[frame : frame + 1],
            batch_pairs[frame : frame + 1],
            batch_ds[frame : frame + 1],
            batch_buffer_scales[frame : frame + 1],
            {"charge": batch_charges[frame : frame + 1]},
        )
        individual_energies.append(energy)
    individual_time = time.time() - start_time

    print(f"Individual processing time: {individual_time:.4f} seconds")
    print(f"Batch processing time: {batch_time:.4f} seconds")
    print(f"Speedup: {individual_time/batch_time:.2f}x")

    # Verify results are identical
    individual_energies = torch.stack(individual_energies).squeeze()
    max_diff = torch.max(torch.abs(batch_energies - individual_energies))
    print(f"Max difference between batch and individual: {max_diff.item():.2e} eV")
    assert max_diff < 1e-10
    print("✓ Batch and individual results are identical")

    # Test with different configurations
    print(f"\nTesting batch processing with different configurations...")

    # Test with slab correction
    module_slab = CoulombForceModule(
        rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2
    )
    batch_energies_slab = module_slab(
        batch_positions,
        batch_box,
        batch_pairs,
        batch_ds,
        batch_buffer_scales,
        {"charge": batch_charges},
    )

    print(f"Batch energies with slab correction:")
    for frame, energy in enumerate(batch_energies_slab):
        print(
            f"  Frame {frame+1}: {energy.item():.6f} eV (slab correction: {module_slab.slab_corr_energy[frame].item():.6f} eV)"
        )

    # Test with different system sizes
    print(f"\nTesting batch processing with different system sizes...")
    system_sizes = [25, 50, 100]
    n_frames_test = 3

    for size in system_sizes:
        # Generate batch
        test_positions = []
        test_charges = []

        for frame in range(n_frames_test):
            positions = np.random.rand(size, 3) * l_box
            charges = np.random.uniform(-1.0, 1.0, (size))
            charges -= charges.mean()

            test_positions.append(positions)
            test_charges.append(charges)

        test_positions = torch.tensor(test_positions, requires_grad=True)
        test_charges = torch.tensor(test_charges)

        # Create neighbor lists
        test_pairs = []
        test_ds = []
        test_buffer_scales = []

        for frame in range(n_frames_test):
            pairs = nblist(test_positions[frame], batch_box[0])
            test_pairs.append(pairs)
            test_ds.append(nblist.get_ds())
            test_buffer_scales.append(nblist.get_buffer_scales())

        test_pairs = pad_sequence(test_pairs, batch_first=True, padding_value=0)
        test_ds = pad_sequence(test_ds, batch_first=True, padding_value=0)
        test_buffer_scales = pad_sequence(
            test_buffer_scales, batch_first=True, padding_value=0
        )

        # Time batch processing
        start_time = time.time()
        test_energies = module(
            test_positions,
            batch_box[:n_frames_test],
            test_pairs,
            test_ds,
            test_buffer_scales,
            {"charge": test_charges},
        )
        test_time = time.time() - start_time

        print(f"  {size:3d} atoms × {n_frames_test} frames: {test_time:.4f} seconds")

    # Memory usage comparison
    print(f"\nMemory usage comparison...")

    # Simulate individual processing memory
    individual_memory = 0
    for frame in range(n_frames):
        # Rough estimate of memory for individual processing
        individual_memory += batch_positions[frame].numel() * 8  # float64
        individual_memory += batch_pairs[frame].numel() * 8
        individual_memory += batch_ds[frame].numel() * 8
        individual_memory += batch_buffer_scales[frame].numel() * 8
        individual_memory += batch_charges[frame].numel() * 8

    # Batch processing memory
    batch_memory = batch_positions.numel() * 8
    batch_memory += batch_pairs.numel() * 8
    batch_memory += batch_ds.numel() * 8
    batch_memory += batch_buffer_scales.numel() * 8
    batch_memory += batch_charges.numel() * 8

    print(f"  Individual processing (estimated): {individual_memory/1024/1024:.2f} MB")
    print(f"  Batch processing: {batch_memory/1024/1024:.2f} MB")
    print(f"  Memory efficiency: {individual_memory/batch_memory:.2f}x")

    return module, batch_energies


if __name__ == "__main__":
    main()
