# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Batch processing example for QEq calculations.

This example demonstrates how to process multiple molecular configurations
efficiently using batch processing capabilities of torch-admp.
"""

import jax.numpy as jnp
import numpy as np
import openmm.app as app
import openmm.unit as unit
import torch
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
from scipy import constants

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def load_test_data():
    """
    Load test data from PDB and XML files.

    Returns
    -------
    dict
        Dictionary containing molecular system data.
    """
    xml = XMLIO()
    xml.loadXML("qeq.xml")
    res = xml.parseResidues()
    ffinfo = xml.parseXML()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = np.array([a["type"] for a in res[0]["particles"]])

    pdb = app.PDBFile("qeq.pdb")
    dmfftop = DMFFTopology(from_top=pdb.topology)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positions = jnp.array(positions)
    a, b, c = dmfftop.getPeriodicBoxVectors()

    n_atoms = dmfftop.getNumAtoms()
    eta = np.zeros([n_atoms])
    chi = np.zeros([n_atoms])
    hardness = np.zeros([n_atoms])
    for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
        eta[types == _data["attrib"]["type"]] = float(_data["attrib"]["eta"])
        chi[types == _data["attrib"]["type"]] = float(_data["attrib"]["chi"])
        hardness[types == _data["attrib"]["type"]] = float(_data["attrib"]["J"])

    # Convert energy units from kJ/mol to eV/particle
    j2ev = constants.physical_constants["joule-electron volt relationship"][0]
    energy_coeff = j2ev * constants.kilo / constants.Avogadro

    data_dict = {
        "n_atoms": n_atoms,
        "position": np.array(positions),
        "box": np.array([a._value, b._value, c._value]) * 10.0,
        "chi": chi * energy_coeff,
        "hardness": hardness * energy_coeff,
        "eta": eta,
        "charge": charges,
    }
    return data_dict


def generate_batch_configurations(base_positions, n_configs=5, displacement=0.1):
    """
    Generate multiple configurations by applying random displacements.

    Parameters
    ----------
    base_positions : np.ndarray
        Base positions with shape (n_atoms, 3)
    n_configs : int, optional
        Number of configurations to generate, by default 5
    displacement : float, optional
        Maximum displacement in Angstroms, by default 0.1

    Returns
    -------
    torch.Tensor
        Batch of positions with shape (n_configs, n_atoms, 3)
    """
    n_atoms = base_positions.shape[0]

    # Generate random displacements
    rng = np.random.default_rng(seed=42)
    displacements = rng.uniform(-displacement, displacement, (n_configs, n_atoms, 3))

    # Apply displacements to base positions
    batch_positions = base_positions + displacements

    return torch.tensor(batch_positions, dtype=torch.float64)


def test_batch_processing():
    """
    Test batch processing of multiple configurations.
    """
    print("Testing batch processing of multiple configurations...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5
    n_configs = 5

    # Load molecular data
    data_dict = load_test_data()

    # Generate batch of configurations
    base_positions = data_dict["position"]
    batch_positions = generate_batch_configurations(base_positions, n_configs)

    # Convert other data to PyTorch tensors
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )

    # Expand parameters for batch processing
    batch_chi = chi.unsqueeze(0).expand(n_configs, -1)
    batch_hardness = hardness.unsqueeze(0).expand(n_configs, -1)
    batch_eta = eta.unsqueeze(0).expand(n_configs, -1)
    batch_box = box.unsqueeze(0).expand(n_configs, -1, -1)

    # Set up initial charges for batch
    batch_charges = torch.zeros(n_configs, data_dict["n_atoms"], dtype=torch.float64)

    # Calculate neighbor list for each configuration
    nblist = TorchNeighborList(cutoff=rcut)
    batch_pairs = []
    batch_ds = []
    batch_buffer_scales = []

    for i in range(n_configs):
        pairs = nblist(batch_positions[i], batch_box[i])
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        batch_pairs.append(pairs)
        batch_ds.append(ds)
        batch_buffer_scales.append(buffer_scales)

    # Set up charge constraints (total charge = 0) for batch
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Process each configuration in the batch
    print(f"\nProcessing {n_configs} configurations...")
    print("Config | Energy (eV) | Total Charge (e) | Iterations")
    print("-" * 55)

    for i in range(n_configs):
        # Create QEq module
        module = QEqForceModule(rcut=rcut, ethresh=ethresh)

        # Solve for equilibrium charges
        energy, q_opt = module.solve_pgrad(
            batch_charges[i],
            batch_positions[i],
            batch_box[i],
            batch_chi[i],
            batch_hardness[i],
            batch_eta[i],
            batch_pairs[i],
            batch_ds[i],
            batch_buffer_scales[i],
            constraint_matrix,
            constraint_vals,
        )

        print(
            f"{i:5d} | {energy.item():10.6f} | {q_opt.sum().item():14.6f} | {module.converge_iter:10d}"
        )

    # Calculate statistics across the batch
    print("\nBatch processing completed!")
    print("Note: Each configuration is processed independently.")


def test_parallel_processing():
    """
    Test parallel processing of multiple configurations.
    """
    print("\nTesting parallel processing approach...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5
    n_configs = 5

    # Load molecular data
    data_dict = load_test_data()

    # Generate batch of configurations
    base_positions = data_dict["position"]
    batch_positions = generate_batch_configurations(base_positions, n_configs)

    # Convert other data to PyTorch tensors
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create a single QEq module for all configurations
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Process configurations sequentially but efficiently
    print(f"\nProcessing {n_configs} configurations with single module...")
    print("Config | Energy (eV) | Total Charge (e) | Iterations")
    print("-" * 55)

    for i in range(n_configs):
        # Calculate neighbor list for current configuration
        nblist = TorchNeighborList(cutoff=rcut)
        pairs = nblist(batch_positions[i], box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Set up initial charges
        charges = torch.zeros(data_dict["n_atoms"], dtype=torch.float64)

        # Solve for equilibrium charges
        energy, q_opt = module.solve_pgrad(
            charges,
            batch_positions[i],
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
        )

        print(
            f"{i:5d} | {energy.item():10.6f} | {q_opt.sum().item():14.6f} | {module.converge_iter:10d}"
        )


def test_trajectory_processing():
    """
    Test processing a trajectory of configurations.
    """
    print("\nTesting trajectory processing...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5
    n_frames = 10

    # Load molecular data
    data_dict = load_test_data()

    # Generate a simple trajectory (linear interpolation between two states)
    base_positions = data_dict["position"]
    target_positions = base_positions + np.random.normal(0, 0.5, base_positions.shape)

    trajectory_positions = []
    for i in range(n_frames):
        alpha = i / (n_frames - 1)  # Interpolation parameter
        frame_positions = (1 - alpha) * base_positions + alpha * target_positions
        trajectory_positions.append(frame_positions)

    # Convert to PyTorch tensor
    trajectory_positions = torch.tensor(
        np.array(trajectory_positions), dtype=torch.float64
    )

    # Convert other data to PyTorch tensors
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Process trajectory
    print(f"\nProcessing trajectory with {n_frames} frames...")
    print("Frame | Energy (eV) | Total Charge (e) | Iterations")
    print("-" * 55)

    energies = []
    charges_list = []

    for i in range(n_frames):
        # Calculate neighbor list for current frame
        nblist = TorchNeighborList(cutoff=rcut)
        pairs = nblist(trajectory_positions[i], box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Set up initial charges
        charges = torch.zeros(data_dict["n_atoms"], dtype=torch.float64)

        # Create QEq module
        module = QEqForceModule(rcut=rcut, ethresh=ethresh)

        # Solve for equilibrium charges
        energy, q_opt = module.solve_pgrad(
            charges,
            trajectory_positions[i],
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
        )

        energies.append(energy.item())
        charges_list.append(q_opt.tolist())

        print(
            f"{i:5d} | {energy.item():10.6f} | {q_opt.sum().item():14.6f} | {module.converge_iter:10d}"
        )

    # Calculate trajectory statistics
    energies = np.array(energies)
    print(f"\nTrajectory statistics:")
    print(f"  Mean energy: {np.mean(energies):.6f} eV")
    print(f"  Energy std:  {np.std(energies):.6f} eV")
    print(f"  Energy range: {np.max(energies) - np.min(energies):.6f} eV")


def test_batch_force_calculation():
    """
    Test force calculation for multiple configurations.
    """
    print("\nTesting batch force calculation...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5
    n_configs = 3

    # Load molecular data
    data_dict = load_test_data()

    # Generate batch of configurations
    base_positions = data_dict["position"]
    batch_positions = generate_batch_configurations(
        base_positions, n_configs, displacement=0.05
    )

    # Convert to PyTorch tensors with gradient tracking
    batch_positions = batch_positions.clone().detach().requires_grad_(True)

    # Convert other data to PyTorch tensors
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Process each configuration and calculate forces
    print(f"\nCalculating forces for {n_configs} configurations...")
    print("Config | Energy (eV) | Force Norm (eV/Å) | Iterations")
    print("-" * 60)

    for i in range(n_configs):
        # Calculate neighbor list for current configuration
        nblist = TorchNeighborList(cutoff=rcut)
        pairs = nblist(batch_positions[i], box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Set up initial charges
        charges = torch.zeros(data_dict["n_atoms"], dtype=torch.float64)

        # Create QEq module
        module = QEqForceModule(rcut=rcut, ethresh=ethresh)

        # Solve for equilibrium charges
        energy, q_opt = module.solve_pgrad(
            charges,
            batch_positions[i],
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
        )

        # Calculate forces
        forces = -calc_grads(energy, batch_positions)
        force_norm = torch.norm(forces).item()

        print(
            f"{i:5d} | {energy.item():10.6f} | {force_norm:16.6f} | {module.converge_iter:10d}"
        )


def main():
    """
    Main function demonstrating batch processing for QEq.
    """
    print("Running QEq batch processing example...")

    # Test different batch processing approaches
    test_batch_processing()
    test_parallel_processing()
    test_trajectory_processing()
    test_batch_force_calculation()

    print("\n✓ All batch processing tests completed successfully!")


if __name__ == "__main__":
    main()
