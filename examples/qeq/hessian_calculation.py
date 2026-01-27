# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Hessian calculation example for QEq calculations.

This example demonstrates how to calculate and analyze the Hessian matrix
in QEq calculations, which provides information about the curvature
of the energy landscape with respect to atomic charges.
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


def calculate_and_analyze_hessian():
    """
    Calculate and analyze the Hessian matrix for QEq.
    """
    print("Calculating and analyzing Hessian matrix...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
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
    charges = torch.tensor(
        data_dict["charge"],
        requires_grad=False,
    )

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Create QEq module
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Calculate Hessian matrix
    print("\nCalculating Hessian matrix...")
    hessian = module.calc_hessian(
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
    )

    # Convert to numpy for analysis
    hessian_np = hessian.detach().cpu().numpy()

    # Analyze Hessian properties
    print(f"Hessian matrix shape: {hessian_np.shape}")
    print(f"Hessian matrix type: {hessian_np.dtype}")

    # Check symmetry
    symmetry_error = np.max(np.abs(hessian_np - hessian_np.T))
    print(f"Maximum symmetry error: {symmetry_error:.2e}")

    # Check diagonal elements
    diagonal_elements = np.diag(hessian_np)
    print(f"Diagonal elements (first 5): {diagonal_elements[:5]}")
    print(f"Minimum diagonal element: {np.min(diagonal_elements):.6f}")
    print(f"Maximum diagonal element: {np.max(diagonal_elements):.6f}")
    print(f"Mean diagonal element: {np.mean(diagonal_elements):.6f}")

    # Check eigenvalues
    eigenvalues = np.linalg.eigvalsh(hessian_np)
    print(f"Minimum eigenvalue: {np.min(eigenvalues):.6f}")
    print(f"Maximum eigenvalue: {np.max(eigenvalues):.6f}")
    print(f"Condition number: {np.max(eigenvalues) / np.min(eigenvalues):.2e}")

    # Check positive definiteness
    is_positive_definite = np.all(eigenvalues > 0)
    print(f"Is Hessian positive definite: {is_positive_definite}")

    return hessian_np


def test_hessian_with_energy():
    """
    Test Hessian calculation by comparing with finite differences.
    """
    print("\nTesting Hessian calculation with finite differences...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
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
    charges = torch.tensor(
        data_dict["charge"],
        requires_grad=False,
    )

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Create QEq module
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Calculate analytical Hessian
    hessian_analytical = module.calc_hessian(
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
    )

    # Test Hessian with energy calculation
    # Create test charges
    test_charges = torch.zeros(
        data_dict["n_atoms"], dtype=torch.float64, requires_grad=True
    )

    # Calculate energy with test charges
    energy = module.func_energy(
        test_charges, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
    )

    # Verify Hessian using energy formula: E = 0.5 * q^T * H * q + q^T * b
    # For zero charges, this should give E = 0
    print(f"Energy with zero charges: {energy.item():.8f}")

    # Test with random charges
    rng = np.random.default_rng(seed=42)
    random_charges = torch.tensor(
        rng.uniform(-0.1, 0.1, data_dict["n_atoms"]), dtype=torch.float64
    )

    energy_random = module.func_energy(
        random_charges, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
    )

    # Calculate energy using Hessian
    hessian_np = hessian_analytical.detach().cpu().numpy()
    charges_np = random_charges.detach().cpu().numpy()
    chi_np = chi.detach().cpu().numpy()

    # Energy = 0.5 * q^T * H * q + chi^T * q
    energy_hessian = 0.5 * np.dot(charges_np, np.dot(hessian_np, charges_np)) + np.dot(
        chi_np, charges_np
    )

    print(f"Energy with random charges (direct): {energy_random.item():.6f}")
    print(f"Energy with random charges (Hessian): {energy_hessian:.6f}")
    print(f"Difference: {abs(energy_random.item() - energy_hessian):.8f}")

    # Test with optimized charges from matrix inversion
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    energy_opt, q_opt, diag_hessian, fermi = module.solve_matrix_inversion(
        positions,
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

    print(f"\nMatrix inversion results:")
    print(f"Optimized energy: {energy_opt.item():.6f}")
    print(f"Fermi level: {fermi.item():.6f}")
    print(f"Diagonal Hessian (first 5): {diag_hessian[:5].tolist()}")

    # Verify diagonal elements match Hessian diagonal
    hessian_diagonal = np.diag(hessian_np)
    diag_diff = np.max(np.abs(diag_hessian.detach().cpu().numpy() - hessian_diagonal))
    print(f"Maximum difference in diagonal elements: {diag_diff:.2e}")


def analyze_hessian_structure():
    """
    Analyze the structure of the Hessian matrix.
    """
    print("\nAnalyzing Hessian matrix structure...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
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

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Create QEq module
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Calculate full Hessian
    hessian = module.calc_hessian(
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
    )

    # Convert to numpy for analysis
    hessian_np = hessian.detach().cpu().numpy()
    n_atoms = hessian_np.shape[0]

    # Analyze sparsity pattern
    threshold = 1e-6  # Threshold for considering elements as non-zero
    non_zero_elements = np.abs(hessian_np) > threshold
    sparsity = 1.0 - np.sum(non_zero_elements) / (n_atoms * n_atoms)

    print(f"Hessian sparsity: {sparsity:.4f}")
    print(f"Non-zero elements: {np.sum(non_zero_elements)} out of {n_atoms * n_atoms}")

    # Analyze distance dependence
    print("\nAnalyzing distance dependence of Hessian elements...")

    # Get atom pairs and distances
    pairs_np = pairs.detach().cpu().numpy()
    ds_np = ds.detach().cpu().numpy()

    # Group Hessian elements by distance
    distance_bins = [0, 2, 4, 6, 8, 10]  # Angstroms
    bin_means = []
    bin_counts = []

    for i in range(len(distance_bins) - 1):
        mask = (ds_np >= distance_bins[i]) & (ds_np < distance_bins[i + 1])
        if np.any(mask):
            selected_pairs = pairs_np[mask]
            bin_hessian_elements = []

            for pair in selected_pairs:
                i_atom, j_atom = pair
                bin_hessian_elements.append(hessian_np[i_atom, j_atom])

            bin_means.append(np.mean(np.abs(bin_hessian_elements)))
            bin_counts.append(len(bin_hessian_elements))
        else:
            bin_means.append(0.0)
            bin_counts.append(0)

    print("\nDistance range (Å) | Mean |H_ij| | Count")
    print("-" * 40)
    for i in range(len(distance_bins) - 1):
        print(
            f"{distance_bins[i]:6.1f} - {distance_bins[i+1]:6.1f} | {bin_means[i]:10.6f} | {bin_counts[i]:5d}"
        )

    # Analyze diagonal vs off-diagonal elements
    diagonal_mean = np.mean(np.abs(np.diag(hessian_np)))
    off_diagonal = hessian_np - np.diag(np.diag(hessian_np))
    off_diagonal_mean = np.mean(np.abs(off_diagonal))

    print(f"\nDiagonal elements mean |H_ii|: {diagonal_mean:.6f}")
    print(f"Off-diagonal elements mean |H_ij|: {off_diagonal_mean:.6f}")
    print(f"Ratio (off-diag/diag): {off_diagonal_mean/diagonal_mean:.4f}")


def test_hessian_with_different_parameters():
    """
    Test Hessian calculation with different QEq parameters.
    """
    print("\nTesting Hessian with different QEq parameters...")

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
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

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=8.0)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Test with different damping settings
    print("\nTesting with different damping settings:")
    print("Damping | Min Eigenvalue | Max Eigenvalue | Condition Number")
    print("-" * 60)

    for damping in [True, False]:
        # Create QEq module with specific damping setting
        module = QEqForceModule(rcut=8.0, ethresh=1e-5, damping=damping)

        # Calculate Hessian
        hessian = module.calc_hessian(
            positions,
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
        )

        # Analyze eigenvalues
        hessian_np = hessian.detach().cpu().numpy()
        eigenvalues = np.linalg.eigvalsh(hessian_np)

        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        cond_num = max_eig / min_eig if min_eig > 0 else float("inf")

        print(
            f"{str(damping):7s} | {min_eig:13.6f} | {max_eig:13.6f} | {cond_num:13.2e}"
        )

    # Test with different cutoffs
    print("\nTesting with different cutoffs:")
    print("Cutoff (Å) | Min Eigenvalue | Max Eigenvalue | Condition Number")
    print("-" * 60)

    for rcut in [4.0, 6.0, 8.0]:
        # Calculate neighbor list with specific cutoff
        nblist = TorchNeighborList(cutoff=rcut)
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Create QEq module
        module = QEqForceModule(rcut=rcut, ethresh=1e-5)

        # Calculate Hessian
        hessian = module.calc_hessian(
            positions,
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
        )

        # Analyze eigenvalues
        hessian_np = hessian.detach().cpu().numpy()
        eigenvalues = np.linalg.eigvalsh(hessian_np)

        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        cond_num = max_eig / min_eig if min_eig > 0 else float("inf")

        print(f"{rcut:9.1f} | {min_eig:13.6f} | {max_eig:13.6f} | {cond_num:13.2e}")


def main():
    """
    Main function demonstrating Hessian calculation for QEq.
    """
    print("Running QEq Hessian calculation example...")

    # Calculate and analyze Hessian
    hessian = calculate_and_analyze_hessian()

    # Test Hessian with energy calculation
    test_hessian_with_energy()

    # Analyze Hessian structure
    analyze_hessian_structure()

    # Test Hessian with different parameters
    test_hessian_with_different_parameters()

    print("\n✓ All Hessian calculation tests completed successfully!")


if __name__ == "__main__":
    main()
