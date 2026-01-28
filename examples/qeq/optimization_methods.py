# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Optimization methods example for QEq calculations.

This example demonstrates different optimization methods available for solving
the charge equilibration problem, including LBFGS and quadratic methods.
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


def compare_optimization_methods():
    """
    Compare different optimization methods for QEq.

    This function tests LBFGS and quadratic optimization methods
    and compares their performance and results.
    """
    print("Comparing QEq optimization methods...")

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

    # Set up charge constraints (total charge = 0)
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create QEq module with tighter convergence criteria
    module = QEqForceModule(
        rcut=rcut,
        ethresh=ethresh,
        eps=1e-6,  # Tight convergence threshold
        max_iter=50,  # Maximum iterations
    )

    # Test different optimization methods
    methods = ["lbfgs", "quadratic"]
    results = {}

    for method in methods:
        print(f"\nTesting {method.upper()} optimization method...")

        # Solve for equilibrium charges
        energy, q_opt = module.solve_pgrad(
            charges,
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
            method=method,
        )

        # Calculate forces
        forces = -calc_grads(energy, positions)

        # Store results
        results[method] = {
            "energy": energy.item(),
            "charges": q_opt.tolist(),
            "force_norm": torch.norm(forces).item(),
            "converge_iter": module.converge_iter,
            "total_charge": q_opt.sum().item(),
        }

        # Print results for this method
        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")
        print(f"  Force norm: {torch.norm(forces).item():.6f} eV/Å")
        print(f"  Converged in: {module.converge_iter} iterations")

    # Compare results between methods
    print("\nComparison between methods:")
    print(
        f"Energy difference: {abs(results['lbfgs']['energy'] - results['quadratic']['energy']):.8f} eV"
    )

    charge_diff = np.linalg.norm(
        np.array(results["lbfgs"]["charges"])
        - np.array(results["quadratic"]["charges"])
    )
    print(f"Charge difference (norm): {charge_diff:.8f} e")

    print(
        f"Iteration difference: {results['lbfgs']['converge_iter'] - results['quadratic']['converge_iter']}"
    )

    # Determine which method converged faster
    faster_method = (
        "lbfgs"
        if results["lbfgs"]["converge_iter"] < results["quadratic"]["converge_iter"]
        else "quadratic"
    )
    print(f"Faster method: {faster_method.upper()}")

    return results


def test_with_different_initial_guesses():
    """
    Test optimization methods with different initial charge guesses.
    """
    print("\nTesting optimization methods with different initial guesses...")

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

    # Set up charge constraints (total charge = 0)
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create QEq module
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Test different initial guesses
    initial_guesses = [
        ("zeros", torch.zeros(data_dict["n_atoms"], dtype=torch.float64)),
        ("random", torch.rand(data_dict["n_atoms"], dtype=torch.float64) - 0.5),
        (
            "uniform",
            torch.ones(data_dict["n_atoms"], dtype=torch.float64)
            / data_dict["n_atoms"],
        ),
    ]

    for guess_name, initial_charges in initial_guesses:
        print(f"\nTesting with {guess_name} initial guess...")

        # Solve for equilibrium charges using LBFGS
        energy, q_opt = module.solve_pgrad(
            initial_charges,
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
            method="lbfgs",
            reinit_q=True,  # Reinitialize charges based on constraints
        )

        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")
        print(f"  Converged in: {module.converge_iter} iterations")


def main():
    """
    Main function demonstrating different optimization methods for QEq.
    """
    print("Running QEq optimization methods example...")

    # Compare different optimization methods
    results = compare_optimization_methods()

    # Test with different initial guesses
    test_with_different_initial_guesses()

    print("\n✓ All optimization method tests completed successfully!")


if __name__ == "__main__":
    main()
