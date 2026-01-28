# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Convergence criteria example for QEq calculations.

This example demonstrates how to customize convergence criteria in QEq calculations,
including different threshold values and monitoring convergence behavior.
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
from torch_admp.utils import (
    calc_grads,
    calc_pgrads,
    vector_projection,
    vector_projection_coeff_matrix,
)

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


def test_convergence_thresholds():
    """
    Test different convergence thresholds and their effects.
    """
    print("Testing different convergence thresholds...")

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

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Test different convergence thresholds
    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]

    for eps in eps_values:
        print(f"\nTesting with convergence threshold eps = {eps}...")

        # Create QEq module with specific convergence threshold
        module = QEqForceModule(rcut=rcut, ethresh=ethresh, max_iter=100, eps=eps)

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
            method="lbfgs",
        )

        # Calculate final projected gradient norm
        from torch_admp.utils import vector_projection_coeff_matrix

        coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)
        pgrad_norm = calc_pgrads(energy, q_opt, constraint_matrix, coeff_matrix).norm()
        pgrad_norm_normalized = pgrad_norm / q_opt.shape[0]

        print(f"  Converged in: {module.converge_iter} iterations")
        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")
        print(f"  Final projected gradient norm: {pgrad_norm_normalized.item():.8f}")
        print(f"  Target threshold: {eps}")

        # Check if convergence was achieved
        if module.converge_iter >= 0:
            print(f"  ✓ Converged successfully")
        else:
            print(f"  ✗ Did not converge within max_iter")


def monitor_convergence_history():
    """
    Monitor convergence history during optimization.
    """
    print("\nMonitoring convergence history...")

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

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)
    coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

    # Create QEq module with loose convergence for monitoring
    module = QEqForceModule(rcut=rcut, ethresh=ethresh, max_iter=20, eps=1e-6)

    # Monitor convergence manually by running iterations
    print("\nConvergence history for LBFGS method:")
    print("Iter | Energy (eV) | |∇E|/N | Charge Sum")
    print("-" * 45)

    # Initialize charges
    q_current = torch.zeros(
        data_dict["n_atoms"], dtype=torch.float64, requires_grad=True
    )
    q_current = torch.nn.Parameter(q_current)

    # Apply constraints
    q_current.data = vector_projection(
        torch.ones(data_dict["n_atoms"], 1), constraint_matrix, constraint_vals
    )

    # Set up LBFGS optimizer
    optimizer = torch.optim.LBFGS([q_current], max_iter=1)

    energy_history = []
    pgrad_history = []

    for iteration in range(module.max_iter):

        def closure():
            optimizer.zero_grad()
            energy = module.func_energy(
                q_current, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
            )
            pgrads = calc_pgrads(energy, q_current, constraint_matrix, coeff_matrix)
            q_current.grad = pgrads.detach()
            return energy

        energy = optimizer.step(closure)
        energy_history.append(energy.item())

        energy = module.func_energy(
            q_current, positions, box, chi, hardness, eta, pairs, ds, buffer_scales
        )
        # Calculate projected gradient norm
        pgrads = calc_pgrads(energy, q_current, constraint_matrix, coeff_matrix)
        pgrad_norm = pgrads.norm() / q_current.shape[0]
        pgrad_history.append(pgrad_norm.item())

        # Apply constraints after optimization step
        with torch.no_grad():
            q_current.data = vector_projection(
                q_current.data, constraint_matrix, constraint_vals
            )

        # Print iteration info
        print(
            f"{iteration:4d} | {energy.item():10.6f} | {pgrad_norm.item():8.6f} | {q_current.sum().item():8.6f}"
        )

        # Check convergence
        if pgrad_norm.item() < module.eps:
            print(f"Converged after {iteration + 1} iterations")
            break

    return energy_history, pgrad_history


def compare_convergence_methods():
    """
    Compare convergence behavior of different optimization methods.
    """
    print("\nComparing convergence behavior of different methods...")

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

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Test different methods
    methods = ["lbfgs", "quadratic"]

    print("\nMethod | Iterations | Energy (eV) | Final |∇E|/N")
    print("-" * 50)

    for method in methods:
        # Create QEq module
        module = QEqForceModule(rcut=rcut, ethresh=ethresh, max_iter=100, eps=1e-6)

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

        # Calculate final projected gradient norm
        from torch_admp.utils import vector_projection_coeff_matrix

        coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)
        pgrad_norm = calc_pgrads(energy, q_opt, constraint_matrix, coeff_matrix).norm()
        pgrad_norm_normalized = pgrad_norm / q_opt.shape[0]

        print(
            f"{method:7s} | {module.converge_iter:10d} | {energy.item():10.6f} | {pgrad_norm_normalized.item():10.8f}"
        )


def test_line_search_parameters():
    """
    Test line search parameters for quadratic optimization.
    """
    print("\nTesting line search parameters for quadratic optimization...")

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

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Test different ls_eps values (line search threshold)
    ls_eps_values = [1e-2, 1e-4, 1e-6, 1e-8]

    print("\nls_eps | Iterations | Energy (eV)")
    print("-" * 35)

    for ls_eps in ls_eps_values:
        # Create QEq module with specific ls_eps
        module = QEqForceModule(
            rcut=rcut, ethresh=ethresh, max_iter=50, eps=1e-6, ls_eps=ls_eps
        )

        # Solve for equilibrium charges using quadratic method
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
            method="quadratic",
        )

        print(f"{ls_eps:6.1e} | {module.converge_iter:10d} | {energy.item():10.6f}")


def main():
    """
    Main function demonstrating convergence criteria for QEq.
    """
    print("Running QEq convergence criteria example...")

    # Test different convergence thresholds
    test_convergence_thresholds()

    # Monitor convergence history
    monitor_convergence_history()

    # Compare convergence methods
    compare_convergence_methods()

    # Test line search parameters
    test_line_search_parameters()

    print("\n✓ All convergence criteria tests completed successfully!")


if __name__ == "__main__":
    main()
