# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Advanced parameters example for QEq calculations.

This example demonstrates how to use various advanced parameters in QEq calculations,
including max_iter, ls_eps, eps, damping, and other configuration options.
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


def test_max_iter_parameter():
    """
    Test the effect of max_iter parameter on QEq convergence.
    """
    print("Testing max_iter parameter...")

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

    # Test different max_iter values
    max_iter_values = [5, 10, 20, 50]

    for max_iter in max_iter_values:
        print(f"\nTesting with max_iter = {max_iter}...")

        # Create QEq module with specific max_iter
        module = QEqForceModule(
            rcut=rcut,
            ethresh=ethresh,
            max_iter=max_iter,
            eps=1e-6,  # Tight convergence threshold
        )

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

        print(f"  Converged in: {module.converge_iter} iterations")
        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")

        # Check if convergence was achieved
        if module.converge_iter >= 0:
            print(f"  ✓ Converged within {max_iter} iterations")
        else:
            print(f"  ✗ Did not converge within {max_iter} iterations")


def test_eps_parameter():
    """
    Test the effect of eps (convergence threshold) parameter.
    """
    print("\nTesting eps (convergence threshold) parameter...")

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

    # Test different eps values
    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]

    for eps in eps_values:
        print(f"\nTesting with eps = {eps}...")

        # Create QEq module with specific eps
        module = QEqForceModule(rcut=rcut, ethresh=ethresh, max_iter=50, eps=eps)

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

        print(f"  Converged in: {module.converge_iter} iterations")
        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")


def test_damping_parameter():
    """
    Test the effect of damping parameter.
    """
    print("\nTesting damping parameter...")

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

    # Test with and without damping
    damping_options = [True, False]

    for damping in damping_options:
        print(f"\nTesting with damping = {damping}...")

        # Create QEq module with specific damping setting
        module = QEqForceModule(
            rcut=rcut, ethresh=ethresh, max_iter=50, damping=damping
        )

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

        print(f"  Converged in: {module.converge_iter} iterations")
        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")

        # Check which submodels are included
        submodels = list(module.submodels.keys())
        print(f"  Active submodels: {submodels}")


def test_kspace_parameters():
    """
    Test kspace and rspace parameters.
    """
    print("\nTesting kspace and rspace parameters...")

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

    # Test different kspace/rspace combinations
    configurations = [
        {"kspace": True, "rspace": True, "name": "Full PME (kspace + rspace)"},
        {"kspace": True, "rspace": False, "name": "Reciprocal space only"},
        {"kspace": False, "rspace": True, "name": "Real space only"},
    ]

    for config in configurations:
        print(f"\nTesting {config['name']}...")

        # Create QEq module with specific kspace/rspace settings
        module = QEqForceModule(
            rcut=rcut,
            ethresh=ethresh,
            max_iter=50,
            kspace=config["kspace"],
            rspace=config["rspace"],
        )

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

        print(f"  Converged in: {module.converge_iter} iterations")
        print(f"  Energy: {energy.item():.6f} eV")
        print(f"  Total charge: {q_opt.sum().item():.6f} e")


def main():
    """
    Main function demonstrating advanced parameters for QEq.
    """
    print("Running QEq advanced parameters example...")

    # Test different parameters
    test_max_iter_parameter()
    test_eps_parameter()
    test_damping_parameter()
    test_kspace_parameters()

    print("\n✓ All advanced parameter tests completed successfully!")


if __name__ == "__main__":
    main()
