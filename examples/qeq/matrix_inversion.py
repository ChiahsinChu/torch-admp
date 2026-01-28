# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Matrix inversion method example for QEq calculations.

This example demonstrates how to use the matrix inversion method to solve
the charge equilibration problem, which directly solves the linear system
without iterative optimization.
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
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml = XMLIO()
    xml.loadXML(os.path.join(script_dir, "qeq.xml"))
    res = xml.parseResidues()
    ffinfo = xml.parseXML()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = np.array([a["type"] for a in res[0]["particles"]])

    pdb = app.PDBFile(os.path.join(script_dir, "qeq.pdb"))
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


def main():
    """
    Main function demonstrating matrix inversion method for QEq.
    """
    print("Running QEq matrix inversion example...")

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
    constraint_matrix = torch.ones(
        [1, data_dict["n_atoms"]], dtype=positions.dtype, device=positions.device
    )
    constraint_vals = torch.zeros(1, dtype=positions.dtype, device=positions.device)

    # Create QEq module
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Solve for equilibrium charges using matrix inversion method
    print("\nSolving QEq using matrix inversion method...")
    energy, q_opt, diag_hessian, fermi = module.solve_matrix_inversion(
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
        check_hessian=False,
    )

    # Calculate forces
    forces = -calc_grads(energy, positions)

    # Print results
    print(f"Total energy: {energy.item():.6f} eV")
    print(f"Initial charges: {charges.tolist()}")
    print(f"Optimized charges: {q_opt.tolist()}")
    print(f"Total charge (should be ~0): {q_opt.sum().item():.6f}")
    print(f"Force norm: {torch.norm(forces).item():.6f} eV/Å")
    print(f"Fermi level: {fermi.item():.6f} eV")
    print(f"Diagonal Hessian (first 5): {diag_hessian[:5].tolist()}")

    # Compare with projected gradient method
    print("\nComparing with projected gradient method...")
    energy_pgrad, q_opt_pgrad = module.solve_pgrad(
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
    )

    # Calculate differences
    energy_diff = torch.abs(energy - energy_pgrad).item()
    charge_diff = torch.norm(q_opt - q_opt_pgrad).item()

    print(f"Energy difference: {energy_diff:.8f} eV")
    print(f"Charge difference (norm): {charge_diff:.8f} e")
    print(f"QEq (PG) converged in {module.converge_iter} step(s)")

    # Verify consistency
    if energy_diff < 1e-6 and charge_diff < 1e-6:
        print(
            "✓ Matrix inversion and projected gradient methods give consistent results!"
        )
    else:
        print("⚠ Warning: Methods show significant differences")


if __name__ == "__main__":
    main()
