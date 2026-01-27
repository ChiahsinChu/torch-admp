# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Basic QEq example demonstrating fundamental charge equilibration usage.

This example shows how to:
1. Load molecular data from PDB and XML files
2. Set up a QEq calculation with basic parameters
3. Solve for equilibrium charges using the projected gradient method
4. Calculate energy and forces
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
        Dictionary containing molecular system data including positions, box,
        electronegativity (chi), hardness, eta, and initial charges.
    """
    # Load force field parameters from XML
    xml = XMLIO()
    xml.loadXML("qeq.xml")
    res = xml.parseResidues()
    ffinfo = xml.parseXML()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = np.array([a["type"] for a in res[0]["particles"]])

    # Load molecular structure from PDB
    pdb = app.PDBFile("qeq.pdb")
    dmfftop = DMFFTopology(from_top=pdb.topology)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positions = jnp.array(positions)
    a, b, c = dmfftop.getPeriodicBoxVectors()

    # Extract QEq parameters
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

    # Create data dictionary with all necessary parameters
    data_dict = {
        "n_atoms": n_atoms,
        "position": np.array(positions),
        "box": np.array([a._value, b._value, c._value]) * 10.0,  # Convert to Angstroms
        "chi": chi * energy_coeff,
        "hardness": hardness * energy_coeff,
        "eta": eta,
        "charge": charges,
    }
    return data_dict


def main():
    """
    Main function demonstrating basic QEq usage.
    """
    print("Running basic QEq example...")

    # Set up QEq parameters
    rcut = 8.0  # Cutoff radius in Angstroms
    ethresh = 1e-5  # Energy threshold for electrostatic interactions

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

    # Create QEq module
    module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Solve for equilibrium charges using projected gradient method
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
    )

    # Calculate forces
    forces = -calc_grads(energy, positions)

    # Print results
    print(f"QEq converged in {module.converge_iter} step(s)")
    print(f"Total energy: {energy.item():.6f} eV")
    print(f"Initial charges: {charges.tolist()}")
    print(f"Optimized charges: {q_opt.tolist()}")
    print(f"Total charge (should be ~0): {q_opt.sum().item():.6f}")
    print(f"Force norm: {torch.norm(forces).item():.6f} eV/Å")


if __name__ == "__main__":
    main()
