# QEq Example

This example demonstrates how to use the Charge Equilibration (QEq) implementation in torch-admp to calculate atomic charges and forces based on the electronegativity equalization principle.

## Overview

The QEq method calculates atomic charges by minimizing the system's electrostatic energy subject to charge conservation constraints. This example shows how to:

1. Load molecular system data from XML and PDB files
2. Set up QEq parameters (electronegativity, hardness, damping)
3. Solve the QEq equations using projected gradient optimization
4. Calculate forces with automatic differentiation

## Code Example

```python
import numpy as np
import torch
from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule
from torch_admp.utils import calc_grads
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
import openmm.app as app
import openmm.unit as unit
from scipy import constants

# Set default device and precision
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)


def load_test_data():
    """Load system data from XML and PDB files"""
    xml = XMLIO()
    xml.loadXML("qeq.xml")
    res = xml.parseResidues()
    ffinfo = xml.parseXML()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = np.array([a["type"] for a in res[0]["particles"]])

    # Load molecular structure
    pdb = app.PDBFile("qeq.pdb")
    dmfftop = DMFFTopology(from_top=pdb.topology)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    a, b, c = dmfftop.getPeriodicBoxVectors()

    n_atoms = dmfftop.getNumAtoms()
    eta = np.zeros([n_atoms])
    chi = np.zeros([n_atoms])
    hardness = np.zeros([n_atoms])

    # Extract QEq parameters from force field
    for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
        eta[types == _data["attrib"]["type"]] = float(_data["attrib"]["eta"])
        chi[types == _data["attrib"]["type"]] = float(_data["attrib"]["chi"])
        hardness[types == _data["attrib"]["type"]] = float(_data["attrib"]["J"])

    # Convert energy units from kJ/mol to eV
    j2ev = constants.physical_constants["joule-electron volt relationship"][0]
    energy_coeff = j2ev * constants.kilo / constants.Avogadro

    return {
        "n_atoms": n_atoms,
        "position": np.array(positions),
        "box": np.array([a._value, b._value, c._value]) * 10.0,
        "chi": chi * energy_coeff,
        "hardness": hardness * energy_coeff,
        "eta": eta,
        "charge": charges,
    }


# Load system data
data_dict = load_test_data()

# Convert to PyTorch tensors
positions = torch.tensor(data_dict["position"], requires_grad=True)
box = torch.tensor(data_dict["box"], requires_grad=False)
chi = torch.tensor(data_dict["chi"], requires_grad=False)
hardness = torch.tensor(data_dict["hardness"], requires_grad=False)
eta = torch.tensor(data_dict["eta"], requires_grad=False)
charges = torch.tensor(data_dict["charge"], requires_grad=False)

# Set up neighbor list
rcut = 8.0
nblist = TorchNeighborList(cutoff=rcut)
pairs = nblist(positions, box)
ds = nblist.get_ds()
buffer_scales = nblist.get_buffer_scales()

# Set up charge conservation constraint
constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
constraint_vals = torch.zeros(1, dtype=torch.float64)

# Solve QEq equations
ethresh = 1e-5
module = QEqForceModule(rcut=rcut, ethresh=ethresh)
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

print(f"QEq converges in {module.converge_iter} step(s)")
print(f"Final energy: {energy.item():.6f}")
print(f"Optimized charges: {q_opt}")
```

## Key Components

- **QEqForceModule**: Main QEq implementation with multiple optimization methods
- **solve_pgrad**: Projected gradient method for charge optimization
- **TorchNeighborList**: Efficient neighbor list for pairwise interactions
- **XMLIO/PDBFile**: Load force field parameters and molecular structure

## QEq Parameters

- **chi (χ)**: Atomic electronegativity values
- **hardness (J)**: Atomic hardness parameters
- **eta (η)**: Damping parameters for short-range interactions
- **constraint_matrix**: Charge conservation constraints

## Optimization Methods

The QEqForceModule supports multiple optimization algorithms:

1. **Projected Gradient (solve_pgrad)**: Robust for large systems
2. **Matrix Inversion (matinv_optimize)**: Fast for small systems
3. **Quadratic Optimization**: Efficient for certain parameterizations

## Performance Tips

- Use appropriate cutoff distance (typically 8-12 Å)
- Adjust ethresh for desired precision
- The projected gradient method is most stable for large systems
- GPU acceleration provides significant speedup for systems with >100 atoms
