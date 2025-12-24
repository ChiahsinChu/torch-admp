# PME Example

This example demonstrates how to use the Particle Mesh Ewald (PME) implementation in torch-admp to calculate electrostatic interactions and forces.

## Overview

The PME method efficiently calculates long-range electrostatic interactions by splitting the calculation into real-space and reciprocal-space components. This example shows how to:

1. Set up a system with random positions and charges
2. Create a neighbor list for efficient pair calculations
3. Use the CoulombForceModule to calculate energy and forces
4. Compute forces using automatic differentiation

## Code Example

```python
import numpy as np
import torch
from torch_admp.nblist import TorchNeighborList
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float64)

# System parameters
rcut = 6.0  # Cutoff distance
n_atoms = 500  # Number of atoms
ethresh = 1e-4  # Ewald precision threshold
l_box = 20.0  # Box length

# Generate random system
positions = np.random.rand(n_atoms, 3) * l_box
box = np.diag([l_box, l_box, l_box])
charges = np.random.uniform(-1.0, 1.0, (n_atoms))
charges -= charges.mean()  # Make system charge-neutral

# Convert to PyTorch tensors
positions = torch.tensor(positions, requires_grad=True)
box = torch.tensor(box, requires_grad=False)
charges = torch.tensor(charges, requires_grad=False)

# Create neighbor list
nblist = TorchNeighborList(cutoff=rcut)
pairs = nblist(positions, box)
ds = nblist.get_ds()
buffer_scales = nblist.get_buffer_scales()

# Calculate PME energy and forces
module = CoulombForceModule(rcut=rcut, ethresh=ethresh)
energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
forces = -calc_grads(energy, positions)

print(f"Energy: {energy.item():.6f}")
print(f"Forces shape: {forces.shape}")
```

## Key Components

- **TorchNeighborList**: Efficient neighbor list construction for periodic systems
- **CoulombForceModule**: PME implementation for electrostatic calculations
- **calc_grads**: Utility function for computing gradients using automatic differentiation

## Performance Considerations

- The implementation is optimized for GPU acceleration
- Uses double precision (float64) for numerical accuracy
- Neighbor list is updated automatically when positions change significantly

## Parameters

- `rcut`: Real-space cutoff distance (typically 6-12 Å)
- `ethresh`: Ewald convergence threshold (typically 1e-4 to 1e-6)
- Box vectors should be orthogonal for optimal performance
