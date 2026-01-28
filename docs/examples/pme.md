# PME Example

This example demonstrates the comprehensive use of the Particle Mesh Ewald (PME) implementation in torch-admp to calculate electrostatic interactions and forces.

## Overview

The PME method efficiently calculates long-range electrostatic interactions by splitting the calculation into real-space and reciprocal-space components. This comprehensive example shows how to:

1. Set up a basic system with random positions and charges
2. Use advanced PME parameters (slab correction, kappa, spacing, kmesh)
3. Access individual energy components (real, reciprocal, self, non-neutral)
4. Use JIT compilation for performance optimization
5. Process multiple configurations in a batch
6. Handle errors and validate inputs
7. Use getter methods to retrieve module parameters
8. Compare 3D PBC and 2D slab correction cases
9. Use the setup_ewald_parameters utility function

## Running the Example

To run the comprehensive PME example:

```bash
cd examples/pme
python run.py
```

## Key Features Demonstrated

### 1. Basic PME Usage

The example starts with a basic PME calculation for a periodic system:

```python
# System parameters
rcut = 6.0  # Cutoff distance in Angstroms
n_atoms = 100  # Number of atoms
ethresh = 1e-5  # Ewald precision threshold
l_box = 20.0  # Box length in Angstroms

# Generate random system
positions = np.random.rand(n_atoms, 3) * l_box
box = np.diag([l_box, l_box, l_box])
charges = np.random.uniform(-1.0, 1.0, (n_atoms))
charges -= charges.mean()  # Make system charge-neutral

# Create neighbor list and PME module
nblist = TorchNeighborList(cutoff=rcut)
pairs = nblist(positions, box)
ds = nblist.get_ds()
buffer_scales = nblist.get_buffer_scales()

# Calculate PME energy and forces
module = CoulombForceModule(rcut=rcut, ethresh=ethresh)
energy = module(positions, box, pairs, ds, buffer_scales, {"charge": charges})
forces = -calc_grads(energy, positions)
```

### 2. Advanced Parameters

The example demonstrates various advanced PME parameters:

#### Custom kappa (inverse screening length)

```python
custom_kappa = 0.3  # Å^-1
module_kappa = CoulombForceModule(rcut=rcut, ethresh=ethresh, kappa=custom_kappa)
```

#### Custom grid spacing

```python
custom_spacing = 1.0  # Å
module_spacing = CoulombForceModule(rcut=rcut, ethresh=ethresh, spacing=custom_spacing)
```

#### Custom kmesh

```python
custom_kmesh = [24, 24, 24]
module_kmesh = CoulombForceModule(rcut=rcut, ethresh=ethresh, kmesh=custom_kmesh)
```

#### Slab correction

```python
module_slab = CoulombForceModule(
    rcut=rcut,
    ethresh=ethresh,
    slab_corr=True,
    slab_axis=2,  # Apply correction along z-axis
)
```

### 3. Energy Component Access

The example shows how to access individual energy components:

```python
# After calculating energy
real_energy = module.real_energy
reciprocal_energy = module.reciprocal_energy
self_energy = module.self_energy
non_neutral_energy = module.non_neutral_energy
slab_corr_energy = module.slab_corr_energy  # If slab_corr=True

print(f"Real-space energy:     {real_energy.item():.6f} eV")
print(f"Reciprocal energy:     {reciprocal_energy.item():.6f} eV")
print(f"Self energy:            {self_energy.item():.6f} eV")
print(f"Non-neutral correction: {non_neutral_energy.item():.6f} eV")
print(f"Total energy:           {energy.item():.6f} eV")
```

### 4. JIT Compilation

The example demonstrates JIT compilation for performance optimization:

```python
# Create regular module
module = CoulombForceModule(rcut=rcut, ethresh=ethresh)

# Create JIT-compiled module
jit_module = torch.jit.script(module)

# Save JIT module for later use
torch.jit.save(jit_module, "pme_module_jit.pt")

# Load JIT module later
loaded_jit_module = torch.jit.load("pme_module_jit.pt")
```

### 5. Batch Processing

The example shows how to process multiple configurations in a batch:

```python
# Create batch of systems
batch_positions = torch.tensor(
    batch_positions, requires_grad=True
)  # (n_frames, n_atoms, 3)
batch_box = (
    torch.tensor(np.diag([l_box, l_box, l_box])).unsqueeze(0).repeat(n_frames, 1, 1)
)  # (n_frames, 3, 3)
batch_charges = torch.tensor(batch_charges)  # (n_frames, n_atoms)

# Calculate batch energies
batch_energies = module(
    batch_positions,
    batch_box,
    batch_pairs,
    batch_ds,
    batch_buffer_scales,
    {"charge": batch_charges},
)
```

### 6. Error Handling

The example demonstrates proper error handling:

```python
# Test invalid slab_axis
try:
    module = CoulombForceModule(rcut=6.0, ethresh=1e-5, slab_corr=True, slab_axis=3)
except (ValueError, AssertionError) as e:
    print(f"Correctly caught error: {type(e).__name__}: {e}")

# Test invalid ethresh
try:
    module = CoulombForceModule(rcut=6.0, ethresh=-1e-5)
except (ValueError, AssertionError) as e:
    print(f"Correctly caught error: {type(e).__name__}: {e}")
```

### 7. Getter Methods

The example shows how to use getter methods:

```python
# Create module with custom parameters
rcut = 6.0
sel = [10, 20, 30]  # Example selection list
module = CoulombForceModule(rcut=rcut, ethresh=1e-5, sel=sel)

# Use getter methods
retrieved_rcut = module.get_rcut()
retrieved_sel = module.get_sel()

print(f"Retrieved rcut: {retrieved_rcut} Å")
print(f"Retrieved sel: {retrieved_sel}")
```

### 8. 3D PBC vs 2D Slab Correction

The example compares 3D PBC and 2D slab correction cases:

```python
# 3D PBC calculation
module_3d = CoulombForceModule(rcut=rcut, ethresh=ethresh, slab_corr=False)
energy_3d = module_3d(positions, box_3d, pairs, ds, buffer_scales, {"charge": charges})

# 2D slab correction calculation (z-axis)
module_2d = CoulombForceModule(rcut=rcut, ethresh=ethresh, slab_corr=True, slab_axis=2)
energy_2d = module_2d(positions, box_2d, pairs, ds, buffer_scales, {"charge": charges})

print(f"3D PBC energy:           {energy_3d.item():.6f} eV")
print(f"2D slab correction energy: {energy_2d.item():.6f} eV")
print(f"Slab correction term:     {module_2d.slab_corr_energy.item():.6f} eV")
```

### 9. Ewald Parameters Setup

The example demonstrates the `setup_ewald_parameters` utility function:

```python
from torch_admp.pme import setup_ewald_parameters

# OpenMM method
kappa_omm, kx_omm, ky_omm, kz_omm = setup_ewald_parameters(
    rcut=rcut, box=box, threshold=ethresh, method="openmm"
)

# Gromacs method
kappa_gmx, kx_gmx, ky_gmx, kz_gmx = setup_ewald_parameters(
    rcut=rcut, box=box, threshold=ethresh, spacing=1.0, method="gromacs"
)
```

## Key Components

- **TorchNeighborList**: Efficient neighbor list construction for periodic systems
- **CoulombForceModule**: PME implementation for electrostatic calculations
- **calc_grads**: Utility function for computing gradients using automatic differentiation
- **setup_ewald_parameters**: Utility function to compute optimal PME parameters

## Performance Considerations

- The implementation is optimized for GPU acceleration
- Uses double precision (float64) for numerical accuracy
- JIT compilation can provide significant speedups for repeated calculations
- Batch processing is more efficient than processing configurations individually
- Neighbor list is updated automatically when positions change significantly

## Parameters

- `rcut`: Real-space cutoff distance (typically 6-12 Å)
- `ethresh`: Ewald convergence threshold (typically 1e-4 to 1e-6)
- `kappa`: Inverse screening length (computed automatically if not specified)
- `spacing`: Grid spacing for reciprocal space (affects kmesh if specified)
- `kmesh`: Number of grid points in each dimension (computed automatically if not specified)
- `slab_corr`: Whether to apply slab correction (default: False)
- `slab_axis`: Axis for slab correction (0=x, 1=y, 2=z, default: 2)
- `kspace`: Whether to include reciprocal space contribution (default: True)
- `rspace`: Whether to include real space contribution (default: True)
- `sel`: Selection list for neighbor list (default: None)

## Output

The example provides detailed output for each demonstration, including:

- Energy values and breakdowns into components
- Performance comparisons between regular and JIT execution
- Error handling demonstrations
- Parameter retrieval examples
- Comparisons between different PME configurations
