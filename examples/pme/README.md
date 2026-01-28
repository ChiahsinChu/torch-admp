# PME Examples for torch-admp

This directory contains comprehensive examples demonstrating the Particle Mesh Ewald (PME) implementation in torch-admp. Each example focuses on a specific aspect of the PME functionality.

## Overview

The PME method efficiently calculates long-range electrostatic interactions by splitting the calculation into real-space and reciprocal-space components. These examples showcase the full capabilities of the PME implementation in torch-admp.

## Available Examples

### 1. Basic PME Usage (`basic_pme.py`)

Demonstrates the fundamental usage of PME for periodic systems:

- Setting up a system with random positions and charges
- Creating a neighbor list for efficient pair calculations
- Using the CoulombForceModule to calculate energy and forces
- Computing forces using automatic differentiation

### 2. Advanced Parameters (`advanced_parameters.py`)

Shows how to use advanced PME parameters:

- Custom kappa (inverse screening length)
- Custom grid spacing
- Custom kmesh (grid points)
- Slab correction for 2D systems

### 3. Energy Components (`energy_components.py`)

Demonstrates how to access individual energy components:

- Real-space energy
- Reciprocal-space energy
- Self energy
- Non-neutral correction
- Slab correction energy

### 4. JIT Compilation (`jit_compilation.py`)

Shows how to use JIT compilation for performance optimization:

- Creating JIT-compiled modules
- Performance comparison between regular and JIT execution
- Saving and loading JIT modules

### 5. Batch Processing (`batch_processing.py`)

Demonstrates batch processing for multiple configurations:

- Setting up batched systems
- Calculating energies for multiple frames simultaneously
- Efficient processing of multiple configurations

### 6. Error Handling (`error_handling.py`)

Shows proper error handling and validation:

- Handling invalid parameters
- Validating input configurations
- Catching and reporting errors appropriately

### 7. 3D PBC vs 2D Slab Correction (`slab_vs_3d.py`)

Compares 3D periodic boundary conditions with 2D slab correction:

- Setting up slab geometries
- Comparing energy calculations
- Testing different slab axes

### 8. Ewald Parameters Setup (`setup_ewald.py`)

Shows how to use the setup_ewald_parameters utility function:

- OpenMM method for parameter calculation
- Gromacs method for parameter calculation
- Custom parameter configuration

## Running the Examples

### Individual Examples

To run a specific example:

```bash
cd examples/pme
python basic_pme.py  # Run basic PME example
python advanced_parameters.py  # Run advanced parameters example
# ... etc for other examples
```

### All Examples

To run all examples in sequence:

```bash
cd examples/pme
python run_all.py  # Runs all examples in order
```

## Requirements

- PyTorch
- NumPy
- torch-admp package

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

## Common Parameters

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

Each example provides detailed output for its specific demonstration, including:

- Energy values and breakdowns into components
- Performance comparisons
- Error handling demonstrations
- Parameter retrieval examples
- Comparisons between different PME configurations

## References

For more detailed information about the PME implementation, see:

- [PME API Documentation](../../docs/api/pme.md)
