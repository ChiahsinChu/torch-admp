# QEq Examples

This directory contains comprehensive examples demonstrating the full capabilities of QEq (Charge Equilibration) module in torch-admp. The examples cover everything from basic usage to advanced features like JIT compilation, constraint handling, and Hessian analysis.

## Overview

Charge Equilibration (QEq) is a method for determining atomic charges in molecular systems by minimizing electrostatic energy subject to constraints. The torch-admp implementation provides multiple optimization methods, constraint handling, and advanced features for efficient charge calculation.

## Available Examples

### 1. Basic QEq Usage (`basic_qeq.py`)

Demonstrates fundamental QEq usage including:

- Loading molecular data from PDB and XML files
- Setting up QEq calculation with basic parameters
- Solving for equilibrium charges using projected gradient method
- Calculating energy and forces

**Key Features:**

- Basic QEq setup and execution
- Charge conservation constraints
- Energy and force calculation

**Usage:**

```bash
python basic_qeq.py
```

### 2. Matrix Inversion Method (`matrix_inversion.py`)

Shows how to use the matrix inversion method for QEq calculations:

- Direct solution of linear system without iterative optimization
- Comparison with projected gradient method
- Hessian matrix analysis

**Key Features:**

- Matrix inversion vs projected gradient comparison
- Diagonal Hessian elements
- Fermi level calculation

**Usage:**

```bash
python matrix_inversion.py
```

### 3. Optimization Methods (`optimization_methods.py`)

Demonstrates different optimization methods available for QEq:

- LBFGS optimization
- Quadratic optimization
- Performance comparison between methods
- Testing with different initial guesses

**Key Features:**

- Multiple optimization algorithms
- Convergence behavior analysis
- Initial guess sensitivity

**Usage:**

```bash
python optimization_methods.py
```

### 4. Advanced Parameters (`advanced_parameters.py`)

Shows how to use various advanced QEq parameters:

- `max_iter`: Maximum number of iterations
- `eps`: Convergence threshold
- `damping`: Gaussian damping toggle
- `kspace`/`rspace`: Reciprocal/real space control

**Key Features:**

- Parameter tuning demonstrations
- Performance impact analysis
- Submodel configuration

**Usage:**

```bash
python advanced_parameters.py
```

### 5. Convergence Criteria (`convergence_criteria.py`)

Demonstrates convergence customization and monitoring:

- Different convergence thresholds
- Convergence history tracking
- Method-specific convergence behavior
- Line search parameter tuning

**Key Features:**

- Convergence threshold testing
- Iteration monitoring
- Performance optimization

**Usage:**

```bash
python convergence_criteria.py
```

### 6. Batch Processing (`batch_processing.py`)

Shows how to efficiently process multiple configurations:

- Batch processing workflows
- Trajectory processing
- Force calculation for multiple frames
- Performance optimization strategies

**Key Features:**

- Multiple configuration handling
- Trajectory analysis
- Efficient batch workflows

**Usage:**

```bash
python batch_processing.py
```

### 7. JIT Compilation (`jit_compilation.py`)

Demonstrates JIT compilation for performance optimization:

- JIT vs regular performance comparison
- Method-specific JIT optimization
- Warm-up effects
- Batch processing with JIT

**Key Features:**

- Performance benchmarking
- JIT compilation benefits
- Optimization strategies

**Usage:**

```bash
python jit_compilation.py
```

### 8. Hessian Calculation (`hessian_calculation.py`)

Shows how to calculate and analyze the Hessian matrix:

- Hessian matrix calculation
- Eigenvalue analysis
- Structure analysis
- Parameter effects on Hessian

**Key Features:**

- Hessian matrix properties
- Positive definiteness checking
- Distance dependence analysis
- Parameter sensitivity

**Usage:**

```bash
python hessian_calculation.py
```

<!--
### 9. Constraint Handling (`constraint_handling.py`)

Demonstrates various constraint types and handling:

- Charge conservation constraints
- Fixed charge constraints
- Group charge constraints
- Vector projection coefficient matrices

**Key Features:**

- Multiple constraint types
- Constraint matrix properties
- Vector projection mathematics
- Constraint verification

**Usage:**

```bash
python constraint_handling.py
``` -->

## Running All Examples

To run all examples in sequence, use the provided script:

```bash
python run_all.py
```

This will execute each example in order and display the results, providing a comprehensive demonstration of all QEq features.

## Basic Usage

Here's a minimal example of QEq usage:

```python
import torch
from torch_admp.qeq import QEqForceModule
from torch_admp.nblist import TorchNeighborList

# Create QEq module
module = QEqForceModule(rcut=8.0, ethresh=1e-5)

# Calculate neighbor list
nblist = TorchNeighborList(cutoff=8.0)
pairs = nblist(positions, box)
ds = nblist.get_ds()
buffer_scales = nblist.get_buffer_scales()

# Set up constraints (total charge = 0)
constraint_matrix = torch.ones([1, n_atoms], dtype=torch.float64)
constraint_vals = torch.zeros(1, dtype=torch.float64)

# Solve for charges using projected gradient method
energy, charges = module.solve_pgrad(
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
```

## Advanced Features

### Matrix Inversion Method

For direct solution without iterative optimization:

```python
# Solve using matrix inversion
energy, charges, diag_hessian, fermi = module.solve_matrix_inversion(
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
```

### JIT Compilation

For performance optimization with repeated calculations:

```python
# Create JIT-compiled module
jit_module = torch.jit.script(QEqForceModule(rcut=8.0, ethresh=1e-5))

# Use with pgrad_optimize function
from torch_admp.qeq import pgrad_optimize

energy, charges = pgrad_optimize(
    jit_module,
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
```

### Custom Constraints

Implement custom constraints using constraint matrix:

```python
# Define constraint matrix A and values b
A = torch.tensor([...])  # Constraint coefficients
b = torch.tensor([...])  # Constraint values

# Calculate coefficient matrix
from torch_admp.utils import vector_projection_coeff_matrix

coeff_matrix = vector_projection_coeff_matrix(A)

# Solve with constraints
energy, charges = module.solve_pgrad(..., A, b, coeff_matrix=coeff_matrix)
```

### Hessian Analysis

For analyzing the energy landscape curvature:

```python
# Calculate Hessian matrix
hessian = module.calc_hessian(
    positions, box, chi, hardness, eta, pairs, ds, buffer_scales
)

# Analyze eigenvalues
eigenvalues = torch.linalg.eigvalsh(hessian)
print(f"Condition number: {eigenvalues.max() / eigenvalues.min()}")
```

## Performance Tips

1. **Use JIT Compilation**: For repeated calculations, use JIT compilation for significant speedup
2. **Choose Appropriate Method**: LBFGS is generally faster, quadratic may be more robust
3. **Tune Convergence**: Adjust `eps` and `max_iter` for your specific system
4. **Batch Processing**: Process multiple configurations together when possible
5. **Constraint Optimization**: Use the most specific constraints needed

## Common Use Cases

### Molecular Dynamics

Use `batch_processing.py` as a template for processing MD trajectories:

```python
# Process each frame
for frame in trajectory:
    charges = solve_qeq(frame.positions, frame.box)
    # Use charges for force calculation
```

### Parameter Optimization

Use `advanced_parameters.py` and `convergence_criteria.py` to optimize parameters:

```python
# Test different convergence thresholds
for eps in [1e-4, 1e-5, 1e-6]:
    module = QEqForceModule(eps=eps, ...)
    # Test performance
```

### Large Systems

For large systems, consider:

- Using matrix inversion method if memory allows
- JIT compilation for repeated calculations
- Appropriate cutoff values

## API Reference

### QEqForceModule

Main class for QEq calculations.

**Parameters:**

- `rcut` (float): Cutoff radius for short-range interactions
- `ethresh` (float, optional): Energy threshold for electrostatic interactions
- `max_iter` (int, optional): Maximum iterations for optimization
- `eps` (float, optional): Convergence threshold
- `damping` (bool, optional): Whether to include Gaussian damping

**Methods:**

- `solve_pgrad()`: Solve using projected gradient method
- `solve_matrix_inversion()`: Solve using matrix inversion
- `calc_hessian()`: Calculate Hessian matrix
- `func_energy()`: Calculate energy for given charges

### Utility Functions

- `vector_projection()`: Project vector onto constraint subspace
- `vector_projection_coeff_matrix()`: Calculate coefficient matrix for projection
- `pgrad_optimize()`: Function for projected gradient optimization
- `calc_pgrads()`: Calculate projected gradients

## Troubleshooting

### Common Issues

1. **Non-convergence**:
   - Increase `max_iter`
   - Relax `eps` threshold
   - Try different optimization method

2. **Memory issues**:
   - Reduce cutoff radius
   - Use projected gradient instead of matrix inversion
   - Process in smaller batches

3. **Incorrect charges**:
   - Check constraint setup
   - Verify parameter units
   - Validate input data

### Performance Issues

1. **Slow convergence**:
   - Use better initial guesses
   - Try LBFGS method
   - Enable JIT compilation

2. **Large memory usage**:
   - Reduce system size
   - Use appropriate cutoffs
   - Disable unnecessary submodels

## References

For more detailed information about QEq theory and implementation:

1. Rappé, A. K., & Goddard, W. A. (1991). Charge equilibration for molecular dynamics simulations. The Journal of Physical Chemistry, 95(8), 3358-3363.

2. Chen, J., & Martínez, T. J. (2007). Charge equilibration: A variational approach. The Journal of Chemical Physics, 126(14), 144107.

3. torch-admp documentation: [https://github.com/ChiahsinChu/torch-admp](https://github.com/ChiahsinChu/torch-admp)
