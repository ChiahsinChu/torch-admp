# QEq Examples

This directory contains comprehensive examples demonstrating various features and capabilities of the Charge Equilibration (QEq) implementation in torch-admp.

## Overview

QEq (Charge Equilibration) is a method for determining atomic charges in molecular systems by minimizing the electrostatic energy subject to constraints. The torch-admp implementation provides multiple optimization methods, constraint handling, and advanced features for efficient charge calculation.

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

## Running All Examples

To run all examples in sequence, use the provided script:

```bash
python run_all.py
```

This will execute each example in order and display the results.

## Data Files

The examples use the following data files:

- `qeq.pdb`: Molecular structure in PDB format
- `qeq.xml`: Force field parameters in XML format

These files contain the molecular system and QEq parameters (electronegativity, hardness, eta values) used in all examples.

## Requirements

All examples require the following packages:

- torch
- jax
- numpy
- openmm
- dmff
- scipy

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
    module = QEqForceModule(eps=eps)
    # Test performance
```

### Large Systems

For large systems, consider:

- Using matrix inversion method if memory allows
- JIT compilation for repeated calculations
- Appropriate cutoff values

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

## Advanced Topics

### Custom Constraints

Implement custom constraints using the constraint matrix:

```python
# Define constraint matrix A and values b
A = torch.tensor([...])  # Constraint coefficients
b = torch.tensor([...])  # Constraint values

# Calculate coefficient matrix
coeff_matrix = vector_projection_coeff_matrix(A)

# Solve with constraints
energy, charges = module.solve_pgrad(..., A, b, coeff_matrix=coeff_matrix)
```

### Performance Profiling

Profile your QEq calculations:

```python
import time

start = time.time()
energy, charges = module.solve_pgrad(...)
elapsed = time.time() - start

print(f"QEq solved in {elapsed:.4f} seconds")
print(f"Converged in {module.converge_iter} iterations")
```

## References

For more detailed information about QEq theory and implementation:

1. Rappé, A. K., & Goddard, W. A. (1991). Charge equilibration for molecular dynamics simulations. The Journal of Physical Chemistry, 95(8), 3358-3363.
