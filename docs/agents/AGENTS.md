---
status: draft
author: AI Agent, Jia-Xin Zhu
last_updated: 2026-01-12
---

# torch-admp Project Documentation

## Project Overview

`torch-admp` is a PyTorch implementation of the ADMP (Automatic Differentiable Multipolar Polarizable) module in DMFF (Differentiable Molecular Force Field) package. This package provides efficient implementations of various molecular dynamics force calculations including PME (Particle Mesh Ewald), QEq (Charge Equilibration), polarizable electrode models, and optimization algorithms.

## Key Components

### Core Modules

- **PME Module**: Electrostatic interaction calculations using Particle Mesh Ewald method
- **QEq Module**: Charge equilibration methods for dynamic charge distribution
- **Electrode Module**: Polarizable electrode models for electrochemical simulations
- **Neighbor List**: Efficient neighbor list construction for periodic systems
- **Optimizer**: Various optimization algorithms for charge and force calculations

### Architecture

- Built on PyTorch for GPU acceleration and automatic differentiation
- Modular design with clear separation of force components
- JIT compilation support for performance optimization
- Extensible framework for adding new force modules

## Development Guidelines

### Code Standards

- Follow PEP 8 Python style guidelines
- Use NumPy-style docstrings for all public functions and classes
- Include type hints for all function parameters and return values
- Maintain backward compatibility when possible

### Testing

- Unit tests for all core functionality
- Integration tests for complete workflows
- Performance benchmarks for critical paths
- Numerical accuracy validation against reference implementations

### Documentation Requirements

- All public APIs must have comprehensive docstrings
- Examples should be provided for major use cases
- Theory sections should explain the mathematical foundations
- Performance considerations should be documented

## Common Workflows

### Installation and Setup

```bash
# Basic installation
pip install torch-admp

# Development installation
git clone https://github.com/ChiahsinChu/torch-admp.git
pip install -e torch-admp[docs,test]
# DMFF is required for tests
pip install "DMFF @ git+https://github.com/ChiahsinChu/DMFF.git@devel"
```

### Performance Optimization

- Use GPU acceleration when available
- Enable JIT compilation for repeated calculations

## Validation Scenarios

### Basic Functionality Validation

- **Module Import**: Test all modules can be imported successfully
- **Device Compatibility**: Verify CPU and GPU functionality
- **JIT Compilation**: Test TorchScript compilation of key modules

### Scientific Validation

- **Consistency test**: Compare output energy/forces against reference data (e.g., analytical solutions, results from other packages)

## Critical Warnings

- **Numerical Precision**: Use double precision for charge calculations

## File Organization

```
torch_admp/
├── __init__.py          # Package initialization
├── base_force.py        # Base class for force modules
├── pme.py              # PME implementation
├── qeq.py              # QEq methods
├── electrode.py        # Polarizable electrode models
├── nblist.py           # Neighbor list generation
├── optimizer.py        # Optimization algorithms
├── recip.py            # Reciprocal space calculations
├── spatial.py          # Spatial transformations
└── utils.py            # Utility functions
```

## Changelog

Add Changelog section with the following format in the end of each file in the subfolders of this path.

| Date       | Changes         | Author |
| ---------- | --------------- | ------ |
| yyyy-mm-dd | Descriptions... | xxx    |
