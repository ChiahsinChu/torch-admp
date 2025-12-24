# torch-admp

Automatic Differentiable Multipolar Polarizable (ADMP) in PyTorch backend

## Overview

`torch-admp` is a PyTorch implementation of the Automatic Differentiable Multipolar Polarizable (ADMP) framework. This package provides efficient implementations of various molecular dynamics force calculations including:

- Particle Mesh Ewald (PME) for electrostatic interactions
- Charge Equilibration (QEq) methods
- Polarizable electrode models
- Neighbor list management
- Optimization algorithms

## Installation

```bash
pip install torch-admp
```

For development:

```bash
git clone https://github.com/ChiahsinChu/torch-admp.git
cd torch-admp
pip install -e .[docs,test]
```

## Quick Start

```python
import torch
from torch_admp import QEqForceModule

# Initialize QEq module
qeq = QEqForceModule()

# Calculate charges and forces
positions = torch.randn(100, 3)  # 100 atoms
charges, forces = qeq(positions)
```

## Features

- **GPU Accelerated**: Built on PyTorch for efficient GPU computation
- **JIT Compilation**: Support for TorchScript compilation
- **Modular Design**: Clean separation of different force components
- **Extensible**: Easy to add new force modules

## Documentation

- [API Reference](api/torch_admp.md) - Complete API documentation
- [Examples](examples/pme.md) - Usage examples and tutorials

## Citation

If you use torch-admp in your research, please cite:

```bibtex
@software{torch_admp,
  author = {ChiahsinChu},
  title = {torch-admp: ADMP in PyTorch backend},
  url = {https://github.com/ChiahsinChu/torch-admp},
  year = {2024}
}
```

## License

This project is licensed under the LGPL-3.0-or-later License - see the [LICENSE](https://github.com/ChiahsinChu/torch-admp/blob/main/LICENSE) file for details.
