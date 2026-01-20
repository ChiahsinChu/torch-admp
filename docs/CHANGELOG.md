---
status: draft
author: AI Agent, Jia-Xin Zhu
last_updated: 2026-01-20
---

# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.4] - 2026-01-20

### Added

- Docstring coverage badge to documentation
- DeepMD package integration for enhanced compatibility

### Changed

- Enhanced testing infrastructure
- Improved error handling in tests
- Updated precision handling for better compatibility with DeepMD

### Removed

- DeepMD code from tests (moved to integration)

### Fixed

- Device and dtype consistency in matinv_optimize function

## [1.1.3] - 2026-01-17

### Added

- Global precision handling for consistent tensor operations
- Comprehensive NumPy-style docstrings throughout the codebase
- Development tools:
  - Python test script (`scripts/python_test.sh`)
  - PyPI release automation script (`scripts/pypi_release.sh`)
- Enhanced documentation in `scripts/README.md`

### Changed

- Fixed dtype consistency and device placement across all tensors
- Replaced deprecated `torch.inverse` with `torch.linalg.inverse`
- Improved CI configuration to trigger documentation deployment only on master branch
- Updated README.md with latest project information
- Enhanced PME and QEq modules with better precision handling
- Improved neighbor list implementation with better error handling

## [1.1.2] - 2026-01-16

### Added

- Batch inference support for PME and QEq modules
- Enhanced BaseForceModule with standardized input tensor handling
- Shape verification for forward methods in BaseForceModule and derived classes
- Comprehensive documentation updates for tensor shapes and batch processing

### Changed

- Converted PME (Particle Mesh Ewald) to support batch inference
- Converted QEq (Charge Equilibration) to support batch inference
- Updated docstrings to specify tensor shapes for batch processing
- Improved test tolerance settings for numerical tests
- Temporarily removed polarizable electrode module (will be re-added in future release)

### Fixed

- Improved numerical stability in batch processing
- Enhanced error handling for tensor shape validation

## [1.1.1] - 2026-01-08

### Added

- Pypi release

### Fixed

- Bug fixes and stability improvements

## [1.1.0] - 2025-11-18

### Added

- Enhanced PME functionality
- Improved QEq methods
- Additional electrode simulation features

### Fixed

- Performance optimizations
- Memory usage improvements

## [1.0.0] - 2025-03-28

### Added

- Initial release of torch-admp
- Core functionality for:
  - PME (Particle Mesh Ewald) calculations
  - QEq (Charge Equilibration) methods
  - Electrode simulations
  - Neighbor list management
  - Spatial calculations
  - Optimization utilities
- Example scripts for PME and QEq
- Comprehensive test suite
- Documentation with API references and examples

## Related Documents

| Document Type     | Link                            | Description                        |
| ----------------- | ------------------------------- | ---------------------------------- |
| Project Context   | [AGENTS.md](./agents/AGENTS.md) | Project overview and current focus |
| API Documentation | [api/](./api/)                  | Detailed API references            |
| Examples          | [examples/](./examples/)        | Usage examples and tutorials       |
