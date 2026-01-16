---
status: draft
author: AI Agent, Jia-Xin Zhu
last_updated: 2026-01-12
---

# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
