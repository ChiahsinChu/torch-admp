---
status: draft
author: AI Agent, Jia-Xin Zhu
last_updated: 2026-01-12
---

# torch-admp Development Plans

## Features

### Support batch inference

- [x] update docstrings for `BaseForceModule` and its derived classes, and specify the shape of input tensors
- [x] add shape verification for forward methods in for `BaseForceModule` and its derived classes
- [x] change required shapes of input tensors by adding the dimension of nframes
- [x] support multi-batch in PME
- [x] support multi-batch in QEq
- [x] support multi-batch in polarizable electrode
- [ ] check numerical uncertainty of polarizable electrode and reduce the tolerance in `tests/test_electrode.py`

### constant Q with finite field

- [ ] implement ffield with conq
- [ ] update `tests/test_electrode.py::TestConqInterface3DBIAS`

## Documentation

### Set up Basic Vibe Coding Structure

- [x] Create docs/agents/ directory
- [x] Create AGENTS.md with project context
- [x] Create CHANGELOG in docs/ for the tagged versions

## Chores
