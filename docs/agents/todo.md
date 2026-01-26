---
status: draft
author: AI Agent, Jia-Xin Zhu
last_updated: 2026-01-26
---

# torch-admp Development Plans

## Features

### Restore polarizable electrode package

- [x] reintroduce polarizable electrode module with batch support
- [x] add numerical uncertainty tests for polarizable electrode

### constant Q with finite field

- [ ] implement ffield with conq
- [ ] add electrode tests for ConqInterface3DBIAS

### Support batch inference

- [x] update docstrings for `BaseForceModule` and its derived classes, and specify the shape of input tensors
- [x] add shape verification for forward methods in for `BaseForceModule` and its derived classes
- [x] change required shapes of input tensors by adding the dimension of nframes
- [x] support multi-batch in PME
- [x] support multi-batch in QEq

## Documentation

### Set up Basic Vibe Coding Structure

- [x] Create docs/agents/ directory
- [x] Create AGENTS.md with project context
- [x] Create CHANGELOG in docs/ for the tagged versions

## Chores

- [x] update release in github and zenodo
