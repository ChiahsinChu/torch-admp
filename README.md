# DMFF in PyTorch backend

[![codecov](https://codecov.io/gh/ChiahsinChu/torch_admp/graph/badge.svg?token=9PXNT5XB7C)](https://codecov.io/gh/ChiahsinChu/torch_admp)

> torch version of ADMP is initialized by [Zheng Cheng](https://github.com/zhengcheng233) (AISI).

This package implements the PME method (for monopoles) and the QEq method in [DMFF](https://github.com/deepmodeling/DMFF) with PyTorch, allowing not only GPU-accelerated calculation of PME/QEq methods but also further customization and extension of other PyTorch-based models.

## Installation

This package can be installed by:

```bash
git clone https://github.com/ChiahsinChu/torch-admp
pip install torch-admp
```

For the unit tests, you can install the package with the following command:

```bash
git clone https://github.com/ChiahsinChu/torch-admp
pip install torch-admp[test]
pip install "DMFF @ git+https://github.com/ChiahsinChu/DMFF.git@devel"
```

## To-do

- [ ] Add doc for usage
- [ ] Add unittest for QEq v.s. RuNNer

## Examples

###

## QEq

[openmm-torch](https://github.com/openmm/openmm-torch)

```python
from torch_admp.qeq import QEqAllForceModule
from torch_admp.pme import setup_ewald_parameters


kappa, kx, ky, kz = setup_ewald_parameters(rcut, box)
module = QEqAllForceModule(q0, chi, hardness, eta, rcut, kappa, (kx, ky, kz))
jit_module = torch.jit.script(module)
out = jit_module(positions, box)
```

| Notation   | Description                                 | Unit           |
| ---------- | ------------------------------------------- | -------------- |
| $q_i$      | charge of atom $i$                          | charge         |
| $z_i$      | z-coordinate of atom $i$                    | length         |
| $L_z$      | length of the simulation box in z-direction | length         |
| V          | volume of supercell                         | length $^3$    |
| $\alpha$   | Ewald screening parameter                   | length $^{-1}$ |
| $\sigma_i$ | Gaussian width of atom $i$                  | length         |

### Coulomb interaction [[ref]](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#coulomb-interaction-with-ewald-summation)

For non-periodic systems, the Coulomb interaction can be calculated directly:

$$
\begin{align}
E_{elec}=\sum_i \sum_{j\neq i} \frac{q_i q_j}{r_{ij}}.
\end{align}
$$

For periodic systems, we consider the Ewald summation under 3D-PBC:

$$
\begin{aligned}
E_{elec}=E_{real}+E_{recip}+E_{self}+E_{corr}.
\end{aligned}
$$

The real space interaction:

$$
\begin{align}
E_{real}=\frac{1}{2}\sum_{i,j}q_iq_j \frac{\text{erfc}(\alpha r_{ij})}{r_{ij}}.
\end{align}
$$

The reciprocal interaction:

$$
\begin{align}
E_{recip}=\frac{2\pi}{V}\sum_{k^\prime}\frac{S(k)^2}{k^2}\exp{\left(-\frac{k^2}{4\alpha^2}\right)},
\end{align}
$$

where the structural factor $S(k)$ is given by:

$$
\begin{align}
S(k)=\sum_i q_i\exp(ik\cdot r_i).
\end{align}
$$

The self interaction:

$$
\begin{align}
E_{self}=-\frac{\alpha}{\sqrt{\pi}}\sum_i q_i^2.
\end{align}
$$

Non-neutral correction (only with which the energy from 3D Ewald summation is independent with $\alpha$):

$$
\begin{aligned}
E=-\frac{\pi}{2V\alpha^2}Q_{tot}^2.
\end{aligned}
$$

### Gaussian damping [[ref]](http://dx.doi.org/10.1016/j.cplett.2010.10.010)

- `DampingForceModule`

While the standard Ewald summation is used to calculate the electrostatic interactions between point charges, an additional Gaussian damping term can be applied to adapt the Ewald summation for the Gaussian charges. The damping term is, in fact, a modification of interactions in the real space:

$$
\begin{align}
E=-\frac{1}{2}\sum_{i,j}q_iq_j \frac{\text{erfc}(\frac{r_{ij}}{2\sigma_{ij}})}{r_{ij}}+\frac{1}{2\sigma_i\sqrt{\pi}}\sum_i q_i^2,
\end{align}
$$

where

$$
\sigma_{ij} = \sqrt{\frac{\sigma_i^2 + \sigma_j^2}{2}}.
$$

### Slab correction [[ref]](https://pubs.aip.org/aip/jcp/article-abstract/131/9/094107/982953/)

- `SlabCorrForceModule`

When aiming for 2D periodic boundary conditions, the slab correction can be appiled [[ref]](https://docs.lammps.org/kspace_modify.html):

> This is done by treating the system as if it were periodic in z, but inserting empty volume between atom slabs and removing dipole inter-slab interactions so that slab-slab interactions are effectively turned off.

The energy for slab correction is given by:

$$
\begin{align}
E &= \frac{2\pi}{V} \left( M_z^2 - Q_{\text{tot}} \sum_i q_i z_i^2 + Q_{\text{tot}}^2\frac{L_z^2}{12} \right),
\end{align}
$$

where

$$
\begin{align}
M_z &= \sum_i q_i z_i, \\
Q_{\text{tot}} & = \sum_i q_i .
\end{align}
$$

Unlike lammps, where the empty volume can be inserted internally by the program, the users of this package are expected to **insert vacuum with sufficient thickness when building the models to avoid interactions between slabs**. Empirically, the thickness of the vacuum is suggested to be twice of the slab thickness.

### Chemical interaction [[ref]](https://pubs.acs.org/doi/abs/10.1021/j100161a070)

- `SiteForceModule`

In the QEq model, not only the electrostatic interaction but the chemical interaction are considered:

$$
\begin{align}
E=\sum_i \chi_i q_i+\frac{1}{2}\sum_i J_iq_i^2.
\end{align}
$$
