# SPDX-License-Identifier: LGPL-3.0-or-later
import torch


def get_recip_grid_vectors(
    box_inv: torch.Tensor,
    t_kmesh: torch.Tensor,
):
    """
    Compute reciprocal lattice vectors of grids

    Parameters
    ----------
    box_inv : torch.Tensor
        (3 * 3)-matrix for inv cell vectors
        inv_box = torch.linalg.inv(box)
    t_kmesh : torch.Tensor
        (3,)-shaped tensor [kx, ky, kz]

    Returns
    -------
    recip_grid_vectors: torch.Tensor
        (3 * 3)-matrix for reciprocal lattice vectors of grids
    """
    recip_grid_vectors = (t_kmesh.reshape(1, 3) * box_inv).transpose(0, 1)
    return recip_grid_vectors


def u_reference(
    positions: torch.Tensor,
    recip_grid_vectors: torch.Tensor,
    pme_order: int,
):
    """
    Each atom is meshed to dispersion_ORDER**3 points on the m-meshgrid.
    This function computes the xyz-index of the reference point, which is the point on the meshgrid just above atomic coordinates,
    and the corresponding values of xyz fractional displacements from real coordinate to the reference point.

    Parameters
    ----------
    positions : torch.Tensor
        Na * 3: positions of atoms
    recip_grid_vectors : torch.Tensor
        (3 * 3)-matrix for reciprocal lattice vectors of grids

    Returns
    -------
    m_u0: torch.Tensor
        N_a * 3 matrix, positions of the reference points of R_a on the m-meshgrid
    u0: torch.Tensor
        N_a * 3 matrix, (R_a - R_m)*a_star values
    """
    R_in_m_basis = torch.einsum("ij,kj->ki", recip_grid_vectors, positions)
    m_u0 = torch.ceil(R_in_m_basis).to(torch.int)
    u0 = (m_u0 - R_in_m_basis) + pme_order / 2
    return m_u0, u0


def sph_harmonics_GO(
    u0: torch.Tensor,
    shifts: torch.Tensor,
    pme_order: int,
):
    """
    Find out the value of spherical harmonics GRADIENT OPERATORS, assume the order is:
    00, 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, ...
    Currently supports lmax <= 2

    Parameters
    ----------
    u0 : torch.Tensor
        (N_a * 3)-matrix containing all positions
    recip_grid_vectors : torch.Tensor
        (3 * 3)-matrix for reciprocal lattice vectors of grids

    Returns
    -------
    harmonics: torch.Tensor
        a Na * (6**3) * (l+1)^2 matrix, STGO operated on theta,
        evaluated at 6*6*6 integer points about reference points m_u0
    """
    n_mesh = int(pme_order**3)
    N_a = u0.shape[0]

    # mesh points around each site
    u = u0[:, None, :] + shifts
    u_reshape = torch.reshape(u, (N_a * n_mesh, 3))
    # bspline may have little different value
    M_u = bspline(u_reshape)
    theta = torch.prod(M_u, dim=-1)
    return theta.reshape(N_a, n_mesh, 1)


def bspline(u: torch.Tensor):
    """
    Computes the cardinal B-spline function
    """
    return torch.where(
        (u >= 0.0) & (u < 1.0),
        u**5 / 120,
        torch.where(
            u < 2.0,
            u**5 / 120 - (u - 1) ** 5 / 20,
            torch.where(
                u < 3.0,
                u**5 / 120 + (u - 2) ** 5 / 8 - (u - 1) ** 5 / 20,
                torch.where(
                    u < 4.0,
                    u**5 / 120
                    - (u - 3) ** 5 / 6
                    + (u - 2) ** 5 / 8
                    - (u - 1) ** 5 / 20,
                    torch.where(
                        u < 5.0,
                        u**5 / 24
                        - u**4
                        + 19 * u**3 / 2
                        - 89 * u**2 / 2
                        + 409 * u / 4
                        - 1829 / 20,
                        torch.where(
                            u < 6.0,
                            -(u**5) / 120
                            + u**4 / 4
                            - 3 * u**3
                            + 18 * u**2
                            - 54 * u
                            + 324 / 5,
                            torch.zeros_like(u),
                        ),
                    ),
                ),
            ),
        ),
    )


def Q_m_peratom(
    charges: torch.Tensor,
    sph_harms: torch.Tensor,
    pme_order: int,
):
    """
    Computes <R_t|Q>. See eq. (49) of https://doi.org/10.1021/ct5007983

    Inputs:
        Q:
            N_a * (l+1)**2 matrix containing global frame multipole moments up to lmax,
        sph_harms:
            N_a, 216, (l+1)**2
        lmax:
            int: maximal L

    Output:
        Q_m_pera:
            N_a * 216 matrix, values of theta evaluated on a 6 * 6 block about the atoms
    """
    n_mesh = int(pme_order**3)
    N_a = sph_harms.shape[0]
    Q_dbf = torch.atleast_2d(charges)[:, 0:1]
    Q_m_pera = torch.sum(Q_dbf[:, None, :] * sph_harms, dim=2)
    assert Q_m_pera.shape == (N_a, n_mesh)
    return Q_m_pera


def Q_mesh_on_m(
    Q_mesh_pera: torch.Tensor,
    m_u0: torch.Tensor,
    t_kmesh: torch.Tensor,
    shifts: torch.Tensor,
):
    """
    Reduce the local Q_m_peratom into the global mesh

    Input:
        Q_mesh_pera, m_u0, N

    Output:
        Q_mesh:
            Nx * Ny * Nz matrix
    """
    indices_arr = torch.fmod(
        m_u0[:, None, :] + shifts + t_kmesh[None, None, :] * 10, t_kmesh[None, None, :]
    )
    Q_mesh = torch.zeros(
        t_kmesh[0] * t_kmesh[1] * t_kmesh[2],
        device=t_kmesh.device,
        dtype=Q_mesh_pera.dtype,
    )
    indices_0 = indices_arr[:, :, 0].flatten()
    indices_1 = indices_arr[:, :, 1].flatten()
    indices_2 = indices_arr[:, :, 2].flatten()
    flat_indices = (
        indices_0 * t_kmesh[1] * t_kmesh[2] + indices_1 * t_kmesh[2] + indices_2
    )
    Q_mesh.index_add_(0, flat_indices, Q_mesh_pera.view(-1))
    Q_mesh = Q_mesh.reshape(
        t_kmesh[0].item(),
        t_kmesh[1].item(),
        t_kmesh[2].item(),
    )

    return Q_mesh


def spread_charges(
    positions: torch.Tensor,
    box_inv: torch.Tensor,
    charges: torch.Tensor,
    t_kmesh: torch.Tensor,
    shifts: torch.Tensor,
    pme_order: int,
):
    """
    This is the high level wrapper function, in charge of spreading the charges/multipoles on grid

    Parameters
    ----------
    positions : torch.Tensor
        Na * 3: positions of atoms
    box : torch.Tensor
        3 * 3: cell vectors
    charges : torch.Tensor
        Na * (lmax+1)**2: the multipole of each atomic site in global frame

    Returns
    -------
    torch.Tensor
        Nx * Ny * Nz: the meshed multipoles
    Output:
        Q_mesh:
            K1 * K2 * K3: the meshed multipoles

    """
    recip_grid_vectors = get_recip_grid_vectors(box_inv, t_kmesh)
    # For each atom, find the reference mesh point, and u position of the site
    m_u0, u0 = u_reference(positions, recip_grid_vectors, pme_order)
    # find out the STGO values of each grid point
    sph_harms = sph_harmonics_GO(u0, shifts, pme_order)
    # find out the local meshed values for each site
    Q_mesh_pera = Q_m_peratom(charges, sph_harms, pme_order)
    return Q_mesh_on_m(Q_mesh_pera, m_u0, t_kmesh, shifts)


def setup_kpts_integer(
    t_kmesh: torch.Tensor,
):
    """
    Outputs:
        kpts_int:
            n_k * 3 matrix, n_k = N[0] * N[1] * N[2]
    """
    kx, ky, kz = [
        torch.roll(
            torch.arange(
                torch.div(-(t_kmesh[i] - 1), 2, rounding_mode="floor"),
                torch.div(t_kmesh[i] + 1, 2, rounding_mode="floor"),
                device=t_kmesh.device,
            ),
            shifts=[-(int(t_kmesh[i].item()) - 1) // 2],
        )
        for i in range(3)
    ]
    kpts_int = torch.hstack(
        [ki.flatten()[:, None] for ki in torch.meshgrid(kx, ky, kz, indexing="ij")]
    )
    return kpts_int


def setup_kpts(box_inv, kpts_int):
    """
    This function sets up the k-points used for reciprocal space calculations

    Input:
        box_inv:
            3 * 3, three axis arranged in rows
        kpts_int:
            n_k * 3 matrix

    Output:
        kpts:
            4 * K, K=K1*K2*K3, contains kx, ky, kz, k^2 for each kpoint
    """
    # in this array, a*, b*, c* (without 2*pi) are arranged in column
    # K * 3, coordinate in reciprocal space
    kpts = 2 * torch.pi * torch.matmul(kpts_int.double(), box_inv)
    ksq = torch.sum(kpts**2, dim=1)
    # 4 * K
    kpts = torch.hstack((kpts, ksq[:, None])).transpose(0, 1)
    return kpts
