# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional

import torch


@torch.jit.script
def pbc_shift(positions: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """
    wrap positions into the box

    Parameters
    ----------
    positions : torch.Tensor
        N * 3, positions of the particles
    box : torch.Tensor
        3 * 3, box vectors arranged in rows

    Returns
    -------
    wrapped_positions: torch.Tensor
        N * 3, wrapped positions
    """
    # box_inv = torch.inverse(box + torch.eye(3, device=positions.device) * 1e-36)
    box_inv = torch.linalg.inv(box)
    unshifted_positions = torch.matmul(positions, box_inv)
    wrapped_positions = unshifted_positions - torch.floor(unshifted_positions + 0.5)
    return torch.matmul(wrapped_positions, box)


@torch.jit.script
def ds_pairs(
    positions: torch.Tensor,
    pairs: torch.Tensor,
    box: Optional[torch.Tensor] = None,
    pbc_flag: bool = True,
) -> torch.Tensor:
    indices = torch.tile(pairs[:, 0].reshape(-1, 1), [1, 3])
    pos1 = torch.gather(positions, 0, indices)
    indices = torch.tile(pairs[:, 1].reshape(-1, 1), [1, 3])
    pos2 = torch.gather(positions, 0, indices)
    dr = pos1 - pos2
    if pbc_flag:
        assert box is not None, "Box should be provided for periodic system."
        dr = pbc_shift(dr, box)
    ds = torch.linalg.norm(dr + 1e-64, dim=1)  # add eta to avoid division by zero
    return ds


@torch.jit.script
def build_quasi_internal(
    r1: torch.Tensor, r2: torch.Tensor, dr: torch.Tensor, norm_dr: torch.Tensor
) -> torch.Tensor:
    """
    Build the quasi-internal frame between a pair of sites
    In this frame, the z-axis is pointing from r2 to r1

    Parameters
    ----------
    r1 : torch.Tensor
        N * 3, positions of the first vector
    r2 : torch.Tensor
        N * 3, positions of the second vector
    dr : torch.Tensor
        N * 3, vector pointing from r1 to r2
    norm_dr : torch.Tensor
        (N,), distances between r1 and r2

    Returns
    -------
    local_frames:
        N * 3 * 3: local frames, three axes arranged in rows
    """
    # n x 3
    vectorZ = dr / norm_dr.reshape(-1, 1)
    vectorX = torch.where(
        torch.logical_or(r1[1] != r2[1], r1[2] != r2[2]),
        vectorZ + torch.tensor([1.0, 0.0, 0.0], device=vectorZ.device),
        vectorZ + torch.tensor([0.0, 1.0, 0.0], device=vectorZ.device),
    )

    dot_xz = torch.matmul(vectorZ, vectorX)
    vectorX = vectorX - vectorZ * dot_xz
    vectorX = vectorX / torch.norm(vectorX)
    vectorY = torch.linalg.cross(vectorZ, vectorX)
    return torch.stack([vectorX, vectorY, vectorZ])
