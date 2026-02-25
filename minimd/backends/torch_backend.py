"""PyTorch backend for neighbour list and LJ forces.

Mirrors numpy_backend but uses torch.Tensor for all mathematical operations. 
CUDA is used if available.
Inputs/outputs remain NumPy arrays as required by the interface;
tensor conversion happens at the boundary only.
It assumes that a working installation of PyTorch is present,
does not check or enforce that during pip install because correct torch
installation can be non-trivial.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchNeighborList(NeighborList):
    """Verlet neighbour list built on PyTorch tensors (CPU or CUDA)."""

    def __init__(self) -> None:
        self._device = _get_device()
        self._last_positions: torch.Tensor | None = None
        self.pairs_i: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.pairs_j: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        pos = torch.tensor(positions, dtype=torch.float64, device=self._device)
        bx = torch.tensor(box, dtype=torch.float64, device=self._device)
        n = pos.shape[0]
        r_list_sq = (r_cut + r_skin) ** 2

        indices = torch.triu_indices(n, n, offset=1, device=self._device)
        ii, jj = indices[0], indices[1]
        dr = pos[jj] - pos[ii]
        dr -= bx * torch.round(dr / bx)
        dist_sq = (dr * dr).sum(dim=1)
        mask = dist_sq < r_list_sq

        self.pairs_i = ii[mask].cpu().numpy().astype(np.intp)
        self.pairs_j = jj[mask].cpu().numpy().astype(np.intp)
        self._last_positions = pos.clone()

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        if self._last_positions is None:
            return True
        pos = torch.tensor(positions, dtype=torch.float64, device=self._device)
        max_disp = ((pos - self._last_positions) ** 2).sum(dim=1).max().sqrt()
        return bool(max_disp.item() > 0.5 * r_skin)


class TorchLJForces(ForceEvaluator):
    """LJ forces via PyTorch tensor ops with minimum image convention."""

    def __init__(self) -> None:
        self._device = _get_device()

    def compute(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        pairs_i: NDArray[np.intp],
        pairs_j: NDArray[np.intp],
        r_cut: float,
        sigma: float = 1.0,
        epsilon: float = 1.0,
    ) -> tuple[NDArray[np.floating], float]:
        n = positions.shape[0]
        device = self._device

        if pairs_i.size == 0:
            return np.zeros((n, 3), dtype=np.float64), 0.0

        pos = torch.tensor(positions, dtype=torch.float64, device=device)
        bx = torch.tensor(box, dtype=torch.float64, device=device)
        pi = torch.tensor(pairs_i, dtype=torch.long, device=device)
        pj = torch.tensor(pairs_j, dtype=torch.long, device=device)

        dr = pos[pj] - pos[pi]
        dr -= bx * torch.round(dr / bx)
        dist_sq = (dr * dr).sum(dim=1)

        r_cut_sq = r_cut * r_cut
        mask = (dist_sq > 0.0) & (dist_sq < r_cut_sq)
        dr = dr[mask]
        dist_sq = dist_sq[mask]
        pi = pi[mask]
        pj = pj[mask]

        sigma_sq = sigma * sigma
        inv_r2 = sigma_sq / dist_sq
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r12 = inv_r6 * inv_r6

        inv_rc2 = sigma_sq / r_cut_sq
        inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
        inv_rc12 = inv_rc6 * inv_rc6
        v_shift = 4.0 * (inv_rc12 - inv_rc6)

        energy = float(epsilon * (4.0 * (inv_r12 - inv_r6) - v_shift).sum().item())

        f_over_r = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq
        fij = f_over_r.unsqueeze(1) * dr  # (n_pairs, 3)

        forces = torch.zeros((n, 3), dtype=torch.float64, device=device)
        forces.index_add_(0, pj, fij)
        forces.index_add_(0, pi, -fij)

        return forces.cpu().numpy(), energy
