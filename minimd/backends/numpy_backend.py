"""NumPy implementations of neighbour list and LJ forces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class NumpyNeighborList(NeighborList):
    """Verlet neighbour list with skin, built via brute-force O(N^2) scan."""

    def __init__(self) -> None:
        self._last_positions: NDArray[np.floating] | None = None
        self.pairs_i: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.pairs_j: NDArray[np.intp] = np.empty(0, dtype=np.intp)

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        n = positions.shape[0]
        r_list = r_cut + r_skin
        r_list_sq = r_list * r_list

        # all unique pairs
        ii, jj = np.triu_indices(n, k=1)
        dr = positions[jj] - positions[ii]

        # minimum image convention
        dr -= box * np.round(dr / box)

        dist_sq = np.sum(dr * dr, axis=1)
        mask = dist_sq < r_list_sq

        self.pairs_i = ii[mask]
        self.pairs_j = jj[mask]
        self._last_positions = positions.copy()

    def needs_rebuild(
        self,
        positions: NDArray[np.floating],
        r_skin: float,
    ) -> bool:
        if self._last_positions is None:
            return True
        disp = np.sqrt(np.max(np.sum((positions - self._last_positions) ** 2, axis=1)))
        # rebuild when the largest displacement exceeds half the skin
        return bool(disp > 0.5 * r_skin)


class NumpyLJForces(ForceEvaluator):
    """Lennard-Jones 12-6 forces with shifted potential with configurable sigma and epsilon."""

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
        forces = np.zeros((n, 3), dtype=np.float64)

        if pairs_i.size == 0:
            return forces, 0.0

        dr = positions[pairs_j] - positions[pairs_i]
        dr -= box * np.round(dr / box)
        dist_sq = np.sum(dr * dr, axis=1)

        r_cut_sq = r_cut * r_cut
        mask = (dist_sq > 0.0) & (dist_sq < r_cut_sq)
        dr = dr[mask]
        dist_sq = dist_sq[mask]
        pi = pairs_i[mask]
        pj = pairs_j[mask]

        sigma_sq = sigma * sigma
        inv_r2 = sigma_sq / dist_sq
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r12 = inv_r6 * inv_r6

        # shifted LJ potential: V(r) = 4*(1/r^12 - 1/r^6) - V(r_cut)
        inv_rc2 = sigma_sq / r_cut_sq
        inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
        inv_rc12 = inv_rc6 * inv_rc6
        v_shift = 4.0 * (inv_rc12 - inv_rc6)

        energy = float(epsilon * np.sum(4.0 * (inv_r12 - inv_r6) - v_shift))

        # f_over_r = -dV/dr / r = 24*eps/r^2 * [2*(sig/r)^12 - (sig/r)^6]
        # fij = f_over_r * dr points from i→j; force ON j is +fij, ON i is -fij
        f_over_r = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq
        fij = f_over_r[:, None] * dr  # (n_pairs, 3)

        np.add.at(forces, pj, fij)
        np.add.at(forces, pi, -fij)

        return forces, energy
