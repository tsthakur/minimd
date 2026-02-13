"""Pure-Python implementations of neighbour list and LJ forces.

All inner loops use plain Python arithmetic and the math module.
numpy is only used for array allocation and to satisfy the interface
signatures (positions, forces, etc. are still NDArrays).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class PythonNeighborList(NeighborList):
    """Verlet neighbour list with skin, built via brute-force O(N^2) scan.

    Pure-Python version: loops over all unique pairs in plain Python.
    """

    def __init__(self) -> None:
        self._last_positions: list[list[float]] | None = None
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
        bx, by, bz = float(box[0]), float(box[1]), float(box[2])

        list_i: list[int] = []
        list_j: list[int] = []

        for i in range(n):
            xi, yi, zi = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])
            for j in range(i + 1, n):
                dx = float(positions[j, 0]) - xi
                dy = float(positions[j, 1]) - yi
                dz = float(positions[j, 2]) - zi

                # minimum image convention
                dx -= bx * round(dx / bx)
                dy -= by * round(dy / by)
                dz -= bz * round(dz / bz)

                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < r_list_sq:
                    list_i.append(i)
                    list_j.append(j)

        self.pairs_i = np.array(list_i, dtype=np.intp)
        self.pairs_j = np.array(list_j, dtype=np.intp)

        # save positions for displacement check
        self._last_positions = [
            [float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])]
            for i in range(n)
        ]

    def needs_rebuild(
        self,
        positions: NDArray[np.floating],
        r_skin: float,
    ) -> bool:
        if self._last_positions is None:
            return True

        half_skin = 0.5 * r_skin
        n = positions.shape[0]

        for i in range(n):
            dx = float(positions[i, 0]) - self._last_positions[i][0]
            dy = float(positions[i, 1]) - self._last_positions[i][1]
            dz = float(positions[i, 2]) - self._last_positions[i][2]
            disp = math.sqrt(dx * dx + dy * dy + dz * dz)
            if disp > half_skin:
                return True
        return False


class PythonLJForces(ForceEvaluator):
    """Lennard-Jones 12-6 forces with shifted potential with configurable sigma and epsilon.

    Pure-Python version: loops over pairs in plain Python.
    """

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

        n_pairs = pairs_i.shape[0]
        if n_pairs == 0:
            return forces, 0.0

        bx, by, bz = float(box[0]), float(box[1]), float(box[2])
        r_cut_sq = r_cut * r_cut

        # shifted potential constant
        sigma_sq = sigma * sigma
        inv_rc2 = sigma_sq / r_cut_sq
        inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
        inv_rc12 = inv_rc6 * inv_rc6
        v_shift = 4.0 * (inv_rc12 - inv_rc6)

        energy = 0.0

        for p in range(n_pairs):
            i = int(pairs_i[p])
            j = int(pairs_j[p])

            dx = float(positions[j, 0]) - float(positions[i, 0])
            dy = float(positions[j, 1]) - float(positions[i, 1])
            dz = float(positions[j, 2]) - float(positions[i, 2])

            # minimum image
            dx -= bx * round(dx / bx)
            dy -= by * round(dy / by)
            dz -= bz * round(dz / bz)

            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq <= 0.0 or dist_sq >= r_cut_sq:
                continue

            inv_r2 = sigma_sq / dist_sq
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6

            # potential
            energy += epsilon * (4.0 * (inv_r12 - inv_r6) - v_shift)

            # force magnitude / r: 24*eps/r^2 * [2*(sig/r)^12 - (sig/r)^6]
            f_over_r = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq

            fx = f_over_r * dx
            fy = f_over_r * dy
            fz = f_over_r * dz

            forces[j, 0] += fx
            forces[j, 1] += fy
            forces[j, 2] += fz
            forces[i, 0] -= fx
            forces[i, 1] -= fy
            forces[i, 2] -= fz

        return forces, energy
