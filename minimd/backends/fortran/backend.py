"""Fortran backend for neighbour list and LJ forces via f2py.

Compile the Fortran extension first:
    cd minimd/backends/fortran && python -m numpy.f2py -c -m _lj_fortran lj_fortran.f90
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList

try:
    from minimd.backends.fortran import _lj_fortran
except ImportError:
    _lj_fortran = None


def _check_compiled() -> None:
    if _lj_fortran is None:
        raise ImportError(
            "Fortran extension not compiled. Run:\n"
            "  cd minimd/backends/fortran && python -m numpy.f2py -c -m _lj_fortran lj_fortran.f90"
        )


class FortranNeighborList(NeighborList):
    """Neighbor list built in Fortran via f2py."""

    def __init__(self) -> None:
        _check_compiled()
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
        max_pairs = n * (n - 1) // 2
        pos = np.asfortranarray(positions, dtype=np.float64)
        bx = np.asarray(box, dtype=np.float64)

        pairs_i, pairs_j, n_pairs = _lj_fortran.build_neighbour_list(
            pos, bx, r_cut, r_skin, max_pairs,
        )
        self.pairs_i = np.asarray(pairs_i[:n_pairs], dtype=np.intp)
        self.pairs_j = np.asarray(pairs_j[:n_pairs], dtype=np.intp)
        self._last_positions = positions.copy()

    def needs_rebuild(
        self,
        positions: NDArray[np.floating],
        r_skin: float,
    ) -> bool:
        if self._last_positions is None:
            return True
        pos = np.asfortranarray(positions, dtype=np.float64)
        last = np.asfortranarray(self._last_positions, dtype=np.float64)
        result = _lj_fortran.check_needs_rebuild(pos, last, r_skin)
        return bool(result)


class FortranLJForces(ForceEvaluator):
    """LJ forces computed in Fortran via f2py with configurable sigma and epsilon."""

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
        if pairs_i.size == 0:
            return np.zeros((n, 3), dtype=np.float64), 0.0

        pos = np.asfortranarray(positions, dtype=np.float64)
        bx = np.asarray(box, dtype=np.float64)
        pi = np.asarray(pairs_i, dtype=np.intc)
        pj = np.asarray(pairs_j, dtype=np.intc)

        forces, energy = _lj_fortran.compute_lj_forces(
            pos, bx, pi, pj, r_cut, sigma, epsilon,
        )
        return np.asarray(forces, dtype=np.float64), float(energy)
