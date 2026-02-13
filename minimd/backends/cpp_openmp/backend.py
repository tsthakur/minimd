"""C++ / OpenMP backend stubs for neighbour list and LJ forces.

TODO: implement via pybind11 or cython wrapped C++ with OpenMP threading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList

try:
    from minimd.backends.cpp_openmp import _lj_cpp_openmp
except ImportError:
    _lj_cpp_openmp = None


def _check_compiled() -> None:
    if _lj_cpp_openmp is None:
        raise ImportError(
            "C++ OpenMP extension not compiled. Run setup.py again."
        )
    

class CppOpenMPNeighborList(NeighborList):
    """Neighbor list built in C++ with OpenMP parallelism."""

    def __init__(self) -> None:
        self.pairs_i: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.pairs_j: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._last_positions: list[list[float]] | None = None

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        _check_compiled()

        self.pairs_i, self.pairs_j = _lj_cpp_openmp.build_neighbour_list(
            positions, box, r_cut, r_skin
        )
        self._last_positions = positions.copy()

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        if self._last_positions is None:
            return True
        return _lj_cpp_openmp.check_rebuild(positions, self._last_positions, r_skin)

class CppOpenMPLJForces(ForceEvaluator):
    """LJ forces computed in C++ with OpenMP parallelism."""

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
        _check_compiled()
        forces, potential_energy = _lj_cpp_openmp.compute_forces(
            positions, box, pairs_i, pairs_j, r_cut, sigma, epsilon
        )
        return forces, potential_energy
