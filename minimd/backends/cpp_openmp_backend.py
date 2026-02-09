"""C++ / OpenMP backend stubs for neighbor list and LJ forces.

TODO: implement via pybind11 or cython wrapped C++ with OpenMP threading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class CppOpenMPNeighborList(NeighborList):
    """Neighbor list built in C++ with OpenMP parallelism."""

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        raise NotImplementedError("CppOpenMPNeighborList not yet implemented")

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        raise NotImplementedError("CppOpenMPNeighborList not yet implemented")


class CppOpenMPLJForces(ForceEvaluator):
    """LJ forces computed in C++ with OpenMP parallelism."""

    def compute(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        pairs_i: NDArray[np.intp],
        pairs_j: NDArray[np.intp],
        r_cut: float,
    ) -> tuple[NDArray[np.floating], float]:
        raise NotImplementedError("CppOpenMPLJForces not yet implemented")
