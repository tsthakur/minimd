"""C++ / MPI backend stubs for neighbor list and LJ forces.

TODO: implement via pybind11 or cython wrapped C++ with MPI domain decomposition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class CppMPINeighborList(NeighborList):
    """Neighbor list with MPI-distributed domain decomposition."""

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        raise NotImplementedError("CppMPINeighborList not yet implemented")

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        raise NotImplementedError("CppMPINeighborList not yet implemented")


class CppMPILJForces(ForceEvaluator):
    """LJ forces with MPI-distributed domain decomposition."""

    def compute(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        pairs_i: NDArray[np.intp],
        pairs_j: NDArray[np.intp],
        r_cut: float,
    ) -> tuple[NDArray[np.floating], float]:
        raise NotImplementedError("CppMPILJForces not yet implemented")
