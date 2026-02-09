"""CUDA kernel backend stubs for neighbor list and LJ forces.

TODO: implement via CuPy custom kernels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class CudaNeighborList(NeighborList):
    """Neighbor list built on GPU via CUDA kernels."""

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        raise NotImplementedError("CudaNeighborList not yet implemented")

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        raise NotImplementedError("CudaNeighborList not yet implemented")


class CudaLJForces(ForceEvaluator):
    """LJ forces computed via CUDA kernels."""

    def compute(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        pairs_i: NDArray[np.intp],
        pairs_j: NDArray[np.intp],
        r_cut: float,
    ) -> tuple[NDArray[np.floating], float]:
        raise NotImplementedError("CudaLJForces not yet implemented")
