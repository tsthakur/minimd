"""PyTorch backend stubs for neighbour list and LJ forces.

TODO: implement using torch. Tensor for GPU-accelerated MD.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class TorchNeighborList(NeighborList):
    """Neighbor list built on PyTorch tensors (GPU-capable)."""

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        raise NotImplementedError("TorchNeighborList not yet implemented")

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        raise NotImplementedError("TorchNeighborList not yet implemented")


class TorchLJForces(ForceEvaluator):
    """LJ force evaluation using PyTorch autograd or manual kernels."""

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
        raise NotImplementedError("TorchLJForces not yet implemented")
