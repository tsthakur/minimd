"""Abstract interfaces for the two pluggable subsystems.

Only neighbour-list building and force evaluation are designed as
swappable backends. Everything else is hardcoded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class NeighborList(ABC):
    """Build and cache a pair neighbour list under PBC."""

    pairs_i: NDArray[np.intp]  # cached i-indices from last build
    pairs_j: NDArray[np.intp]  # cached j-indices from last build

    @abstractmethod
    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        """Build pair list for all pairs within r_cut + r_skin.

        Stores results in self.pairs_i, self.pairs_j.
        Each pair appears once (j > i).
        """

    @abstractmethod
    def needs_rebuild(
        self,
        positions: NDArray[np.floating],
        r_skin: float,
    ) -> bool:
        """True if any atom moved more than r_skin/2 since last build."""


class ForceEvaluator(ABC):
    """Evaluate pair forces and potential energy."""

    @abstractmethod
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
        """Compute forces and potential energy.

        Args:
            positions: (N, 3) array.
            box: (3,) orthorhombic box lengths.
            pairs_i, pairs_j: neighbour pair indices from NeighborList.
            r_cut: cutoff radius (potential is shifted to zero at r_cut).
            sigma: LJ size parameter.
            epsilon: LJ well depth.

        Returns:
            forces: (N, 3) array.
            potential_energy: scalar.
        """
