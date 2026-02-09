"""Fortran backend stubs for neighbor list and LJ forces.

TODO: implement via f2py-wrapped Fortran subroutines.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList


class FortranNeighborList(NeighborList):
    """Neighbor list built in Fortran via f2py."""

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        raise NotImplementedError("FortranNeighborList not yet implemented")

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        raise NotImplementedError("FortranNeighborList not yet implemented")


class FortranLJForces(ForceEvaluator):
    """LJ forces computed in Fortran via f2py."""

    def compute(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        pairs_i: NDArray[np.intp],
        pairs_j: NDArray[np.intp],
        r_cut: float,
    ) -> tuple[NDArray[np.floating], float]:
        raise NotImplementedError("FortranLJForces not yet implemented")
