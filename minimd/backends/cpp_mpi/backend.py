"""C++ backend with ghost atom approach for PBC.

Uses ghost atoms instead of minimum image convention, which naturally
extends to MPI domain decomposition where ghost atoms are communicated
between processors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minimd.interfaces import ForceEvaluator, NeighborList

try:
    from minimd.backends.cpp_mpi import _lj_cpp_mpi
except ImportError:
    _lj_cpp_mpi = None


def _check_compiled() -> None:
    if _lj_cpp_mpi is None:
        raise ImportError(
            "C++ MPI extension not compiled. Run: pip install -e ."
        )


class CppNeighborList(NeighborList):
    """Neighbor list built using ghost atoms for PBC."""

    def __init__(self) -> None:
        self.pairs_i: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.pairs_j: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._last_positions: NDArray[np.floating] | None = None
        self.ghost_map: NDArray[np.intp] | None = None
        self.ghost_shifts: NDArray[np.floating] | None = None

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        _check_compiled()
        r_list = r_cut + r_skin

        ghost_positions, self.ghost_map, self.ghost_shifts = (
            _lj_cpp_mpi.build_ghost_atoms(positions, box, r_list)
        )

        self.pairs_i, self.pairs_j = _lj_cpp_mpi.build_neighbour_list(
            positions, ghost_positions, self.ghost_map, r_list
        )
        self._last_positions = positions.copy()

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        if self._last_positions is None:
            return True
        return _lj_cpp_mpi.check_rebuild(positions, self._last_positions, r_skin)


class CppLJForces(ForceEvaluator):
    """LJ forces computed using ghost atoms (no minimum image convention)."""

    def __init__(self, nlist: CppNeighborList) -> None:
        self._nlist = nlist

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

        ghost_positions = _lj_cpp_mpi.update_ghost_positions(
            positions, self._nlist.ghost_map, self._nlist.ghost_shifts
        )

        forces, potential_energy = _lj_cpp_mpi.compute_forces(
            positions, ghost_positions, self._nlist.ghost_map,
            pairs_i, pairs_j, r_cut, sigma, epsilon
        )
        return forces, potential_energy


class CppMPINeighborList(NeighborList):
    """Neighbor list using MPI domain decomposition (but only in x direction).

    Each rank calls build() with the full positions array; the function
    partitions atoms into local and ghost sets via MPI halo exchange.
    """

    def __init__(self) -> None:
        _check_compiled()
        try:
            from mpi4py import MPI  # to trigger MPI_Init and register MPI_Finalize via atexit
        except ImportError as e:
            raise ImportError("mpi4py is required for the cpp_mpi backend") from e
        self.pairs_i: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.pairs_j: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self._last_positions: NDArray[np.floating] | None = None
        self.local_positions: NDArray[np.floating] | None = None
        self.ghost_positions: NDArray[np.floating] | None = None
        self._r_list: float = 0.0
        self._box: NDArray[np.floating] | None = None

    def build(
        self,
        positions: NDArray[np.floating],
        box: NDArray[np.floating],
        r_cut: float,
        r_skin: float,
    ) -> None:
        _check_compiled()
        self._r_list = r_cut + r_skin
        self._box = box

        self.local_positions, self.ghost_positions = (
            _lj_cpp_mpi.build_ghost_atoms_mpi(positions, box, self._r_list)
        )
        self.pairs_i, self.pairs_j = _lj_cpp_mpi.build_neighbour_list_mpi(
            self.local_positions, self.ghost_positions, self._r_list
        )
        self._last_positions = positions.copy()

    def needs_rebuild(self, positions: NDArray[np.floating], r_skin: float) -> bool:
        # Ghost atom indices in pairs_i/pairs_j must match the ghost_positions
        # array used in compute(). Unlike the serial path, the MPI path cannot 
        # have a cheap update_ghost_positions equivalent, so we rebuild every 
        # step to keep the two consistent.
        return True


class CppMPILJForces(ForceEvaluator):
    """LJ forces via MPI domain decomposition.

    Internally Allreduces local forces to the full [N x 3] array so the
    integrator is unchanged. All ranks return the same global forces/energy.
    """

    def __init__(self, nlist: CppMPINeighborList) -> None:
        self._nlist = nlist

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
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise ImportError("mpi4py is required for the cpp_mpi backend") from e

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        local_forces, local_pe = _lj_cpp_mpi.compute_forces_mpi(
            self._nlist.local_positions, self._nlist.ghost_positions,
            pairs_i, pairs_j, r_cut, sigma, epsilon
        )

        Lx = box[0]
        x_lo = rank * Lx / size
        x_hi = (rank + 1) * Lx / size
        local_idx = np.where((positions[:, 0] >= x_lo) & (positions[:, 0] < x_hi))[0]

        global_forces = np.zeros_like(positions)
        global_forces[local_idx] = local_forces
        comm.Allreduce(MPI.IN_PLACE, global_forces, op=MPI.SUM)

        global_pe = comm.allreduce(local_pe, op=MPI.SUM)

        return global_forces, global_pe
