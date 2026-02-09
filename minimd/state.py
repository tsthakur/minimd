"""Simulation state: positions, velocities, forces, box."""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass
class SimState:
    """Complete microstate of the system.

    All arrays are (N, 3) float64. Units are LJ reduced units throughout:
    sigma = epsilon = mass = 1, so kB*T is in units of epsilon.
    """

    symbols: list[str]
    positions: NDArray[np.floating]   # (N, 3)
    velocities: NDArray[np.floating]  # (N, 3)
    forces: NDArray[np.floating]      # (N, 3)
    box: NDArray[np.floating]         # (3,) orthorhombic box lengths
    masses: NDArray[np.floating]      # (N,)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    @property
    def n_dof(self) -> int:
        """Degrees of freedom (3N minus 3 for COM momentum removal)."""
        return 3 * self.n_atoms - 3

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy: sum(0.5 * m * v^2)."""
        return 0.5 * float(np.sum(self.masses[:, None] * self.velocities**2))

    @property
    def temperature(self) -> float:
        """Instantaneous temperature from equipartition: T = 2*KE / n_dof."""
        n = self.n_dof
        if n <= 0:
            return 0.0
        return 2.0 * self.kinetic_energy / n

    def remove_com_velocity(self) -> None:
        """Zero the centre-of-mass velocity."""
        total_mass = self.masses.sum()
        com_vel = (self.masses[:, None] * self.velocities).sum(axis=0) / total_mass
        self.velocities -= com_vel

    def wrap_positions(self) -> None:
        """Wrap positions into the primary periodic box."""
        self.positions -= self.box * np.floor(self.positions / self.box)
