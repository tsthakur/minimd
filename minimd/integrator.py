"""Velocity Verlet integrator with optional SVR thermostat.

References:
    SVR thermostat: Bussi, Donadio, Parrinello, J. Chem. Phys. 126, 014101 (2007)
    Implementation closely follows i-PI: https://github.com/i-pi/i-pi/blob/v1/ipi/engine/thermostats.py
"""

from __future__ import annotations

import numpy as np

from minimd.config import Config
from minimd.interfaces import ForceEvaluator, NeighborList
from minimd.state import SimState


def velocity_verlet_step(
    state: SimState,
    config: Config,
    nlist: NeighborList,
    forces: ForceEvaluator,
) -> float:
    """One full velocity Verlet step. Returns potential energy."""
    dt = config.dt
    box = state.box
    inv_m = 1.0 / state.masses[:, None]

    # --- half-kick ---
    state.velocities += 0.5 * dt * state.forces * inv_m

    # --- drift ---
    state.positions += dt * state.velocities
    state.wrap_positions()

    # --- rebuild neighbor list if needed ---
    if nlist.needs_rebuild(state.positions, config.r_skin):
        nlist.build(state.positions, box, config.r_cut, config.r_skin)

    # --- new forces ---
    state.forces, pe = forces.compute(
        state.positions, box, nlist.pairs_i, nlist.pairs_j, config.r_cut
    )

    # --- half-kick ---
    state.velocities += 0.5 * dt * state.forces * inv_m

    return pe


def svr_thermostat(
    state: SimState,
    target_temp: float,
    tau: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    """Stochastic velocity rescaling.

    Applied after a full Verlet step. Returns the energy exchanged
    with the thermostat (positive = energy added to the system).

    The rescaling factor alpha is drawn so that the kinetic energy
    is sampled from the canonical distribution.
    """
    ke = state.kinetic_energy
    if ke == 0.0:
        return 0.0

    n_dof = state.n_dof
    # K_target = kB * T / 2 per degree of freedom (kB=1 in LJ units)
    k_target = 0.5 * target_temp

    # exponential damping factor
    et = np.exp(-0.5 * dt / tau)

    # stochastic term: sum of (n_dof - 1) squared standard normals
    r1 = rng.standard_normal()
    ndof_m1 = n_dof - 1
    if ndof_m1 % 2 == 0:
        rg = 2.0 * rng.gamma(ndof_m1 / 2)
    else:
        rg = 2.0 * rng.gamma((ndof_m1 - 1) / 2) + rng.standard_normal() ** 2

    # new kinetic energy ratio
    alpha2 = (
        et
        + k_target / ke * (1.0 - et) * (r1**2 + rg)
        + 2.0 * r1 * np.sqrt(k_target / ke * et * (1.0 - et))
    )

    # sign convention from i-PI
    alpha = np.sqrt(abs(alpha2))
    if r1 + np.sqrt(2.0 * ke / k_target * et / (1.0 - et)) < 0.0:
        alpha *= -1

    energy_delta = ke * (1.0 - alpha2)
    state.velocities *= alpha
    return float(energy_delta)
