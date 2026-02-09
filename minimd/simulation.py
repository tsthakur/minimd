"""Main simulation driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minimd.config import Config
from minimd.integrator import svr_thermostat, velocity_verlet_step
from minimd.interfaces import ForceEvaluator, NeighborList
from minimd.io import read_xyz, write_log_header, write_log_line, write_xyz_frame
from minimd.state import SimState


def _get_backend(name: str) -> tuple[NeighborList, ForceEvaluator]:
    """Instantiate neighbor list and force evaluator by backend name."""
    if name == "numpy":
        from minimd.backends.numpy_backend import NumpyLJForces, NumpyNeighborList

        return NumpyNeighborList(), NumpyLJForces()
    raise ValueError(
        f"Unknown backend '{name}'. Available: numpy "
        f"(stubs: torch, fortran, cpp_openmp, cpp_mpi, cuda)"
    )


def run(config: Config) -> None:
    """Run the full MD simulation described by *config*."""
    box = np.array(config.box, dtype=np.float64)

    # --- initialise state ---
    state = read_xyz(config.xyz_file, box, config.temperature)
    nlist, force_eval = _get_backend(config.backend)

    # --- initial force evaluation ---
    nlist.build(state.positions, box, config.r_cut, config.r_skin)
    state.forces, pe = force_eval.compute(
        state.positions, box, nlist.pairs_i, nlist.pairs_j, config.r_cut
    )

    rng = np.random.default_rng()

    # --- output files ---
    stem = Path(config.xyz_file).stem
    traj_path = f"{stem}_traj.xyz"
    log_path = f"{stem}.log"

    with open(traj_path, "w") as f_traj, open(log_path, "w") as f_log:
        write_log_header(f_log)

        # --- log step 0 ---
        _log_and_traj(f_log, f_traj, state, 0, pe, config)

        # --- main loop ---
        for step in range(1, config.n_steps + 1):
            pe = velocity_verlet_step(state, config, nlist, force_eval)

            if config.ensemble == "nvt":
                svr_thermostat(state, config.temperature, config.tau, config.dt, rng)

            if step % config.log_every == 0 or step == config.n_steps:
                _log_and_traj(f_log, f_traj, state, step, pe, config)


def _log_and_traj(
    f_log, f_traj, state: SimState, step: int, pe: float, config: Config
) -> None:
    """Write log line and (optionally) trajectory frame."""
    ke = state.kinetic_energy
    write_log_line(f_log, step, pe, ke, state.temperature)
    f_log.flush()
    if step % config.traj_every == 0 or step == 0:
        write_xyz_frame(f_traj, state, step, pe)
        f_traj.flush()
