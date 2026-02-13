"""Main simulation driver."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from minimd.backends.cpp_openmp.backend import CppOpenMPLJForces, CppOpenMPNeighborList
from minimd.config import Config
from minimd.integrator import svr_thermostat, velocity_verlet_step
from minimd.interfaces import ForceEvaluator, NeighborList
from minimd.io import make_supercell, read_xyz, write_log_header, write_log_line, write_log_timing, write_xyz_frame
from minimd.state import SimState


def _get_backend(name: str) -> tuple[NeighborList, ForceEvaluator]:
    """Instantiate neighbour list and force evaluator by backend name."""
    if name == "numpy":
        from minimd.backends.numpy_backend import NumpyLJForces, NumpyNeighborList

        return NumpyNeighborList(), NumpyLJForces()
    if name == "python":
        from minimd.backends.python_backend import PythonLJForces, PythonNeighborList

        return PythonNeighborList(), PythonLJForces()
    if name == "fortran":
        from minimd.backends.fortran import FortranLJForces, FortranNeighborList

        return FortranNeighborList(), FortranLJForces()
    
    if name == "cpp_openmp":
        from minimd.backends.fortran import FortranLJForces, FortranNeighborList

        return CppOpenMPNeighborList(), CppOpenMPLJForces()
    raise ValueError(
        f"Unknown backend '{name}'. Available: numpy, python, fortran, cpp_openmp"
        f"(stubs: torch, cpp_mpi, cuda)"
    )


def run(config: Config) -> None:
    """Run the full MD simulation described by *config*."""
    box = np.array(config.box, dtype=np.float64)

    # --- initialise state ---
    state = read_xyz(config.input_file, box, config.temperature)

    # --- apply supercell replication ---
    nx, ny, nz = config.supercell
    if nx > 1 or ny > 1 or nz > 1:
        state = make_supercell(state, nx, ny, nz)
        box = state.box  # update box to match supercell

    nlist, force_eval = _get_backend(config.backend)

    # --- initial force evaluation ---
    nlist.build(state.positions, box, config.r_cut, config.r_skin)
    state.forces, pe = force_eval.compute(
        state.positions, box, nlist.pairs_i, nlist.pairs_j, config.r_cut,
        config.sigma, config.epsilon,
    )

    rng = np.random.default_rng()

    # --- output files ---
    if config.output_file:
        stem = config.output_file
    else:
        stem = Path(config.input_file).stem

    traj_path = f"{stem}_traj.xyz"
    log_path = f"{stem}.log"

    with open(traj_path, "w") as f_traj, open(log_path, "w") as f_log:
        write_log_header(f_log)

        # --- log step 0 ---
        write_log_line(f_log, 0, pe, state.kinetic_energy, state.temperature)

        # --- main loop ---
        t_start = time.perf_counter()
        for step in range(1, config.n_steps + 1):
            pe = velocity_verlet_step(state, config, nlist, force_eval)

            if config.ensemble == "nvt":
                svr_thermostat(state, config.temperature, config.tau, config.dt, rng)

            if step % config.log_every == 0 or step == config.n_steps:
                write_log_line(f_log, step, pe, state.kinetic_energy, state.temperature)
                f_log.flush()
            if step % config.traj_every == 0 or step == config.n_steps:
                write_xyz_frame(f_traj, state, step, pe)
                f_traj.flush()
        t_end = time.perf_counter()

        total_time = t_end - t_start
        avg_step_time = total_time / config.n_steps
        write_log_timing(f_log, total_time, avg_step_time)

