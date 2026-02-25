""" To estimate the cost of C++ functions

For debugging run:
perf stat -e cycles,instructions,cache-misses,cache-references python -m minimd nvt.yaml
perf record -g python -m minimd nvt.yaml
perf report --stdio --no-children | head -40

"""
import time
import numpy as np
from minimd.backends.cpp_mpi import _lj_cpp_mpi
from minimd.backends.cpp_openmp import _lj_cpp_openmp
from minimd.config import Config
from minimd import io

N = 25
S = 4
THREADS = 8


def main():
    cfg = Config.from_yaml("nvt.yaml")
    box = np.array(cfg.box, dtype=np.float64)
    state = io.read_xyz(cfg.input_file, box, cfg.temperature, cfg.seed)
    state_s = io.make_supercell(state, S, S, S, cfg.seed)
    box = state_s.box
    positions = state_s.positions
    r_list = cfg.r_cut + cfg.r_skin


    print("MPI version:")
    t0 = time.perf_counter()
    for _ in range(N):
        ghost_pos, ghost_map, ghost_shifts = _lj_cpp_mpi.build_ghost_atoms(positions, box, r_list)
    print(f"build_ghost_atoms:      {(time.perf_counter()-t0)/N*1e3:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(N):
        pi, pj = _lj_cpp_mpi.build_neighbour_list(positions, ghost_pos, ghost_map, r_list)
    print(f"build_neighbour_list:   {(time.perf_counter()-t0)/N*1e3:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(N):
        gp = _lj_cpp_mpi.update_ghost_positions(positions, ghost_map, ghost_shifts)
    print(f"update_ghost_positions: {(time.perf_counter()-t0)/N*1e6:.2f} μs")

    t0 = time.perf_counter()
    for _ in range(N):
        f, e = _lj_cpp_mpi.compute_forces(positions, gp, ghost_map, pi, pj, cfg.r_cut, cfg.sigma, cfg.epsilon)
    print(f"compute_forces:         {(time.perf_counter()-t0)/N*1e3:.2f} ms")

    print(f"OpenMP version ({THREADS} threads):")
    _lj_cpp_openmp.set_num_threads(THREADS)

    t0 = time.perf_counter()
    for _ in range(N):
        pi, pj = _lj_cpp_openmp.build_neighbour_list(positions, box, cfg.r_cut, cfg.r_skin)
    print(f"build_neighbour_list:   {(time.perf_counter()-t0)/N*1e3:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(N):
        f, e = _lj_cpp_openmp.compute_forces(positions, box, pi, pj, cfg.r_cut, cfg.sigma, cfg.epsilon)
    print(f"compute_forces:         {(time.perf_counter()-t0)/N*1e3:.2f} ms")


if __name__ == "__main__":
    main()