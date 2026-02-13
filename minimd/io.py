"""I/O helpers: XYZ reader/writer and log file writer.
In princinple, ase can be used, but I want to keep dependencies minimal for now.
XYZ format (extended):
    Line 1:  N (number of atoms)
    Line 2:  comment — we store step, PE, KE, T here
    Lines 3…: symbol x y z [vx vy vz]

Velocities are optional on read (columns 5-7). If absent they are
initialised from the Maxwell-Boltzmann distribution at the target T.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import numpy as np

from minimd.state import SimState


# ---------- XYZ reading ----------

def read_xyz(path: str | Path, box: np.ndarray, temperature: float, seed: int | None = None) -> SimState:
    """Read an XYZ file and return a SimState.

    If velocities are not present in the file, they are drawn from a
    Maxwell-Boltzmann distribution at *temperature* (LJ units, kB=1).
    If *seed* is provided, velocities will be deterministic for that seed.
    """
    lines = Path(path).read_text().strip().splitlines()
    n_atoms = int(lines[0])
    # skip comment line
    symbols: list[str] = []
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    velocities: np.ndarray | None = None
    has_velocities = False

    for i, line in enumerate(lines[2 : 2 + n_atoms]):
        parts = line.split()
        symbols.append(parts[0])
        positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        if len(parts) > 4 and i == 0:
            has_velocities = True
            velocities = np.zeros((n_atoms, 3), dtype=np.float64)
        if has_velocities and velocities is not None:
            velocities[i] = [float(parts[4]), float(parts[5]), float(parts[6])]

    masses = np.ones(n_atoms, dtype=np.float64)  # all mass = 1

    if velocities is None:
        velocities = _maxwell_boltzmann(n_atoms, masses, temperature, seed)

    state = SimState(
        symbols=symbols,
        positions=positions,
        velocities=velocities,
        forces=np.zeros_like(positions),
        box=box,
        masses=masses,
    )
    state.remove_com_velocity()
    return state


def _maxwell_boltzmann(
    n: int,
    masses: np.ndarray,
    temperature: float,
    seed: int | None = None,
) -> np.ndarray:
    """Draw velocities from Maxwell-Boltzmann at given temperature (kB=1).
    
    If *seed* is provided, velocities will be deterministic.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(temperature / masses)  # (N,)
    return rng.standard_normal((n, 3)) * sigma[:, None]


# ---------- supercell generation ----------

def make_supercell(
    state: SimState,
    nx: int,
    ny: int,
    nz: int,
    seed: int | None = None,
) -> SimState:
    """Replicate *state* by (nx, ny, nz) to build a supercell.

    Positions are tiled along each axis and the box is scaled accordingly.
    Velocities are re-initialised from Maxwell-Boltzmann distribution.
    Returns a new SimState; the original is not modified.
    If `seed` is provided, velocities will be deterministic.
    """
    if nx == 1 and ny == 1 and nz == 1:
        return state  # nothing to do

    n_atoms = state.n_atoms
    n_total = n_atoms * nx * ny * nz
    box = state.box.copy()

    new_positions = np.zeros((n_total, 3), dtype=np.float64)
    new_velocities = np.zeros((n_total, 3), dtype=np.float64)
    new_symbols: list[str] = []

    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = np.array([ix * box[0], iy * box[1], iz * box[2]])
                new_positions[idx : idx + n_atoms] = state.positions + shift
                new_symbols.extend(state.symbols)
                idx += n_atoms

    new_box = box * np.array([nx, ny, nz], dtype=np.float64)
    new_masses = np.tile(state.masses, nx * ny * nz)
    # When creating a supercell, original velocities are not meaningful
    new_velocities = _maxwell_boltzmann(n_total, new_masses, state.temperature, seed)

    return SimState(
        symbols=new_symbols,
        positions=new_positions,
        velocities=new_velocities,
        forces=np.zeros((n_total, 3), dtype=np.float64),
        box=new_box,
        masses=new_masses,
    )


# ---------- XYZ writing ----------

def write_xyz_frame(
    f: TextIO,
    state: SimState,
    step: int,
    pe: float,
) -> None:
    """Append one XYZ frame (positions + velocities) to an open file."""
    n = state.n_atoms
    ke = state.kinetic_energy
    temp = state.temperature
    f.write(f"{n}\n")
    f.write(f"step={step} PE={pe:.6f} KE={ke:.6f} T={temp:.6f}\n")
    for i in range(n):
        s = state.symbols[i]
        x, y, z = state.positions[i]
        vx, vy, vz = state.velocities[i]
        f.write(f"{s} {x:.8f} {y:.8f} {z:.8f} {vx:.8f} {vy:.8f} {vz:.8f}\n")


# ---------- log writing ----------

def write_log_header(f: TextIO) -> None:
    """Write the log file column header."""
    f.write(f"{'step':>10s} {'PE':>14s} {'KE':>14s} {'E_tot':>14s} {'T':>14s}\n")


def write_log_line(f: TextIO, step: int, pe: float, ke: float, temp: float) -> None:
    """Append one line to the log file."""
    e_tot = pe + ke
    f.write(f"{step:10d} {pe:14.6f} {ke:14.6f} {e_tot:14.6f} {temp:14.6f}\n")


def write_log_timing(f: TextIO, total_time: float, avg_step_time: float) -> None:
    """Write timing summary at the end of the log file."""
    f.write(f"# Total wall time:     {total_time:.3f} s\n")
    f.write(f"# Avg time per step:   {avg_step_time:.6f} s\n")
