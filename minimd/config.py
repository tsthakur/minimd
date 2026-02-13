"""Simulation configuration loaded from YAML."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class Config:
    """All user-facing simulation parameters.

    Units are LJ reduced units (mass = kB = 1).
    """

    # --- files ---
    input_file: str
    output_file: str 

    # --- ensemble ---
    ensemble: str = "nvt"       # "nve" or "nvt"
    temperature: float = 1.0    # target T (used for init and thermostat)
    tau: float = 100.0          # SVR coupling timescale (dt units)

    # --- simulation ---
    dt: float = 0.005
    n_steps: int = 1000
    log_every: int = 100
    traj_every: int = 100

    # --- potential ---
    r_cut: float = 2.5          # LJ cutoff radius
    r_skin: float = 0.5         # Verlet list skin
    epsilon: float = 1.0        # LJ well depth
    sigma: float = 1.0          # LJ size parameter

    # --- random seed ---
    seed: float = 1

    # --- box ---
    box: tuple[float, float, float] = (10.0, 10.0, 10.0)

    # --- supercell ---
    supercell: tuple[int, int, int] = (1, 1, 1)

    # --- backend ---
    backend: str = "numpy"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        # convert box list -> tuple
        if "box" in raw and isinstance(raw["box"], list):
            raw["box"] = tuple(raw["box"])
        if "supercell" in raw and isinstance(raw["supercell"], list):
            raw["supercell"] = tuple(raw["supercell"])
        return cls(**raw)
