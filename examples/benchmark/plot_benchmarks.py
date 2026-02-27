#!/usr/bin/env python
"""Plot benchmark results: time vs supercell size for each backend and thread count."""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark_results.csv")

STYLES = {
    "python":     ("Python",            "tab:gray",   "solid",  "o"),
    "numpy":      ("NumPy",             "tab:blue",   "solid",  "o"),
    "fortran":    ("Fortran",           "tab:orange", "solid",  "o"),
    "torch_cpu":  ("PyTorch (CPU)",     "tab:purple", "solid",  "o"),
    "torch":      ("PyTorch (3090)",    "tab:purple", "dashed", "s"),
    "cpp_mpi":    ("MPI (ghost atoms)", "tab:green",  None,     None),
    "cpp_openmp": ("OpenMP",            "tab:red",    None,     None),
}
THREAD_STYLE = {1: ("solid", "o"), 2: ("dashed", "o"), 8: ("dashed", "s"), 14: ("dotted", "s")}
THREAD_VARYING = {"cpp_openmp", "cpp_mpi"}

present = set(df["backend"].unique())
fig, ax = plt.subplots(figsize=(8, 6))

for backend, (label, color, ls, mk) in STYLES.items():
    if backend not in present:
        continue
    if backend in THREAD_VARYING:
        for t in sorted(df[df["backend"] == backend]["threads"].unique()):
            if t in [2, 14]: continue
            
            data = df[(df["backend"] == backend) & (df["threads"] == t)]
            tls, tmk = THREAD_STYLE.get(t, ("solid", "o"))
            ax.plot(data["supercell"]**3, data["time"], color=color,
                    linestyle=tls, marker=tmk, label=f"{label} ({t}T)", linewidth=2)
    else:
        data = df[df["backend"] == backend]
        ax.plot(data["supercell"]**3, data["time"], color=color,
                linestyle=ls, marker=mk, label=label, linewidth=2)

ax.set_xlabel("# atoms", fontsize=12)
ax.set_ylabel("Time (s)", fontsize=12)
ax.set_title("Simulation Time vs Supercell Size (n_steps=1000)", fontsize=14)
ax.legend(title="Implementation")
ax.set_yscale("log")
ax.set_xscale("log")

plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=600)
plt.show()
