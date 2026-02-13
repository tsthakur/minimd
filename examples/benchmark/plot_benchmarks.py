#!/usr/bin/env python
"""Plot benchmark results: time vs supercell size for each backend."""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark_results.csv")

fig, ax = plt.subplots(figsize=(8, 6))

for backend in df["backend"].unique():
    data = df[df["backend"] == backend]
    ax.plot(data["supercell"]**3, data["time"], marker="o", label=backend, linewidth=2)

ax.set_xlabel("# atoms", fontsize=12)
ax.set_ylabel("Time (s)", fontsize=12)
ax.set_title("Simulation Time vs Supercell Size (n_steps=1000)", fontsize=14)
ax.legend(title="Backend")

ax.set_yscale("log")
ax.set_xscale("log")

plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=600)
plt.show()
