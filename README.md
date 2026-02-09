A minimalist molecular dynamics (MD) code in Python. It simulates Lennard-Jones particles under periodic boundary conditions using velocity Verlet integration, with optional stochastic velocity rescaling (SVR) thermostat for NVT ensemble. All quantities are in LJ reduced units (sigma = epsilon = mass = kB = 1).

```bash
# Install (editable)
pip install -e .

# Run a simulation (takes a YAML config file)
python -m minimd examples/nve.yaml
# or equivalently:
minimd examples/nve.yaml
```

The main draw of this code is that there are backends to test out various implementations (currently only supports python with numpy) for building neighbour list and force evaluation as these are the most computationally intensive part of any MD code. The dynamics is always done on python.