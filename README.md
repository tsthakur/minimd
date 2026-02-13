A minimalist molecular dynamics (MD) code in Python. It simulates Lennard-Jones particles under periodic boundary conditions using velocity Verlet integration, with optional stochastic velocity rescaling (SVR) thermostat for NVT ensemble. All quantities are in LJ reduced units (mass = kB = 1).

```bash
# Install in editable mode
pip install -e .

# Compile fortran shared library manually by:
cd minimd/backends/fortran && python -m numpy.f2py -c -m _lj_fortran lj_fortran.f90

# Run a simulation (takes a YAML config file)
python -m minimd examples/nve.yaml
# or equivalently:
minimd examples/nve.yaml
```

The main draw of this code is that there are backends to test out various implementations for building neighbour list and force evaluation as these are the most computationally intensive part of any MD code. Currently supported:
* Python with base Python functions
* Python with NumPy
* Fortran 
* C++ with openMP

The dynamics is always done on python.