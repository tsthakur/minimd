"""C++ backend with ghost atom approach (MPI-ready architecture)."""

from minimd.backends.cpp_mpi.backend import CppMPILJForces, CppMPINeighborList

__all__ = ["CppMPINeighborList", "CppMPILJForces"]
