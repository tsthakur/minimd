import subprocess
import pybind11
from setuptools import setup, Extension

def mpi_flags(flag):
    out = subprocess.check_output(["mpicxx", flag]).decode().split()
    return out

mpi_compile = mpi_flags("--showme:compile")
mpi_link    = mpi_flags("--showme:link")
mpi_inc     = [f[2:] for f in mpi_compile if f.startswith("-I")]
mpi_lib_dirs= [f[2:] for f in mpi_link    if f.startswith("-L")]
mpi_libs    = [f[2:] for f in mpi_link    if f.startswith("-l")]
mpi_extra   = [f     for f in mpi_compile if not f.startswith("-I")]

cpp_ext = Extension(
    name="minimd.backends.cpp_openmp._lj_cpp_openmp",
    sources=["minimd/backends/cpp_openmp/lj_cpp_openmp.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-std=c++17", "-O3", "-fopenmp", "-g"],
    extra_link_args=["-fopenmp"],
)

cpp_mpi_ext = Extension(
    name="minimd.backends.cpp_mpi._lj_cpp_mpi",
    sources=["minimd/backends/cpp_mpi/lj_cpp_mpi.cpp"],
    include_dirs=[pybind11.get_include()] + mpi_inc,
    library_dirs=mpi_lib_dirs,
    libraries=mpi_libs,
    extra_compile_args=["-std=c++17", "-O3", "-g"] + mpi_extra,
)

setup(ext_modules=[cpp_ext, cpp_mpi_ext])
