import pybind11
from setuptools import setup, Extension

cpp_ext = Extension(
    name="minimd.backends.cpp_openmp._lj_cpp_openmp",
    sources=["minimd/backends/cpp_openmp/lj_cpp_openmp.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-std=c++17", "-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(ext_modules=[cpp_ext])
