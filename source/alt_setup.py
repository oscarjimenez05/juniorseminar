from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import os

# Note: If you are on Windows (MSVC), use '/O2' instead of '-O3' and '-march=native'.
c_args = [
    "-O3",
    "-ffast-math",
    "-funroll-loops",
]

extensions = [
    Extension(
        "lcg_fenwick",
        ["alternatives/lcg_fenwick.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "xor_fenwick",
        ["alternatives/xor_fenwick.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "logistic_lh",
        ["alternatives/logistic_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "slope_lh",
        ["alternatives/slope_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
    Extension(
        "decay_lh",
        ["alternatives/decay_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
]

setup(
    name="Lehmerized Generator Alternatives Cython Module",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
            'initializedcheck': False,
            'language_level': "3",
        }
    ),
)