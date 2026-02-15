from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# Note: If you are on Windows (MSVC), use '/O2' instead of '-O3' and '-march=native'.
c_args = [
    "-O3",
    "-ffast-math",
    "-funroll-loops",
]

extensions = [
    Extension(
        "crypto.crypto_lh",
        ["crypto/crypto_lh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=c_args,
    ),
]

setup(
    name="Lehmerized CSPRNG Cython Module",
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