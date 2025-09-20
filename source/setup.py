from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# Define the extension module. This tells setuptools what to compile.
extensions = [
    Extension(
        # The name of the resulting .so or .pyd file
        "c_lcg_lh",
        # The list of source Cython files
        ["c_lcg_lh.pyx"],
        # We need to include the NumPy C header files for compilation
        include_dirs=[numpy.get_include()],
    ),
]

# The main setup function
setup(
    name="LCG_LH Cython Module",
    # cythonize() converts the .pyx file to a .c file and compiles it
    ext_modules=cythonize(extensions),
)
