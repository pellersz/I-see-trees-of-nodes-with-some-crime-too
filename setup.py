from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Random forest implementation',
    ext_modules=cythonize("rrr.pyx"),
    include_dirs=[numpy.get_include()]
)
