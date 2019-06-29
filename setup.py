from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(name='H&E histopathological staining normalization',
      ext_modules=cythonize("normalizeStaining_cython.pyx"),
      include_dirs=[numpy.get_include()])