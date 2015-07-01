# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:16:50 2015

@author: ajaver
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(ext_modules = cythonize("imageDifferenceMask.pyx"), include_dirs=[numpy.get_include()])

#python3 setup.py build_ext --inplace
