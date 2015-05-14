# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:16:50 2015

@author: ajaver
"""
from distutils.core import setup#, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize("min_avg_difference.pyx"), include_dirs=[numpy.get_include()]#"calculate_ratio.pyx" "image_difference.pyx" calContrastMaps.pyx
      #ext_modules=cythonize("min_avg_difference.pyx"), include_dirs=[numpy.get_include()]
)    