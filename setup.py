#!/usr/bin/env python

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os

MODULE_NAME = "tierpsy"
AUTHOR = 'Avelino Javer'
AUTHOR_EMAIL = 'avelino.javer@imperial.ac.uk'
URL = 'https://github.com/ver228/tierpsy-tracker'
DOWNLOAD_URL = 'https://github.com/ver228/tierpsy-tracker'
DESCRIPTION = "tierpsy: Tierpsy Tracker Multi-Worm Tracker."
exec(open(MODULE_NAME + '/version.py').read())
VERSION = __version__

def _get_ext_modules():
  #build cython files
  # python3 setup.py build_ext --inplace
  path_parts = [MODULE_NAME, 'analysis', 'ske_create', 'segWormPython', 'cython_files']
  cython_path = os.path.join(*path_parts)
  cython_path_e = os.path.join(MODULE_NAME, 'analysis', 'stage_aligment')

  def _add_path(f_list):
  	return [os.path.join(cython_path, x) for x in f_list]

  def _get_mod_path(name):
      return '.'.join(path_parts + [name])

  ext_files = {
  	"circCurvature" : ["circCurvature.pyx", "c_circCurvature.c"],
  	"curvspace" : ["curvspace.pyx", "c_curvspace.c"]
  }

  include_dirs = [numpy.get_include()]
  ext_modules = cythonize(os.path.join(cython_path, "*_cython.pyx"))
  ext_modules += cythonize(os.path.join(cython_path_e, "*.pyx"))
  ext_modules += [Extension(_get_mod_path(name), 
                            sources=_add_path(files), 
                            include_dirs=include_dirs) 
  			   for name, files in ext_files.items()]
  return ext_modules



#install setup
setup(name=MODULE_NAME,
   version=VERSION,
   description=DESCRIPTION,
   author=AUTHOR,
   author_email=AUTHOR_EMAIL,
   url=URL,
   packages=['tierpsy'],
   cmdclass={'build_ext': build_ext},
   ext_modules=_get_ext_modules(),
   include_dirs=[numpy.get_include()],
   include_package_data=True
   )
