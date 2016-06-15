# # -*- coding: utf-8 -*-
# """
# Created on Tue Jul  7 11:29:01 2015

# @author: ajaver
# """

from .version import __version__
#force to use matplotlib with qt5
import matplotlib
matplotlib.use('Qt5Agg', force=True)

import os
if os.name == 'nt':
	import ctypes
	import sys

	if getattr(sys, 'frozen', False):
	  # Override dll search path.
	  python_dir =  os.path.split(sys.executable)
	  ctypes.windll.kernel32.SetDllDirectoryW(os.path.join(python_dir, 'Library', 'bin'))
	  
	  # Init code to load external dll
	  ctypes.CDLL('mkl_avx2.dll')
	  ctypes.CDLL('mkl_def.dll')
	  ctypes.CDLL('mkl_vml_avx2.dll')
	  ctypes.CDLL('mkl_vml_def.dll')

	  # Restore dll search path.
	  ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)
