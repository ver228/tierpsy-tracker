# # -*- coding: utf-8 -*-
# """
# Created on Tue Jul  7 11:29:01 2015

# @author: ajaver
# """
import os
import sys
import warnings
from .version import __version__


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#I want to be sure tierpsy loads tensorflow flow backend
os.environ['KERAS_BACKEND']='tensorflow' 

with warnings.catch_warnings():
    #to remove annoying warnings in case matplotlib was imported before
    warnings.simplefilter("ignore")
    # force qt5 to be the backend of matplotlib.
    import matplotlib
    matplotlib.use('Qt5Agg')

try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS
except Exception:
    base_path = os.path.dirname(__file__)

AUX_FILES_DIR = os.path.abspath(os.path.join(base_path, 'extras'))
DFLT_PARAMS_PATH = os.path.join(AUX_FILES_DIR, 'param_files')

DFLT_PARAMS_FILES = sorted([x for x in os.listdir(DFLT_PARAMS_PATH) if x.endswith('.json')])

#this will be true if it is a pyinstaller "frozen" binary
IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    #if IS_FROZEN: 
    if os.name == 'nt':
            # load dll for numpy in windows
            import ctypes

            # Override dll search path.
            python_dir = os.path.split(sys.executable)[0]
            ctypes.windll.kernel32.SetDllDirectoryW(
                os.path.join(python_dir, 'Library', 'bin'))

            # Init code to load external dll
            ctypes.CDLL('mkl_avx2.dll')
            ctypes.CDLL('mkl_def.dll')
            ctypes.CDLL('mkl_vml_avx2.dll')
            ctypes.CDLL('mkl_vml_def.dll')

            # Restore dll search path.
            ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)

