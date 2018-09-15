# -*- mode: python -*-
#hidden imports needed for tierpsy, maybe there is a better way to call this...
import tierpsy.analysis
base_name = os.path.dirname(tierpsy.analysis.__file__)
analysis_steps = [x for x in os.listdir(base_name) if os.path.exists(os.path.join(base_name, x, '__init__.py'))]
hidden_tierspy = ['tierpsy.analysis.' + x for x in analysis_steps]
print(hidden_tierspy)

import os
from tierpsy.processing.ProcessWorker import BATCH_SCRIPT_WORKER


SRC_SCRIPT_PATH = BATCH_SCRIPT_WORKER[1]
DST_BUILD=os.path.abspath('.')

block_cipher = None

a = Analysis([SRC_SCRIPT_PATH],
             pathex=[DST_BUILD],
             binaries=None,
             datas=None,
             hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy'] + hidden_tierspy,
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4', 'PyQt4.QtCore', 'PyQt4.QtGui'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='ProcessWorker',
          debug=False,
          strip=False,
          upx=True,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='ProcessWorker')
