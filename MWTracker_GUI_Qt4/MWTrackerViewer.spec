# -*- mode: python -*-

block_cipher = None


a = Analysis(['MWTrackerViewer.py'],
             pathex=['/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI'],
             binaries=None,
             datas=None,
             hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy'],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          name='MWTrackerViewer',
          debug=False,
          strip=False,
          upx=True,
          console=False )
app = BUNDLE(exe,
             name='MWTrackerViewer.app',
             icon=None,
             bundle_identifier=None)
