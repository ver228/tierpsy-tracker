# -*- mode: python -*-

block_cipher = None


a = Analysis(['HDF5videoViewer.py'],
             pathex=['/Users/ajaver/Documents/GitHub/Multiworm_Tracking/', '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI'],
             binaries=None,
             datas=None,
             hiddenimports=['tables', 'PyQt5'],
             hookspath=None,
             runtime_hooks=None,
             excludes=None,
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='HDF5videoViewer',
          debug=False,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='HDF5videoViewer')
