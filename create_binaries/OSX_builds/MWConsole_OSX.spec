# -*- mode: python -*-
import os
import MWTracker
import open_worm_analysis_toolbox

MWTracker_path = os.path.dirname(MWTracker.__file__)
open_worm_path = os.path.dirname(open_worm_analysis_toolbox.__file__)

added_files = [
(os.path.join(open_worm_path, 'features/feature_metadata/features_list.csv'), 
'open_worm_analysis_toolbox/features/feature_metadata'),
(os.path.join(open_worm_path, 'features/master_eigen_worms_n2.mat'), 
'open_worm_analysis_toolbox/features'),
(os.path.join(MWTracker_path, 'auxFiles/features_names.csv'), 
'auxFiles'),
(os.path.join(MWTracker_path, 'auxFiles/ffmpeg22'), 
'auxFiles'),
('/usr/local/bin/ffprobe', 'auxFiles'),
('./dist/compressSingleWorker/compressSingleWorker', '.'),
('./dist/trackSingleWorker/trackSingleWorker', '.')
]

block_cipher = None

a = Analysis(['../scripts/MWConsole.py'],
             pathex=['/Users/ajaver/Documents/GitHub/Multiworm_Tracking/create_binaries'],
             binaries=None,
             datas=added_files,
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
          name='MWConsole',
          debug=False,
          strip=False,
          upx=True,
          console=False )
app = BUNDLE(exe,
             name='MWConsole.app',
             icon=None,
             bundle_identifier=None)
