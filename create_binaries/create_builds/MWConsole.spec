# -*- mode: python -*-


import os
import sys
import MWTracker
import open_worm_analysis_toolbox


IS_WIN = sys.platform == 'win32'
SRC_SCRIPT_PATH = os.path.join('..', 'scripts', 'MWConsole.py')
DST_BUILD=os.path.abspath('.')
CREATE_CONSOLE= IS_WIN #make a separated console only in windows. I have to do this due to a problem with pyinstaller


def get_extra_files():
  #openworm additional files
  open_worm_path = os.path.dirname(open_worm_analysis_toolbox.__file__)

  ow_feat_ori = os.path.join(open_worm_path, 'features', 'feature_metadata', 'features_list.csv')
  ow_feat_dst = os.path.join('open_worm_analysis_toolbox', 'features','feature_metadata')

  ow_eigen_ori = os.path.join(open_worm_path, 'features', 'master_eigen_worms_n2.mat')
  ow_eigen_dst = os.path.join('open_worm_analysis_toolbox', 'features')

  #add prev compiled binary (they should have been runned before)
  cSW_f = os.path.join(DST_BUILD, 'dist', 'compressSingleWorker', 'compressSingleWorker')
  tSW_f = os.path.join(DST_BUILD, 'dist', 'trackSingleWorker', 'trackSingleWorker')
  if IS_WIN:
    cSW_f += '.exe'
    tSW_f += '.exe'


  #create added files
  added_files = [(ow_feat_ori, ow_feat_dst),
          (ow_eigen_ori, ow_eigen_dst),
          (cSW_f, '.'),
          (tSW_f, '.')]

  MWTracker_path = os.path.dirname(MWTracker.__file__)
  #I add the file separator at the end, it makes my life easier later on
  MWTracker_path += os.sep

  #add all the files in misc
  for (dirpath, dirnames, filenames) in os.walk(os.path.join(MWTracker_path, 'misc')):
    for fname in filenames:
      if not fname.startswith('.'):
        ori_fname = os.path.join(dirpath, fname)
        dst_path = dirpath.replace(MWTracker_path, '')
        added_files.append((ori_fname, dst_path))




block_cipher = None
a = Analysis([SRC_SCRIPT_PATH],
             pathex=[DST_BUILD],
             binaries=None,
             datas=get_extra_files(),
             hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy',
             'cython', 'sklearn', 'sklearn.neighbors.typedefs'],
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
          console=CREATE_CONSOLE )

if not IS_WIN:
  app = BUNDLE(exe,
               name='MWConsole.app',
               icon=None,
               bundle_identifier=None,
               info_plist={
                'CFBundleShortVersionString' : MWTracker.__version__,
                'NSHighResolutionCapable': 'True'
                }
              )
