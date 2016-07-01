REM getMaskParams HDF5videoViewer MWTrackerViewer SWTrackerViewer 
REM --noconfirm --clean
for %%F in (MWTrackerViewer getMaskParams HDF5videoViewer MWTrackerViewer SWTrackerViewer) ^
do ^
pyinstaller ..\MWTracker_GUI\%%F.py --exclude-module PyQt4 ^
--exclude-module PyQt4.QtCore --exclude-module PyQt4.QtGui ^
--hidden-import=h5py.defs --hidden-import=h5py.utils ^
--hidden-import=h5py.h5ac --hidden-import=h5py._proxy ^
--console & ^
mkdir .\dist\%%F\MWTracker\featuresAnalysis & ^
copy ..\MWTracker\featuresAnalysis\features_names.csv .\dist\%%F\MWTracker\featuresAnalysis & ^
mkdir .\dist\%%F\open_worm_analysis_toolbox\features\feature_metadata & ^
copy ..\..\open-worm-analysis-toolbox\open_worm_analysis_toolbox\features\feature_metadata\features_list.csv ^
.\dist\%%F\open_worm_analysis_toolbox\features\feature_metadata & ^
copy ..\..\open-worm-analysis-toolbox\open_worm_analysis_toolbox\features\master_eigen_worms_n2.mat ^
.\dist\%%F\open_worm_analysis_toolbox\features\