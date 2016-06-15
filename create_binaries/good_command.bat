REM getMaskParams HDF5videoViewer MWTrackerViewer SWTrackerViewer 
REM --noconfirm --clean
for %%F in (MWTrackerViewer) ^
do ^
pyinstaller ..\MWTracker_GUI\%%F.py --exclude-module PyQt4 ^
--exclude-module PyQt4.QtCore --exclude-module PyQt4.QtGui ^
--hidden-import=h5py.defs --hidden-import=h5py.utils ^
--hidden-import=h5py.h5ac --hidden-import=h5py._proxy ^
--console & ^
mkdir .\dist\%%F\MWTracker\featuresAnalysis & ^
copy ..\MWTracker\featuresAnalysis\features_names.csv .\dist\%%F\MWTracker\featuresAnalysis