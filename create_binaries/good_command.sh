
PYVER=`python3 -c "import MWTracker; print(MWTracker.__version__)"`
for FNAME in HDF5videoViewer MWTrackerViewer SWTrackerViewer getMaskParams
do 
pyinstaller ../MWTracker_GUI/$FNAME.py --exclude-module PyQt4 \
--exclude-module PyQt4.QtCore --exclude-module PyQt4.QtGui \
--hidden-import=h5py.defs --hidden-import=h5py.utils \
--hidden-import=h5py.h5ac --hidden-import='h5py._proxy' \
-y --clean --windowed
done

