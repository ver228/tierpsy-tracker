
PYVER=`python3 -c "import MWTracker; print(MWTracker.__version__)"`
for FNAME in MWTrackerViewer HDF5videoViewer SWTrackerViewer getMaskParams
do 
pyinstaller ../MWTracker_GUI/$FNAME.py --exclude-module PyQt4 \
--exclude-module PyQt4.QtCore --exclude-module PyQt4.QtGui \
--hidden-import=h5py.defs --hidden-import=h5py.utils \
--hidden-import=h5py.h5ac --hidden-import='h5py._proxy' \
--noconfirm --console

mkdir -p ./dist/$FNAME/MWTracker/featuresAnalysis
cp ../MWTracker/featuresAnalysis/features_names.csv ./dist/$FNAME/MWTracker/featuresAnalysis

mkdir -p ./dist/$FNAME/open_worm_analysis_toolbox/features/feature_metadata/
cp ../../open-worm-analysis-toolbox/open_worm_analysis_toolbox/features/feature_metadata/features_list.csv \
./dist/$FNAME/open_worm_analysis_toolbox/features/feature_metadata/
cp ../../open-worm-analysis-toolbox/open_worm_analysis_toolbox/features/master_eigen_worms_n2.mat \
./dist/$FNAME/open_worm_analysis_toolbox/features/

done

#-y --clean --windowed