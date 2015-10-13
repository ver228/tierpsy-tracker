sudo echo "Thanks."

MW_MAIN_DIR=`pwd`

sudo chown -R `whoami` /usr/local/


ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

#python dependencies
brew install python3
brew install homebrew/science/hdf5

pip3 install numpy
pip3 install spyder
pip3 install tables
pip3 install pandas
pip3 install h5py
pip3 install matplotlib
pip3 install scipy
pip3 install scikit-learn
pip3 install scikit-image
pip3 install tifffile
pip3 install seaborn

#install pyqt5 for the GUI
brew uninstall --force sip
brew install sip --with-python3
brew uninstall --force pyqt5
brew install pyqt5 --with-python3

#install opencv
brew install cmake
brew install ffmpeg

curl https://github.com/Itseez/opencv/archive/3.0.0.zip > $MW_MAIN_DIR/opencv-3.0.0.zip
unzip $MW_MAIN_DIR/opencv-3.0.0.zip
cd $MW_MAIN_DIR/opencv-3.0.0

PY_VER=`python3 -c "import sys; print(sys.version.partition(' ')[0])"`
PY_VER_SHORT=`python3 -c "import sys; print('.'.join(sys.version.partition(' ')[0].split('.')[0:2]))"`
cmake '"Unix Makefile"' -DBUILD_opencv_python3=ON \
-DBUILD_opencv_python2=OFF \
-DPYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/$PY_VER/Frameworks/Python.framework/Versions/$PY_VER_SHORT/include/python$PY_VER_SHORTm/ \
-DPYTHON3_LIBRARY=/usr/local/Cellar/python3/$PY_VER/Frameworks/Python.framework/Versions/$PY_VER_SHORT/lib/libpython$PY_VER_SHORTm.dylib \
-DPYTHON3_PACKAGES_PATH=/usr/local/lib/python$PY_VER_SHORT/site-packages \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python$PY_VER_SHORT/site-packages/numpy/core/include \
-DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DWITH_CUDA=OFF -DENABLE_AVX=ON -DWITH_OPENGL=ON -DWITH_OPENCL=ON \
-DWITH_IPP=ON -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
-DWITH_QT=ON -DINSTALL_PYTHON_EXAMPLES=ON \
-DCMAKE_BUILD_TYPE=RELEASE \
.
make -j24
make install

#old ffmpeg to read mjpg
cd $MW_MAIN_DIR
rm -rf $MW_MAIN_DIR/opencv-3.0.0
rm $MW_MAIN_DIR/opencv-3.0.0.zip

curl http://ffmpegmac.net/resources/SnowLeopard_Lion_Mountain_Lion_Mavericks_27.03.2014.zip > $MW_MAIN_DIR/ffmpeg_old.zip
unzip $MW_MAIN_DIR/ffmpeg_old.zip ffmpeg
sudo mv ffmpeg /usr/local/bin/ffmpeg22
rm $MW_MAIN_DIR/ffmpeg_old.zip

#compile cython files
cd $MW_MAIN_DIR/MWTracker/trackWorms/segWormPython/cythonFiles/
make
make clean
cd $MW_MAIN_DIR