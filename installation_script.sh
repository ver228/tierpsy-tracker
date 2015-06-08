ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

brew install wget
brew install cmake
brew install ffmpeg

brew install python3
pip3 install numpy

wget https://github.com/Itseez/opencv/archive/3.0.0.zip
unzip 3.0.0.zip

cd opencv-3.0.0

cmake "Unix Makefile" -D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/3.4.3/Frameworks/Python.framework/Versions/3.4/include/python3.4m/ \
-D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.4.3/Frameworks/Python.framework/Versions/3.4/lib/libpython3.4m.dylib \
-D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.4/site-packages \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.4/site-packages/numpy/core/include .

make -j24
make install

cd ..
rm -p opencv-3.0.0
rm 3.0.0.zip

brew install --upgrade pyqt --with-python3
pip3 install spyder

brew install hdf5
pip3 install pytables
pip3 install pandas

pip3 install matplotlib
pip3 install scipy
pip3 install scikit-learn
pip3 install scikit-image
pip3 install tifffile
