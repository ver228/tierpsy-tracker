MW_MAIN_DIR=`pwd`

#install opencv
#wget https://github.com/Itseez/opencv/archive/3.0.0.zip -O $MW_MAIN_DIR/opencv-3.0.0.zip
#unzip $MW_MAIN_DIR/opencv-3.0.0.zip
#cd $MW_MAIN_DIR/opencv-3.0.0

cd $MW_MAIN_DIR/..

git clone https://github.com/Itseez/opencv
git clone https://github.com/Itseez/opencv_contrib

cd $MW_MAIN_DIR/../opencv
git checkout 3.1.0

PY_VER=`python3 -c "import sys; print(sys.version.partition(' ')[0])"`
PY_VER_SHORT=`python3 -c "import sys; print('.'.join(sys.version.partition(' ')[0].split('.')[0:2]))"`

#for some weird reason i have to execute make twice or it does not find the python libraries
for i in 1 2
do
cmake '"Unix Makefile"' -DBUILD_opencv_python3=ON \
-DBUILD_opencv_python2=OFF \
-DPYTHON_EXECUTABLE=`which python3` \
-DPYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/${PY_VER}/Frameworks/Python.framework/Versions/${PY_VER_SHORT}/include/python${PY_VER_SHORT}m/ \
-DPYTHON3_LIBRARY=/usr/local/Cellar/python3/${PY_VER}/Frameworks/Python.framework/Versions/${PY_VER_SHORT}/lib/libpython${PY_VER_SHORT}m.dylib \
-DPYTHON3_PACKAGES_PATH=/usr/local/lib/python${PY_VER_SHORT}/site-packages \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python${PY_VER_SHORT}/site-packages/numpy/core/include \
-DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DWITH_CUDA=OFF -DENABLE_AVX=ON -DWITH_OPENGL=ON -DWITH_OPENCL=ON \
-DWITH_IPP=ON -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
-DWITH_QT=OFF -DINSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DOPENCV_EXTRA_MODULES_PATH=$MW_MAIN_DIR/../opencv_contrib/modules \
-DBUILD_opencv_surface_matching=OFF \
-DBUILD_opencv_hdf=OFF \
-DBUILD_opencv_xphoto=OFF \
.
done
#make -j24
#make install

#cd $MW_MAIN_DIR
#rm $MW_MAIN_DIR/opencv-3.0.0.zip
#rm -Rf $MW_MAIN_DIR/opencv-3.0.0
