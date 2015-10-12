MW_MAIN_DIR=`pwd`

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



