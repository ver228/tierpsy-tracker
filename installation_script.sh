MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
OPENCV_DIR=$MW_MAIN_DIR/../opencv

#get sudo permissions
sudo echo "Thanks."

#make sure the /usr/local/ is writable by the user
sudo chown -R `whoami`:admin /usr/local/

#change permissions so other users can access to this
chmod -R ugo+rx $MW_MAIN_DIR

#old ffmpeg to read mjpg
curl http://ffmpegmac.net/resources/SnowLeopard_Lion_Mountain_Lion_Mavericks_27.03.2014.zip > $MW_MAIN_DIR/ffmpeg_old.zip
unzip $MW_MAIN_DIR/ffmpeg_old.zip ffmpeg
sudo mv ffmpeg /usr/local/bin/ffmpeg22
rm $MW_MAIN_DIR/ffmpeg_old.zip


#clone movement validation repository 
if [ ! -d $OPENWORM_DIR ]
	git clone https://github.com/openworm/open-worm-analysis-toolbox $OPENWORM_DIR
else
	cd $$OPENWORM_DIR
	git pull https://github.com/openworm/open-worm-analysis-toolbox
fi
#change permissions so other users can access to this
chmod -R ugo+rx $MW_MAIN_DIR/../open-worm-analysis-toolbox 


#install homebrew and other software used
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install wget cmake python3

#install pyqt5 for the GUI
#brew uninstall --force sip
brew install sip --with-python3
#brew uninstall --force pyqt5
brew install pyqt5 --with-python3
#brew uninstall --force pyqt
brew install pyqt --with-python3

#ffmpeg libraries, I need them to install opencv
#brew uninstall ffmpeg
brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-libass --with-libquvi --with-libvorbis --with-libvpx --with-opus --with-x265

#python dependencies
brew install homebrew/science/hdf5

pip3 install --upgrade numpy spyder tables pandas h5py matplotlib scipy scikit-learn \
scikit-image tifffile seaborn xlrd gitpython

#install opencv
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

make -j24
make install


#change permissions so other users can access to the python3 installation
#sudo chmod -R ugo+rx '/usr/local/'

#compile cython files
cd $MW_MAIN_DIR/MWTracker/trackWorms/segWormPython/cythonFiles/
make
make clean
cd $MW_MAIN_DIR