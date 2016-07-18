#!/bin/bash
MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
OPENCV_DIR=$MW_MAIN_DIR/../opencv
OPENCV_VER=3.1.0

function clone_worm_analysis_toolbox {
	git clone https://github.com/openworm/open-worm-analysis-toolbox $OPENWORM_DIR
	chmod -R ugo+rx $MW_MAIN_DIR/../open-worm-analysis-toolbox 
}

function install_dependencies_osx {
	xcode-select --install
	#install homebrew and other software used
	ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	#make the current user the owner of homebrew otherewise it can cause some problems
	sudo chown -R `whoami`:admin /usr/local/bin
	sudo chown -R `whoami`:admin /usr/local/share

	brew install wget cmake python3 git

	#ffmpeg libraries, needed to install opencv
	brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-libass --with-libquvi \
	--with-libvorbis --with-libvpx --with-opus --with-x265 --with-openh264 --with-tools --with-fdk-aac
	
	
	#python dependencies
	brew install homebrew/science/hdf5
	brew install sip --with-python3 pyqt --with-python3 pyqt5 --with-python3
	
	#i prefer to install matplotlib and numpy with homebrew it gives less problems of compatilibity down the road
	brew install homebrew/python/matplotlib --with-python3 
	brew install homebrew/python/numpy --with-python3
	
	pip3 install -U numpy spyder tables pandas h5py scipy scikit-learn \
		scikit-image tifffile seaborn xlrd gitpython psutil tiffcapture
}

function install_anaconda {

	curl -L http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh -o Anaconda3-4.1.1-Linux-x86_64.sh
	bash Anaconda3-4.1.1-Linux-x86_64.sh
	rm Anaconda3-4-4.1.1-Linux-x86_64.sh
	
	conda install -c ver228 opencv3
	pip install gitpython pyqt5
} 

function install_dependencies_linux {
	sudo apt install git
	sudo apt install ffmpeg
	install_anaconda

}

function compile_cython_files {
	cd $MW_MAIN_DIR/MWTracker/trackWorms/segWormPython/cythonFiles/
	make
	make clean
	cd $MW_MAIN_DIR
}

function install_opencv3 {
	echo 'Installing opencv.'
	#there is a brew formula for this, but there are more changes this will work.
	cd $MW_MAIN_DIR/..
	
	git clone https://github.com/Itseez/opencv
	#git clone https://github.com/Itseez/opencv_contrib

	cd $OPENCV_DIR
	git checkout -f $OPENCV_VER

	PY_VER=`python3 -c "import sys; print(sys.version.partition(' ')[0])"`
	PY_VER_SHORT=`python3 -c "import sys; print('.'.join(sys.version.partition(' ')[0].split('.')[0:2]))"`

	rm -rf build
	mkdir build
	cd build
	#for some weird reason i have to execute make twice or it does not find the python libraries directory
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
	..
	done
	make -j24
	make install
	make clean
}

function clean_prev_installation_osx {
	rm -Rf $OPENCV_DIR
	rm -Rf $OPENWORM_DIR
	pip3 uninstall -y numpy spyder tables pandas h5py scipy scikit-learn \
		scikit-image tifffile seaborn xlrd gitpython psutil
	brew uninstall --force wget cmake python3 git ffmpeg homebrew/science/hdf5 sip pyqt pyqt5
}

function install_main_modules {
	USER_CONFIG=$OPENWORM_DIR/open_worm_analysis_toolbox/user_config.py
	if [ ! -f $USER_CONFIG ]; then
		mv $OPENWORM_DIR/open_worm_analysis_toolbox/user_config_example.txt $USER_CONFIG
	fi
	cd $OPENWORM_DIR
	python3 setup.py develop

	cd $MW_MAIN_DIR
	python3 setup.py develop
}

SHORT_OS_STR=$(uname -s)
if [ "${SHORT_OS_STR}" == "Darwin" ]; then
	#clean_prev_installation_osx
	install_dependencies_osx
	if [[ ! $OPENCV_VER -eq `python3 -c "import cv2; print(cv2.__version__)"` ]]; then
		install_opencv3
	fi
	
fi

if [ "${SHORT_OS_STR:0:5}" == "Linux" ]; then
	install_dependencies_linux
fi

compile_cython_files
clone_worm_analysis_toolbox
install_main_modules