#!/bin/bash
MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
OPENCV_DIR=$MW_MAIN_DIR/../opencv
OPENCV_VER="3.1.0"
SHORT_OS_STR=$(uname -s)

#############
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
	#brew install homebrew/python/numpy --with-python3
	pip3 install -U numpy spyder tables pandas h5py scipy scikit-learn \
	scikit-image tifffile seaborn xlrd gitpython psutil
	
	CURRENT_OPENCV_VER=`python3 -c "import cv2; print(cv2.__version__)"`
	if [ $OPENCV_VER != $CURRENT_OPENCV_VER ]; then
		install_opencv3
	fi
}

function install_anaconda {
	curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda_installer.sh
	bash miniconda_installer.sh -b -p $HOME/miniconda
	export PATH="$HOME/miniconda/bin:$PATH"
	rm -f miniconda_installer.sh

	conda install -y anaconda-client conda-build numpy matplotlib pytables pandas \
	h5py scipy scikit-learn scikit-image seaborn xlrd statsmodels
	pip install gitpython pyqt5
}

function install_opencv3_anaconda {	
	conda install -y conda-build
	conda config --add channels menpo
	conda build menpo_conda-opencv3
	conda install -y --use-local opencv3
}

function install_dependencies_linux {
	case `lsb_release -si` in
		"Ubuntu")
		ubuntu_dependencies
		;;
		"RedHatEnterpriseWorkstation")
		redhad_dependencies
		;;
	esac

	install_anaconda
	install_opencv3_anaconda
}

function ubuntu_dependencies {
	sudo apt-get install -y curl git ffmpeg
	#opencv3 dependencies http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html
	#[required]
	sudo apt-get install -y build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	#[optional]
	sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
}

function redhad_dependencies {
	yum install git

	# opencv3 dependencies (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_fedora/py_setup_in_fedora.html)
	yum install cmake
	yum install gcc gcc-c++
	yum install gtk2-devel
	yum install libdc1394-devel
	yum install libv4l-devel
	yum install ffmpeg-devel
	yum install gstreamer-plugins-base-devel
	yum install libpng-devel
	yum install libjpeg-turbo-devel
	yum install jasper-devel
	yum install openexr-devel
	yum install libtiff-devel
	yum install libwebp-devel
	yum install tbb-devel
	yum install eigen3-devel
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
	rm -rf build
	mkdir build
	cd build
	#for some weird reason i have to execute make twice or it does not find the python libraries directory
	for i in 1 2
	do
	cmake '"Unix Makefile"' -DBUILD_opencv_python3=ON \
	-DBUILD_opencv_python2=OFF \
	-DPYTHON_EXECUTABLE=`which python3` \
	-DPYTHON3_INCLUDE_DIR=`python3 -c "import sysconfig; print(sysconfig.get_path('platinclude'))"` \
	-DPYTHON3_LIBRARY=`python3 -c "import sysconfig; print(sysconfig.get_path('platstdlib'))"` \
	-DPYTHON3_PACKAGES_PATH=`python3 -c "import sysconfig; print(sysconfig.get_path('platlib'))"` \
	-DPYTHON3_NUMPY_INCLUDE_DIRS=`python3 -c "from numpy.distutils.misc_util import get_numpy_include_dirs; print(get_numpy_include_dirs()[0])"` \
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

##########
if [ "${SHORT_OS_STR}" == "Darwin" ]; then
	#clean_prev_installation_osx
	install_dependencies_osx
fi

if [ "${SHORT_OS_STR:0:5}" == "Linux" ]; then
	install_dependencies_linux
fi

compile_cython_files
clone_worm_analysis_toolbox
install_main_modules
