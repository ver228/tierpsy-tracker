#!/bin/bash
set -e

MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
OPENCV_DIR=$MW_MAIN_DIR/../opencv
OPENCV_VER="3.1.0"

EXAMPLES_LINK="https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1"
EXAMPLES_DIR="$MW_MAIN_DIR/Tests/Data/"

OS=$(uname -s)

#############
function install_homebrew_python {
	if [[ -z `brew ls --versions python3` ]]; then
		brew install python3
	fi

	if [[ -z `brew ls --versions hdf5` ]]; then
		#required for pytables
		brew install homebrew/science/hdf5
	fi

	#python dependencies
	pip3 install -U numpy tables pandas h5py scipy scikit-learn \
	scikit-image seaborn xlrd gitpython cython matplotlib pyqt5
	
	#i prefer to install matplotlib and numpy with homebrew it gives less problems of compatilibity down the road
	#brew install homebrew/python/matplotlib --with-python3
	
	CURRENT_OPENCV_VER=`python3 -c "import cv2; print(cv2.__version__)" 2> /dev/null || true`
	if [[ $OPENCV_VER != $CURRENT_OPENCV_VER ]]; then
		install_opencv3
	fi
}

function install_opencv3 {
	if [[ -z `brew ls --versions cmake` ]]; then
		brew install cmake 
	fi

	echo 'Installing opencv.'
	#there is a brew formula for this, but there are more changes this will work.
	if ! [[ -d $OPENCV_DIR ]] ; then
		cd $MW_MAIN_DIR/..
		git clone https://github.com/Itseez/opencv	
	fi
	
	cd $OPENCV_DIR
	git checkout -f $OPENCV_VER
	
	#remove build directory if it existed before
	rm -rf build || : 
	mkdir build && cd build

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
	-DBUILD_TIFF=ON 
	-DBUILD_opencv_java=OFF 
	-DWITH_CUDA=OFF 
	-DENABLE_AVX=ON 
	-DWITH_OPENGL=ON 
	-DWITH_OPENCL=ON \
	-DWITH_IPP=ON \
	-DWITH_TBB=ON \
	-DWITH_EIGEN=ON \
	-DWITH_V4L=ON \
	-DBUILD_TESTS=OFF \ 
	-DBUILD_PERF_TESTS=OFF \
	-DWITH_QT=OFF \
	-DINSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-DCMAKE_BUILD_TYPE=RELEASE \
	..
	done
	make -j24
	make install
	make clean

	#make a dirty test
	python3 -c "import cv2; print(cv2.__version__)"
}

function clean_prev_installation_osx {
	rm -Rf $OPENCV_DIR
	rm -Rf $OPENWORM_DIR
	brew uninstall --force cmake python3 git ffmpeg homebrew/science/hdf5 #sip pyqt5
}


function install_dependencies_linux {
	case `lsb_release -si` in
		"Ubuntu")
		ubuntu_dependencies
		;;
		"RedHat*"|"CentOS")
		redhat_dependencies
		;;
	esac
}

function ubuntu_dependencies {
	sudo apt-get install -y curl git ffmpeg unzip
	#opencv3 dependencies http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html
	#[required]
	sudo apt-get install -y build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	#[optional]
	sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

}

function redhat_dependencies {
	sudo yum -y install git

	# opencv3 dependencies (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_fedora/py_setup_in_fedora.html)
	sudo yum -y install gcc gcc-c++ cmake gtk2-devel libdc1394-devel libv4l-devel ffmpeg-devel \
	gstreamer-plugins-base-devel libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel \
	libtiff-devel libwebp-devel tbb-devel eigen3-devel
}

function install_dependencies_osx {
	xcode-select --install
	#install homebrew and other software used
	ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	
	#make the current user the owner of homebrew otherewise it can cause some problems
	#sudo chown -R `whoami`:admin /usr/local/bin
	#sudo chown -R `whoami`:admin /usr/local/share
	brew update
	brew upgrade --verbose

	brew install git
	
	#ffmpeg libraries, needed to install opencv
	brew install ffmpeg --verbose --with-fdk-aac --with-libass --with-libquvi \
	--with-libvorbis --with-libvpx --with-x265 --with-openh264 --with-tools --with-fdk-aac

	#image libraries for opencv
	brew install jpeg libpng libtiff openexr eigen tbb
}

function install_anaconda {
	case "${OS}" in
		"Darwin")
		MINICONDA_LINK="https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
		BASH_PROFILE_FILE=$HOME/.bash_profile
		;;
		
		"Linux"*)
		MINICONDA_LINK="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
		BASH_PROFILE_FILE=$HOME/.bashrc
		;;
	esac

	curl -L "$MINICONDA_LINK" -o miniconda_installer.sh
	bash miniconda_installer.sh -b -f -p $HOME/miniconda
	rm -f miniconda_installer.sh
	
	CONDA_PATH="$HOME/miniconda/bin"
	#add the path to the bash profile only if it is not presented on the path
	if [[ ":$PATH:" != "*:$CONDA_PATH:*" ]]; then
        echo "export PATH=$CONDA_PATH:\$PATH" >> $BASH_PROFILE_FILE
        source $BASH_PROFILE_FILE
        export PATH=$CONDA_PATH:$PATH
        hash -r
    fi

	conda install -y anaconda-client conda-build numpy matplotlib pytables pandas \
	h5py scipy scikit-learn scikit-image seaborn xlrd cython
	pip install gitpython pyqt5

	install_opencv3_anaconda
}

function install_opencv3_anaconda {	
	conda install -y conda-build
	conda config --add channels menpo
	conda build --no-anaconda-upload installation/menpo_conda-opencv3
	conda install -y --use-local opencv3
	python3 -c "import cv2; print(cv2.__version__)"
}

function compile_cython_files {
	cd $MW_MAIN_DIR/MWTracker/trackWorms/segWormPython/cythonFiles/
	make
	make clean
	cd $MW_MAIN_DIR
}

function install_main_modules {
	git clone https://github.com/openworm/open-worm-analysis-toolbox $OPENWORM_DIR || :
	cd $OPENWORM_DIR
	git pull origin HEAD || :
	
	cd $MW_MAIN_DIR
	chmod -R ugo+rx $MW_MAIN_DIR/../open-worm-analysis-toolbox

	USER_CONFIG=$OPENWORM_DIR/open_worm_analysis_toolbox/user_config.py
	if [ ! -f $USER_CONFIG ]; then
	mv $OPENWORM_DIR/open_worm_analysis_toolbox/user_config_example.txt $USER_CONFIG
	fi
	
	cd $OPENWORM_DIR
	python3 setup.py develop
	
	cd $MW_MAIN_DIR
	python3 setup.py develop
}

function download_examples {
	curl -L $EXAMPLES_LINK -o test_data.zip
	rm -rf $EXAMPLES_DIR || : 
	mkdir $EXAMPLES_DIR
	unzip test_data.zip -d $EXAMPLES_DIR
	rm test_data.zip
}

##########


case "${OS}" in
	"Darwin")
	install_dependencies_osx || :
	;;
	
	"Linux"*)
	install_dependencies_linux || :
	;;
esac

if [[ $1 == 'brew' ]]; then
	install_homebrew_python
else
	install_anaconda
fi

compile_cython_files
install_main_modules
download_examples
