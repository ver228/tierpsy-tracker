#!/bin/bash
set -e

MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
OPENCV_DIR=$MW_MAIN_DIR/../opencv
OPENCV_VER="3.1.0"

#############
function install_dependencies_linux {
	case `lsb_release -si` in
		"Ubuntu")
		ubuntu_dependencies
		;;
		"RedHatEnterpriseWorkstation")
		redhat_dependencies
		;;
	esac
}

function ubuntu_dependencies {
	sudo apt-get install -y curl git ffmpeg
	#opencv3 dependencies http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html
	#[required]
	sudo apt-get install -y build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	#[optional]
	sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
}

function redhat_dependencies {
	sudo yum install git

	# opencv3 dependencies (http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_fedora/py_setup_in_fedora.html)
	sudo yum install cmake
	sudo yum install gcc gcc-c++
	sudo yum install gtk2-devel
	sudo yum install libdc1394-devel
	sudo yum install libv4l-devel
	sudo yum install ffmpeg-devel
	sudo yum install gstreamer-plugins-base-devel
	sudo yum install libpng-devel
	sudo yum install libjpeg-turbo-devel
	sudo yum install jasper-devel
	sudo yum install openexr-devel
	sudo yum install libtiff-devel
	sudo yum install libwebp-devel
	sudo yum install tbb-devel
	sudo yum install eigen3-devel
}

function install_dependencies_osx {
	xcode-select --install
	#install homebrew and other software used
	ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	
	#make the current user the owner of homebrew otherewise it can cause some problems
	#sudo chown -R `whoami`:admin /usr/local/bin
	#sudo chown -R `whoami`:admin /usr/local/share
	
	brew install git
	
	#ffmpeg libraries, needed to install opencv
	brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-libass --with-libquvi \
	--with-libvorbis --with-libvpx --with-opus --with-x265 --with-openh264 --with-tools --with-fdk-aac

	brew install jpeg libpng libtiff openexr eigen tbb

}

function install_anaconda {
	curl -L "$MINICONDA_LINK" -o miniconda_installer.sh
	bash miniconda_installer.sh -b -f -p $HOME/miniconda
	
	CONDA_PATH="$HOME/miniconda/bin"
	
	export PATH=$CONDA_PATH:$PATH
	hash -r

	#add the path to the bash profile only if it is not presented on the path
	if [[ ":$PATH:" != *":$CONDA_PATH:"* ]]; then
        echo "export PATH=$CONDA_PATH:\$PATH" >> $BASH_PROFILE_FILE
        source $BASH_PROFILE_FILE
    fi
	
	rm -f miniconda_installer.sh

	conda install -y anaconda-client conda-build numpy matplotlib pytables pandas \
	h5py scipy scikit-learn scikit-image seaborn xlrd statsmodels cython
	pip install gitpython pyqt5
}

function install_opencv3_anaconda {	
	conda install -y conda-build
	conda config --add channels menpo
	conda build --no-anaconda-upload menpo_conda-opencv3
	conda install -y --use-local opencv3
}

function compile_cython_files {
	cd $MW_MAIN_DIR/MWTracker/trackWorms/segWormPython/cythonFiles/
	make
	make clean
	cd $MW_MAIN_DIR
}

function install_main_modules {
	git clone https://github.com/openworm/open-worm-analysis-toolbox $OPENWORM_DIR
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


##########
OS=$(uname -s)
case "${OS}" in
	"Darwin")
	install_dependencies_osx || :
	MINICONDA_LINK="https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
	BASH_PROFILE_FILE=$HOME/.bash_profile
	;;
	
	"Linux"*)
	install_dependencies_linux || :
	MINICONDA_LINK="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
	BASH_PROFILE_FILE=$HOME/.bashrc
	;;
esac

install_anaconda
install_opencv3_anaconda
compile_cython_files
install_main_modules
