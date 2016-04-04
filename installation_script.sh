#!/bin/bash
MW_MAIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPENWORM_DIR=$MW_MAIN_DIR/../open-worm-analysis-toolbox
OPENCV_DIR=$MW_MAIN_DIR/../opencv

function create_directories {
	
	# make sure the /usr/local/ is writable by the user
	# sudo chown -R `whoami`:admin /usr/local/

	for DIRECTORY in $MW_MAIN_DIR $OPENWORM_DIR $OPENCV_DIR
	do
		echo $DIRECTORY
		mkdir -p $DIRECTORY
		#change permissions so other users can access to this folder
		chmod -R ugo+rx $DIRECTORY
	done

	echo 'Directories to store the helper repositories created.'
}

function copy_old_ffmpeg {
	cp $MW_MAIN_DIR/aux_files/ffmpeg22 /usr/local/bin/ffmpeg22
	chmod 777 /usr/local/bin/ffmpeg22
	echo 'Old version of .mjpeg used to read .mjpeg files copied as /usr/local/bin/ffmpeg22'
}

function clone_worm_analysis_toolbox {


	#if [ ! -d $OPENWORM_DIR ]
	git clone https://github.com/openworm/open-worm-analysis-toolbox $OPENWORM_DIR
	#else
	#	cd $OPENWORM_DIR
	#	git pull https://github.com/openworm/open-worm-analysis-toolbox
	#fi
	#change permissions so other users can access to this
	
	cd $OPENWORM_DIR
	git checkout 14577fc9c49183bf731a605df687c0739ed56657
	chmod -R ugo+rx $MW_MAIN_DIR/../open-worm-analysis-toolbox 

}

function install_dependencies_homebrew {
	#install homebrew and other software used
	ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	brew install wget cmake python3


	#ffmpeg libraries, I need them to install opencv
	brew install ffmpeg --with-fdk-aac --with-ffplay --with-freetype --with-libass --with-libquvi --with-libvorbis --with-libvpx --with-opus --with-x265

	#python dependencies
	brew install homebrew/science/hdf5
	brew install sip --with-python3 pyqt --with-python3 pyqt5 --with-python3
	
	#i prefer to install matplotlib and numpy with homebrew it gives less problems of compatilibity down the road
	brew install matplotlib --with-python3 numpy --with-python3

	pip3 install -U numpy spyder tables pandas h5py scipy scikit-learn \
		scikit-image tifffile seaborn xlrd gitpython

}

function compile_cython_files {
	cd $MW_MAIN_DIR/MWTracker/trackWorms/segWormPython/cythonFiles/
	make
	make clean
	cd $MW_MAIN_DIR
}

function install_opencv3 {
	if [ ! -z `python3 -c "import cv2"` ]; then
		echo 'Installing opencv.'
		#there is a brew formula for this, but there are more changes this will work.
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
	fi
}

function install_main_modules {
	cp $MW_MAIN_DIR/setup_openworm.py $OPENWORM_DIR/setup.py
	mv $OPENWORM_DIR/open_worm_analysis_toolbox/user_config_example.txt $OPENWORM_DIR/open_worm_analysis_toolbox/user_config.py
	python3 $OPENWORM_DIR/setup.py/$OPENWORM_DIR/setup.py develop
	python3 $MW_MAIN_DIR/setup.py develop
}

#get sudo permissions
sudo echo "Thanks."
create_directories
copy_old_ffmpeg
clone_worm_analysis_toolbox
install_dependencies
install_opencv3
compile_cython_files
install_main_modules


