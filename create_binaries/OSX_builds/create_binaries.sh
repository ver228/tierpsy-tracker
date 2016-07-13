function build {
pyinstaller --noconfirm  --clean \
compressSingleWorker.spec

pyinstaller --noconfirm  --clean \
trackSingleWorker.spec

pyinstaller --noconfirm  --clean \
MWConsole.spec
}

function build_spec {
pyinstaller --noconfirm  --clean \
--exclude-module PyQt4 \
--exclude-module PyQt4.QtCore \
--exclude-module PyQt4.QtGui \
--hidden-import=h5py.defs \
--hidden-import=h5py.utils \
--hidden-import=h5py.h5ac \
--hidden-import='h5py._proxy' \
../scripts/compressSingleWorker.py

pyinstaller --noconfirm  --clean \
--exclude-module PyQt4 \
--exclude-module PyQt4.QtCore \
--exclude-module PyQt4.QtGui \
--hidden-import=h5py.defs \
--hidden-import=h5py.utils \
--hidden-import=h5py.h5ac \
--hidden-import='h5py._proxy' \
../scripts/trackSingleWorker.py


pyinstaller --noconfirm  --clean --onefile --windowed \
--exclude-module PyQt4 \
--exclude-module PyQt4.QtCore \
--exclude-module PyQt4.QtGui \
--hidden-import=h5py.defs \
--hidden-import=h5py.utils \
--hidden-import=h5py.h5ac \
--hidden-import='h5py._proxy' \
../scripts/MWConsole.py
MWConsole.spec
../MWTracker_GUI/MWConsole.py
MWConsole_OSX.spec
--onefile --windowed

}

function clean {
	MWVER=`python3 -c "import MWTracker; print(MWTracker.__version__)"`
	OSXVER=`python3 -c "import platform; print(platform.mac_ver()[0])"`

	mv ./dist/MWConsole.app "../MWConsole $MWVER - MacOSX($OSXVER+).app"
	rm -Rf ./dist
	rm -Rf ./build	
}

build
clean

