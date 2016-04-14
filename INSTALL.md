
# Requirements.
- Python 3 + numpy matplotlib pytables pandas h5py scipy scikit-learn scikit-image seaborn xlrd pyqt cython
- ffmpeg
- openCV3
- hdf5
- Cython including a compatible C/C++ compiler.
- Additionally it requires to clone the [open-worm-analysis-toolbox](https://github.com/openworm/open-worm-analysis-toolbox) from OpenWorm for the feature extraction.

# Installation for OS X
Run `./installation_script.sh`. If it is not a clean installation, I cannot warranty this script will work since I have encountered conflict with previous versions of the libraries installed by homebrew. In the future I might develop a script that allow to remove previous installations.
 
# Anaconda Installation.
- Download [anaconda](https://www.continuum.io/downloads) or [miniconda](http://conda.pydata.org/miniconda.html) with python 3.5.
```
conda install -y anaconda-client conda-build numpy matplotlib pytables pandas h5py scipy scikit-learn scikit-image seaborn xlrd pyqt cython
pip install gitpython
conda install -c https://conda.binstar.org/menpo opencv3
```
- clone [OpenWorm analysis toolbox](https://github.com/openworm/open-worm-analysis-toolbox).
- Go to open-worm-analysis-toolbox directory and run `python3 setup.py develop`
- Go to open-worm-analysis-toolbox/open_worm_analysis_toolbox  and change the file `user_config_example.txt` to `user_config.py`
- Go the the Multiworm_Tracking directory and run `python3 setup.py develop`

- Try to run `python3 -c "import cv2; import h5py; import MWTracker; import open_worm_analysis_toolbox"`. You should recieve no error messages.
##Issues
- The opencv library provided in the menpo repository does not contain ffmpeg support. This means you would not be able to read most of the video formats, and therefore you will not be able to convert video to the .hdf5 format. The rest of the analysis and the GUI should work fine.
- If you recieve an error related with a module in segWormPython you will need to compile the cython files. It will require the same C compiler used to compile python. On OS X you need to install xcode using the app store. On Windows using python 3.5 you have to install [visual studio community 2015](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx) (use custom installation and select Visual C++). Then run `python3 setup.py build_ext --inplace` in the directory `Multiworm_Tracking/MWTracker/trackWorms/segWormPython/cythonFile`, and try to check again if it was a succesful installation.

