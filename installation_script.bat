:: Install additional python dependecies.
conda install -y anaconda-client conda-build numpy matplotlib pytables pandas h5py scipy scikit-learn scikit-image seaborn xlrd statsmodels
pip install gitpython pyqt5
conda install -y -c https://conda.binstar.org/ver228 opencv3

:: Install packages
python setup.py develop
cd ..\open-worm-analysis-toolbox
python setup.py develop
rename .\open_worm_analysis_toolbox\user_config_example.txt user_config.py
cd ..\Multiworm_Tracking

:: Test installation
python -c "import cv2; import h5py; import MWTracker; import open_worm_analysis_toolbox"