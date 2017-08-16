set MW_MAIN_DIR=%~dp0%..\
set OPENWORM_DIR=%MW_MAIN_DIR%..\open-worm-analysis-toolbox

if "%1" == "link_desktop" GOTO link_desktop

: install_openworm_toolbox
git clone https://github.com/openworm/open-worm-analysis-toolbox %OPENWORM_DIR%

cd %OPENWORM_DIR%
git pull
python setup.py develop
copy %OPENWORM_DIR%\open_worm_analysis_toolbox\user_config_example.txt %OPENWORM_DIR%\open_worm_analysis_toolbox\user_config.py

: install_tierpsy
cd %MW_MAIN_DIR%
python setup.py develop

: test
python -c "import cv2; import h5py; import tierpsy; import open_worm_analysis_toolbox"

: link_desktop
echo python %MW_MAIN_DIR%cmd_scripts\TierpsyTrackerConsole.py > %HOMEPATH%\Desktop\TierpsyTracker.bat

:: I do not want to execute this part. I leave it as documentation
GOTO end
: conda_depedencies
:: Install additional python dependecies.
:: Currently only python 3.5 works for tensorflow
conda install -y python=3.5 
conda install -y numpy matplotlib pytables pandas gitpython pyqt h5py scipy scikit-learn scikit-image seaborn xlrd cython statsmodels
conda install -y -c conda-forge tensorflow keras


: install_opencv
:: conda install --channel https://conda.anaconda.org/ver228 opencv3
conda install -y anaconda-client conda-build 
conda config --add channels menpo
conda build --no-anaconda-upload installation/menpo_conda-opencv3
conda install -y --use-local opencv3

: end