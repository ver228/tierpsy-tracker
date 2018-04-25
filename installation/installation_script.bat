set MW_MAIN_DIR=%~dp0%..\
set OPENWORM_DIR=%MW_MAIN_DIR%..\open-worm-analysis-toolbox
set TIERPSYFEATURES_DIR=%MW_MAIN_DIR%..\tierpsy-features

if "%1" == "link_desktop" GOTO link_desktop

: install_openworm_toolbox
git clone https://github.com/openworm/open-worm-analysis-toolbox %OPENWORM_DIR%
cd %OPENWORM_DIR%
git pull origin HEAD
python setup.py develop

copy %OPENWORM_DIR%\open_worm_analysis_toolbox\user_config_example.txt %OPENWORM_DIR%\open_worm_analysis_toolbox\user_config.py

: install_tierpsy_features
git clone https://github.com/ver228/tierpsy-features %TIERPSYFEATURES_DIR%
cd %TIERPSYFEATURES_DIR%
git pull origin HEAD
python setup.py develop

: install_tierpsy
cd %MW_MAIN_DIR%
python setup.py develop
python setup.py clean --all

: test
python -c "import cv2; import h5py; import tierpsy; import open_worm_analysis_toolbox; import tierpsy_features"

: link_desktop
echo python %MW_MAIN_DIR%cmd_scripts\TierpsyTrackerConsole.py > %HOMEPATH%\Desktop\TierpsyTracker.bat
