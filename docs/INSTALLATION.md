*The instructions below are to install Tierpsy Tracker from the source code. I would recommend to do this only if you are using Linux or want to run the development version, otherwise use the double-click executables available for Windows (7 or latest) and OSX (Yosemite or latest) in the [releases page](https://github.com/ver228/tierpsy-tracker/releases).*

# System Requirements 
- Freshly installed [miniconda] (https://conda.io/miniconda.html) or at least setup up a new enviroment.
- Optional [ffmpeg](https://ffmpeg.org/download.html): ffprobe must be accessible from the command line to calculate the video timestamps.
- [C compiler compatible with cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html). In Windows, you can use [Visual C++ 2015 Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools). In OSX, if you install [homebrew](https://brew.sh/) it will setup the C compiler without the need to download XCode from the appstore. 
- [Git](https://git-scm.com/). [Here](https://gist.github.com/derhuerst/1b15ff4652a867391f03) are some instructions to install it.

# Installation

1. Clone this repository either using the [Github Desktop](https://desktop.github.com/) or from the command line as:
```bash
git clone https://github.com/ver228/tierpsy-tracker
```
 
2. Install the conda dependencies from the conda-forge channel:
```bash
conda config --add channels conda-forge 

conda install -y numpy matplotlib pytables pandas gitpython pyqt=5 \
h5py scipy scikit-learn scikit-image seaborn xlrd cython statsmodels

conda install -y -c conda-forge keras opencv
pip install tensorflow 
```

3. Install the rest of the modules:
On the tierpsy-tracker root folder (the folder with the cloned repository) type:
```bash
bash installation/installation_script.sh #OSX or Linux

installation/installation_script.bat #Windows
```

## Possible Issues
- The most common problem in the installation is OpenCV (error in import cv2). Try a fresh miniconda installation (or a fresh enviroment) and make sure your are using the [conda-forge](https://conda-forge.org/) packages. It this does not work I am afraid you would have to solve the problem by yourself (Google is your friend).

- You do not need to install manually the [Open Worm Analysis Toolbox](https://github.com/openworm/open-worm-analysis-toolbox). However if you do (and I do not recommend it), be aware that there is a bug with the pip installer: it is missing some dependencies and it will create a corrupt [.egg](https://stackoverflow.com/questions/2051192/what-is-a-python-egg) in your packages folder. Manually delete the .egg (use the error traceback to find the its location) and re-run `installation_script.sh`. The script will download the Open Worm Analysis Toolbox repository and install it using `python setup.py develop`. 


# Test Examples
On Mac OSX or Linux, some test examples can be downloaded by running 

```bash
installation/installation_script.sh --download_examples
```

The tests can also be manually downloaded using [this link](https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1). Uncompress the data and save it in the main repository folder `tests/data` .

You can then run the tests by running: 

```bash
python tests/run_tests.py
```
