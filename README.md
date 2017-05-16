# Tierpsy Tracker

This repository contains my work on the multiworm tracker for the [MRC-CSC](http://csc.mrc.ac.uk/) [Behavioral Genomics Group](http://behave.csc.mrc.ac.uk/).

# Installation
## Installation using Anaconda (OS X/Linux).

Run `bash installation/installation_script.sh`

There might be some problems with dependencies. It might be requiered to install from scratch ffmpeg and pyqt5, and there might be some issues to compile openCV. Open an issue and I might be able to help.

## Installation for Windows.
- Download and install [git tools for windows](https://git-scm.com/download/win). Make sure to select "Select Windows' default window". Otherwise you will have to use MinTTY as command line.
- Clone this repository and  [OpenWorm analysis toolbox](https://github.com/openworm/open-worm-analysis-toolbox):
```
git clone https://github.com/ver228/Multiworm_Tracking
git clone https://github.com/openworm/open-worm-analysis-toolbox
```
- Install [ffmpeg](https://ffmpeg.org/download.html). [Here](http://adaptivesamples.com/how-to-install-ffmpeg-on-windows/) are friendly installation instructions.
- Download [miniconda](http://conda.pydata.org/miniconda.html).
- Run `installation/installation_script.bat` .

## Test installation and examples.
The installation can be checked running the script [./Tests/run_test.py](https://github.com/ver228/Multiworm_Tracking/blob/master/Tests/run_tests.py). The example data used by the tests is downloaded by [installation_script.sh](https://github.com/ver228/Multiworm_Tracking/blob/master/installation_script.sh), see the variable EXAMPLES_LINK. The data can be downloaded manually using the following [link](https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1).

## Possible issues.
- If you recieve an error related with a module in segWormPython you will need to re-compile the cython files. Cython requires the same C compiler used to compile python. On OS X you need to install xcode using the app store. On Windows using python 3.5 you have to install [visual studio community 2015](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx) (use custom installation and select Visual C++).
