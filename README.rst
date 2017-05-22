.. _MRC-LMS: http://lms.mrc.ac.uk/
.. _`Behavioral Genomics Group` : http://behave.csc.mrc.ac.uk/
.. _`Github Desktop` : https://desktop.github.com/
.. _miniconda : https://conda.io/miniconda.html
.. _ffmpeg : https://ffmpeg.org/download.html
.. _`Visual C++ 2015 Build Tools` : http://landinghub.visualstudio.com/visual-cpp-build-tools
Tierpsy Tracker
===============

This repository contains the code of the multiworm tracker for the MRC-LMS_ `Behavioral Genomics Group`_.

For more details please visit the repository [wiki](https://github.com/ver228/tierpsy-tracker/wiki).

Installation
============
First you have to clone this repository. I recommend to install `Github Desktop`_, open a new terminal (in Windows open Git Shell), and run:
```
git clone https://github.com/ver228/tierpsy-tracker
```


Installation for OS X/Linux
----------------------------------------
- Run `bash installation/installation_script.sh`

Installation for Windows
========================
- Download and install miniconda_.
- Install ffmpeg_. Friendly installation instructions can be found at http://adaptivesamples.com/how-to-install-ffmpeg-on-windows/ .
- Install `Visual C++ 2015 Build Tools`_ .
- Run `./tierpsy-tracker/installation/installation_script.bat` .

Test Installation and Examples
==============================
The test data can be downloaded using
```
installation/instalation_script.sh --tests
```
or manually downloaded at https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1 .

There after the installation can be checked running
```
python tests/run_tests.py
```

Possible Issues
===============
Most of the problems that can occur during the installation are due to missing/conflicting dependencies specially if there were older versions of miniconda installed. Try to upgrade and re-run the scripts. If the problem persist you can raise an issue I'll try to help you but most of the time the problems can be solved by doing a google search.
