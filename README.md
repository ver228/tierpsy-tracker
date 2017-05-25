#Tierpsy Tracker

This repository contains the code of the multiworm tracker for the [MRC-LMS](http://lms.mrc.ac.uk/) [Behavioral Genomics Group](http://behave.csc.mrc.ac.uk/)

##Installation

First you have to clone this repository. I recommend to install [Github Desktop](https://desktop.github.com/). Then open a new terminal (in Windows open Git Shell) and run: 

```
git clone https://github.com/ver228/tierpsy-tracker
```

### Installation for OSX/Linux

Run `bash installation/installation_script.sh`


### Installation for Windows

- Download and install [miniconda](https://conda.io/miniconda.html).
- Install [ffmpeg](https://ffmpeg.org/download.html). Friendly installation instructions can be found [here](http://adaptivesamples.com/how-to-install-ffmpeg-on-windows/).
- Install [Visual C++ 2015 Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools).
- Run `./tierpsy-tracker/installation/installation_script.bat`.

### Possible Issues
Most of the problems that can occur during the installation are due to missing/conflicting dependencies specially if there were older versions of miniconda installed. Try to upgrade and re-run the scripts. If the problem persist you can raise an issue I'll try to help you but most of the time the problems can be solved by doing a google search.


## Tests Examples


In OSX/Linux the test examples can be downloaded executing: 

```
installation/instalation_script.sh --tests
```

It can be also manually downloaded using [this link](https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1). Uncompress the data and save it in the main repository folder `tests/data` .

You can run the tests by running: 

```
python tests/run_tests.py
```

