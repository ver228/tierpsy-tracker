# Installation

To install Tierpsy tracker, download the latest version to your local machine.  This can be done manually [here](https://github.com/ver228/tierpsy-tracker/archive/master.zip).  If you use Git or [Github Desktop](https://desktop.github.com/), then open a new terminal (in Windows open Git Shell) and run: 

```bash
git clone https://github.com/ver228/tierpsy-tracker
```
## Installation for OSX/Linux

Open a terminal window and run `bash installation/installation_script.sh`.

## Installation for Windows

- Download and install [miniconda](https://conda.io/miniconda.html).
- Install [ffmpeg](https://ffmpeg.org/download.html). Friendly installation instructions can be found [here](http://adaptivesamples.com/how-to-install-ffmpeg-on-windows/).
- Install [Visual C++ 2015 Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools).
- Run `./tierpsy-tracker/installation/installation_script.bat`.

## Possible Issues
Most of the problems that can occur during installation are due to missing/conflicting dependencies, especially if there were older versions of miniconda installed. Try updating miniconda and re-running the scripts. Many of these problems can be solved by searching for error messages online, but if problems persist, please raise an issue on Github [project page](https://github.com/ver228/tierpsy-tracker/issues).

## Test Examples
On Mac OSX or Linux, some test examples can be downloaded by running 

```bash
installation/instalation_script.sh --tests
```

The tests can also be manually downloaded using [this link](https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1). Uncompress the data and save it in the main repository folder `tests/data` .

You can then run the tests by running: 

```bash
python tests/run_tests.py
```