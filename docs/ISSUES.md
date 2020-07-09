# Known Issues

- In OSX Sierra I have received the following error when analysing `*.avi` videos.
  ```
  The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
  Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.
  ```
  This issues seems to be related to this [question](https://stackoverflow.com/questions/16254191/python-rpy2-and-matplotlib-conflict-when-using-multiprocessing) in stackoverflow. Apparently there is a conflict between multiprocessing and matplotlib. It can be solved by editing the `matplotlibrc` file and changing the backend line as `backend : Qt5Agg`.

- In Windows, when installing Tierpsy from source, we have seen the following error:
  ```
  error: command 'cl.exe' failed: No such file or directory
  ```
  This happens when pip can not find the necessary C/C++ components.
  If a [compatible C/C++ compiler](https://visualstudio.microsoft.com/visual-cpp-build-tools/) is installed on the machine, then `cl.exe` will exist, but for some reason `pip` cannot find it.
  The steps below have solved the problem for us:
  - Open the Anaconda Prompt
    - Activate the conda environment with `conda activate tierpsy`
    - Find the full path for conda by typing `where conda`
    - keep the prompt open
  - Open the x86_x64 Cross Tools Command Prompt (Windows key -> scroll down until you find the `Visual Studio` folder, it should be there)
    - Activate the Tierpsy environment with `C:\path\to\conda activate tierpsy`, where `C:\path\to\conda` is the path you obtained in the Anaconda Prompt with the `where conda` command.
    - `(tierpsy)` should appear at the beginning of the Prompt
    - `pip install -e C:\path\on\your\machine\to\the\git\folder\named\tierpsy-tracker\`
    - You should get `Successfully installed tierpsy`
    - close this x86_x64 Cross Tools Prompt
  - In the Anaconda prompt:
    - `tierpsy_gui`


- On Windows machines **without a CUDA capable GPU**, Tierpsy throws the following error during the analysis:
  ```
  OSError: [WinError 126] The specified module could not be found
  ```
  The steps below have fixed the issue for us:
  - In the Anaconda Prompt
    - Activate the conda environment with `conda activate tierpsy`
    - Install the CPU-only version of pytorch: `conda install pytorch torchvision cpuonly -c pytorch`
    - `tierpsy_gui`
