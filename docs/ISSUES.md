# Known Issues

- In OSX Sierra I have recieve the following error when analysing *.avi videos.
  ```
  The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
  Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.
  ```
  This issues seems to be related to this [question](https://stackoverflow.com/questions/16254191/python-rpy2-and-matplotlib-conflict-when-using-multiprocessing) in stackoverflow. Apparently there is a conflict between multiprocessing and matplotlib. It can be solved by editing the `matplotlibrc` file and changing the backend line as `backend : Qt5Agg`.
