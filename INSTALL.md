The repository dependencies are:

- ffmpeg
- openCV
- hdf5
- Python 3 + numpy, cv2, tables, h5py, pandas, matplotlib, scipy, scikit-learn, scikit-image, tifffile
- Cython including a compatible C/C++ compiler.
Additionally require to run `python3 setup.py build_ext --inplace` in directories `MWTracker/trackWorms/segWormPython/cythonFile` and `MWTracker/compressVideos/`.

All the steps from a clean installation for OSX are annotated in `./installation_script.sh`. Unfortunately, I cannot warranty this script will work since I have encountered problems in the OpenCV installation in some computers.

Finally, the repository requires to clone [movement_validation](https://github.com/openworm/movement_validation) from OpenWorm for the feature extraction.
You have to specify the absolute path of the cloned movement_validation repository by changing the variable movement_validation_dir in the config_param.py file on the 
 MRTracker directory.
