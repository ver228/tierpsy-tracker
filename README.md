# Multiworm_Tracking

This repository contains my work on the multiworm tracker for the [MRC-CSC](http://csc.mrc.ac.uk/) [Behavioral Genomics Group](http://behave.csc.mrc.ac.uk/).

# Usage

## cmd_scripts/

- See help.
```bash
python3 compressMultipleFiles.py -h
```

- Compress all the .mjpg files in the VIDEOS_ROOT_DIR directory tree and save the results in MASKS_ROOT_DIR.
```bash
python3 compressMultipleFiles.py VIDEOS_ROOT_DIR MASKS_ROOT_DIR
```

- Compress all the .avi files in the VIDEOS_ROOT_DIR directory tree and save the results in MASKS_ROOT_DIR.
```bash
python3 compressMultipleFiles.py VIDEOS_ROOT_DIR MASKS_ROOT_DIR --pattern_include '*.avi'
```

- Analyze only three videos at a time (default 6).
```bash
python compressMultipleFiles.py VIDEOS_ROOT_DIR MASKS_ROOT_DIR --max_num_process 3`
```

- Use the different parameters stores JSON_FILE in a json format. 
```bash
python3 compressMultipleFiles.py VIDEOS_ROOT_DIR MASKS_ROOT_DIR -json_file JSON_FILE
```

- Run the track analysis in all the .hdf5 masked videos found in the MASKS_ROOT_DIR. The results will be saved in a directory formed by replacing the folder /MaskedVideos/ in MASKS_ROOT_DIR by /Results/. If not /MaskedVideos/ is found /Results/ will be appended at the end of MASKS_ROOT_DIR.
```bash
python3 trackMultipleFiles.py MASKS_ROOT_DIR
```

- Do not filter skeletons by worm morphology.
```bash
python3 trackMultipleFiles.py MASKS_ROOT_DIR --no_skel_filter
```

- Stop the analysis just after the skeletons where oriented using the intensity along the worm.
```bash
python3 trackMultipleFiles.py MASKS_ROOT_DIR -end_point INT_SKE_ORIENT
```

## MWTracker_GUI/
- Run to view hdf5 video files. No any other output.
```bash
python3 HDF5videoViewer.py
```

- See the result of the analysis of the MWTracker and manually join worm trajectories.
```bash
python3 MWTrackerViewer.py
```
- See the result of the analysis of the MWTracker used in single worm mode.
```bash
python3 SWTrackerViewer.py
```
