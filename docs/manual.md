#Manual


## How to Start

After running installation script in OSX or Windows there should be a doulbe-click executable named `TierpsyTracker` in the Desktop. If the executable is missing you can re-create the executable by running:

```
installation/installation_script.sh--link_desktop 
#(OSX/Linux)

installation/installation_script.bat --link_desktop
#Windows
```  

Alternatively open a terminal, go to the Tierpsy Tracker main directory and type: 

```
python3 cmd_scripts/TierpsyTrackerConsole.py
```

![TierpsyTrackerConsole](https://cloud.githubusercontent.com/assets/8364368/26398704/30c17b10-4072-11e7-9a90-d3e9e394ef9d.png)   

##Set Parameters

This widget is used to setup the parameters used by [Batch Processing Multiple Files](#batch-processing-multiple-files) Explanation of each parameter can be found by 

![SetParameters](https://cloud.githubusercontent.com/assets/8364368/26410507/6df7ef54-409b-11e7-8139-9ce99daf69cb.gif)  

![SetBgndSubt](https://cloud.githubusercontent.com/assets/8364368/26410958/95a8c09a-409c-11e7-9fc9-14dafeabb467.gif)  


## Batch Processing Multiple Files


It is a GUI for [Command Line Tool](#command-line-tool).
![BatchProcessing](https://cloud.githubusercontent.com/assets/8364368/26411227/4e788006-409d-11e7-8386-28235d859541.png)  


## Multi-Worm Tracker Viewer

![MWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412511/eac27158-40a0-11e7-880c-5671c2c27099.gif)  

![TrackJoined](https://cloud.githubusercontent.com/assets/8364368/26412212/e0e112f8-409f-11e7-867b-512cf044d717.gif) 

## Single-Worm Tracker Viewer
![SWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412826/e608bfea-40a1-11e7-9d3e-d0b8bf482db2.gif) 


## Command Line Tool

scripts are located in `cmd_scripts/`

``` 
  usage: processMultipleFiles.py [-h] [--videos_list VIDEOS_LIST]
                                 [--json_file JSON_FILE]
                                 [--tmp_dir_root TMP_DIR_ROOT]
                                 [--max_num_process MAX_NUM_PROCESS]
                                 [--refresh_time REFRESH_TIME]
                                 [--pattern_include PATTERN_INCLUDE]
                                 [--pattern_exclude PATTERN_EXCLUDE]
                                 [--only_summary] [--unmet_requirements]
                                 [--copy_unfinished]
                                 [--video_dir_root VIDEO_DIR_ROOT]
                                 [--mask_dir_root MASK_DIR_ROOT]
                                 [--results_dir_root RESULTS_DIR_ROOT]
                                 [--is_copy_video] [--use_manual_join]
                                 [--analysis_type {compress,track,all} | --analysis_checkpoints {COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE} [{COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE} ...]]
                                 [--force_start_point {COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE}]
                                 [--end_point {COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE}]

  Process worm video in the local drive using several parallel processes

  optional arguments:
    -h, --help            show this help message and exit
    --videos_list VIDEOS_LIST
                          File containing the full path of the videos to be
                          analyzed, otherwise there will be search from root
                          directory using pattern_include and pattern_exclude.
    --json_file JSON_FILE
                          File (.json) containing the tracking parameters.
    --tmp_dir_root TMP_DIR_ROOT
                          Temporary directory where files are going to be
                          stored.
    --max_num_process MAX_NUM_PROCESS
                          Max number of process to be executed in parallel.
    --refresh_time REFRESH_TIME
                          Refresh time in seconds of the process screen.
    --pattern_include PATTERN_INCLUDE
                          Pattern used to find the valid video files in
                          video_dir_root
    --pattern_exclude PATTERN_EXCLUDE
                          Pattern used to exclude files in video_dir_root
    --only_summary        Use this flag if you only want to print a summary of
                          the files in the directory.
    --unmet_requirements  Use this flag if you only want to print the unmet
                          requirements in the invalid source files.
    --copy_unfinished     Copy files from an uncompleted analysis in the
                          temporary directory.
    --video_dir_root VIDEO_DIR_ROOT
                          Root directory with the raw videos.
    --mask_dir_root MASK_DIR_ROOT
                          Root directory with the masked videos. It must the
                          hdf5 from a previous compression step.
    --results_dir_root RESULTS_DIR_ROOT
                          Root directory where the tracking results will be
                          stored. If not given it will be estimated from the
                          mask_dir_root directory.
    --is_copy_video       The video file would be copied to the temporary
                          directory.
    --use_manual_join     Use this flag to calculate features on manually joined
                          data.
    --analysis_type {compress,track,all}
                          Type of analysis to be processed.
    --analysis_checkpoints {COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE} [{COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE} ...]
                          List of the points to be processed.
    --force_start_point {COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE}
                          Force the program to start at a specific point in the
                          analysis.
    --end_point {COMPRESS,COMPRESS_ADD_DATA,VID_SUBSAMPLE,TRAJ_CREATE,TRAJ_JOIN,SKE_INIT,BLOB_FEATS,SKE_CREATE,SKE_FILT,SKE_ORIENT,STAGE_ALIGMENT,CONTOUR_ORIENT,INT_PROFILE,INT_SKE_ORIENT,FEAT_CREATE,WCON_EXPORT,FEAT_MANUAL_CREATE}
                          End point of the analysis.```
```