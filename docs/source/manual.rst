************
Manual
************

How to Start
==============

After running installation script in OSX or Windows there should be a doulbe-click executable named `TierpsyTracker` in the Desktop. If the executable is missing you can run ``installation/installation_script.sh--link_desktop`` in OSX or ``installation/installation_script.bat --link_desktop`` in Windows to re-create the executable.

The alternative is to open a terminal, move to the Tierpsy Tracker main directory and type ``python3 cmd_scripts/TierpsyTrackerConsole.py``.


.. image:: https://cloud.githubusercontent.com/assets/8364368/26286115/0b0c7376-3e55-11e7-918c-cc0319b90496.png
   :align: center
   


Set Parameters
==============

 - If the worm density is high, the worm occupies a large area of the field of view or the raw video is already heavily compressed, the output hdf5 file can be larger than the original video.
 - The hdf5 storage of the masked images is important in our setup: the high resolution and high-througput make even jpg compressed videos too large to be kept for long time storage. However, in the future this step might be done in real time in our system. 

.. image:: https://cloud.githubusercontent.com/assets/8364368/26286358/793a423c-3e5b-11e7-8e8f-f94da9c26ba9.gif

.. image:: https://cloud.githubusercontent.com/assets/8364368/26287848/475e64f8-3e7b-11e7-8a1c-d4d94dbbcf59.gif


Batch Processing Multiple Files
===============================

It is a GUI for command line tool ProcessMultipleFiles.py.

Single-Worm Tracker Viewer
===============================

Multi-Worm Tracker Viewer
===============================

Command Line Scripts
===============================

scripts are located in **cmd_scripts/**

.. code-block:: bash
  :linenos:

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
