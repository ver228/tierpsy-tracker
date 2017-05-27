# Manual

## How to Start

After running installation script in OSX or Windows there should be a double-click executable named `TierpsyTracker` in the Desktop. If the executable is missing you can re-create it by running:

```bash
#(OSX/Linux)
installation/installation_script.sh--link_desktop 

#(Windows)
installation/installation_script.bat --link_desktop
```  

Alternatively open a terminal, go to the Tierpsy Tracker main directory and type: 

```bash
python3 cmd_scripts/TierpsyTrackerConsole.py
```

The main widget should look like the one below:

![TierpsyTrackerConsole](https://cloud.githubusercontent.com/assets/8364368/26398704/30c17b10-4072-11e7-9a90-d3e9e394ef9d.png)   

## Set Parameters

The purpose of this widget is to setup the parameters used in [Batch Processing Multiple Files](#batch-processing-multiple-files). The graphic interface is designed to help to select the parameters for [video compression](EXPLANATION.md/#COMPRESS). These parameters should be changed under different imaginig conditions.

The most likely parameter to be modified is `Threshold`. The selected value should be enough to exclude as much background as possible without lossing any part of the animals to be tracked. Below is an example on how one can use the interface to do this.

![SetParameters](https://cloud.githubusercontent.com/assets/8364368/26410507/6df7ef54-409b-11e7-8139-9ce99daf69cb.gif)  

In some cases, even after adjusting the threshold there still remain large regions of background. If the tracked objects significatively change position during the movie you can enable the background subtraction as show below. This method will consider background anything that do not change within the specified frame range, therefore if some of your animals are inmobile they will be lost. 

![SetBgndSubt](https://cloud.githubusercontent.com/assets/8364368/26410958/95a8c09a-409c-11e7-9fc9-14dafeabb467.gif)  

Other important parameters to set are:

* `Frame per Seconds` (fps) of your video. An important value since it is used to calculate several other parameters. If `Extract Timestamp` is set to `true`, the software will try to extract the fps from the video timestamp. However, keep it mind that it is not always possible to recover the correct timestamp, and therefore it is recommended give a value for the fps.
* `Frames to Average` to calculate the background mask. This value can significatively speed up the compression step. However, it will not work if the particles are highly motile. Use the buttons `Play` and `Next Chunck` to see how a selected value affects the mask. Note that the averaged is used only for the background mask, the foreground regions are kept intact for each individual frame. 
* `Microns per Pixels`. This value is only used to calculate the [skeleton features](EXPLANATION.md/#FEAT_CREATE), but the results will be in pixels instead of micrometers if the conversion factor is not setted.

You can access to all the parameters by clicking `Edit More Parameters`. The explanation of each parameter can be found by using the [contextual help](https://en.wikipedia.org/wiki/Tooltip). It is not trivial to adjust the parameters, but if you believe you need too, I recommend to use a small movie (few seconds) for testing.

When you are satisfied with the selected parameters select a file name and press `Save Parameters`. The parameters will be saved into a [JSON](http://json.org/) that can be used by [Batch Processing Multiple Files](#batch-processing-multiple-files). If you need to further modify a parameter you can either use a text editor or reload the file by dragging it to the Set Parameters widget.

## Batch Processing Multiple Files

It is a GUI for [Command Line Tool](#command-line-tool).
![BatchProcessing](https://cloud.githubusercontent.com/assets/8364368/26411227/4e788006-409d-11e7-8386-28235d859541.png)  

## Multi-Worm Tracker Viewer

![MWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412511/eac27158-40a0-11e7-880c-5671c2c27099.gif)  

![TrackJoined](https://cloud.githubusercontent.com/assets/8364368/26412212/e0e112f8-409f-11e7-867b-512cf044d717.gif) 

## Single-Worm Tracker Viewer
![SWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412826/e608bfea-40a1-11e7-9d3e-d0b8bf482db2.gif) 

## Command Line Tool

scripts are located in `cmd_scripts/`

```
$ python3 processMultipleFiles.py -h
usage: processMultipleFiles.py [-h] [--video_dir_root VIDEO_DIR_ROOT]
                               [--mask_dir_root MASK_DIR_ROOT]
                               [--results_dir_root RESULTS_DIR_ROOT]
                               [--tmp_dir_root TMP_DIR_ROOT]
                               [--videos_list VIDEOS_LIST]
                               [--json_file JSON_FILE]
                               [--max_num_process MAX_NUM_PROCESS]
                               [--pattern_include PATTERN_INCLUDE]
                               [--pattern_exclude PATTERN_EXCLUDE]
                               [--is_copy_video] [--copy_unfinished]
                               [--force_start_point FORCE_START_POINT]
                               [--end_point END_POINT] [--only_summary]
                               [--unmet_requirements]
                               [--refresh_time REFRESH_TIME]

Process worm video in the local drive using several parallel processes

optional arguments:
  -h, --help            show this help message and exit
  --video_dir_root VIDEO_DIR_ROOT
                        Root directory where the raw videos are located.
  --mask_dir_root MASK_DIR_ROOT
                        Root directory where the masked videos (after
                        COMPRESSION) are located or will be stored. If it is
                        not given it will be created replacing RawVideos by
                        MaskedVideos in the video_dir_root.
  --results_dir_root RESULTS_DIR_ROOT
                        ' Root directory where the tracking results are
                        located or will be stored. If it is not given it will
                        be created replacing MaskedVideos by Results in the
                        mask_dir_root.
  --tmp_dir_root TMP_DIR_ROOT
                        Temporary directory where the unfinished analysis
                        files are going to be stored.
  --videos_list VIDEOS_LIST
                        File containing the full path of the files to be
                        analyzed. If it is not given files will be searched in
                        video_dir_root or mask_dir_root using pattern_include
                        and pattern_exclude.
  --json_file JSON_FILE
                        File (.json) containing the tracking parameters.
  --max_num_process MAX_NUM_PROCESS
                        Maximum number of files to be processed
                        simultaneously.
  --pattern_include PATTERN_INCLUDE
                        Pattern used to search files to be analyzed.
  --pattern_exclude PATTERN_EXCLUDE
                        Pattern used to exclude files to be analyzed.
  --is_copy_video       Set **true** to copy the raw videos files to the
                        temporary directory.
  --copy_unfinished     Copy files to the final destination even if the
                        analysis was not completed successfully.
  --force_start_point FORCE_START_POINT
                        Force the program to start at a specific point in the
                        analysis.
  --end_point END_POINT
                        Stop the analysis at a specific point.
  --only_summary        Set **true** if you only want to see a summary of how
                        many files are going to be analyzed.
  --unmet_requirements  Use this flag if you only want to print the unmet
                        requirements of the invalid source files.
  --refresh_time REFRESH_TIME
                        Refresh time in seconds of the progress screen.
```
