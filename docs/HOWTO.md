## Example Data

Example files can be found [here](https://imperiallondon-my.sharepoint.com/:u:/g/personal/ajaver_ic_ac_uk/EdoZILCgkw5HnTkMtbD_OEsBBSbXD_WnBnWkwdf2bojusg?e=3yKbXR). The zip file contains a multiworm video recorded using a high resolution fixed camera and a single worm video recorded using the [WT2.0](https://www.mrc-lmb.cam.ac.uk/wormtracker/).

You can analyze the videos using the [Batch Processing Multiple Files](#batch-processing-multiple-files) App. The videos require different analysis parameters since they belong to different setups, therefore they cannot be processed together.

For the multiworm video the `Parameters File` must be set to `MULTI_RIG.json` and the `File Pattern to Include` as `*.mov` as shown below:

<img width="450" alt="screen shot 2018-04-25 at 09 04 05" src="https://user-images.githubusercontent.com/8364368/39233808-6344ddca-4869-11e8-9c34-d3db9102109f.png">

For the multiworm video the `Parameters File` must be set to `WT2_clockwise.json` and the `File Pattern to Include` as `*.avi` as shown below:

<img width="450" alt="screen shot 2018-04-25 at 09 13 19" src="https://user-images.githubusercontent.com/8364368/39233903-9a07ec62-4869-11e8-921e-27769f3dc87c.png">

The processing times for in MacBook Pro (15-inch, 2017) were 04:31 minutes for the multiworm video and 11:43 minutes for the singleworm video.

# Detailed Instructions

## Getting Started

If you are using any of the [binary executables](https://github.com/ver228/tierpsy-tracker/releases) you only have to double click the file and the program should start after a few seconds.

If you [installed tierpsy tracker from the source](https://github.com/ver228/tierpsy-tracker/blob/master/docs/INSTALLATION.md) in OSX or Windows there should be a clickable executable named `TierpsyTracker` on the Desktop. If the executable is missing you can re-create it by running:

```bash
#(OSX/Linux)
installation/installation_script.sh --link_desktop 

#(Windows)
installation/installation_script.bat --link_desktop
```  

Alternatively open a terminal, go to the directory where Tierpsy Tracker is installed and type: 

```bash
python3 cmd_scripts/TierpsyTrackerConsole.py
```

The main widget should look like the one below:

![TierpsyTrackerConsole](https://cloud.githubusercontent.com/assets/8364368/26624637/64275e1c-45e9-11e7-8bd6-69a386007d89.png)   

## Set Parameters

The purpose of this widget is to setup the parameters used for [Batch Processing Multiple Files](#batch-processing-multiple-files). The interface is designed to help select the parameters that determine how videos are [segmented and compressed](EXPLANATION.md/#compress). When imaging conditions such as lighting or magnification are changed, these parameters will need to be updated for good performance.

The most commonly adjusted parameter is the `Threshold`. If you have dark worms on a light background, `Is Light Background?` should be checked.  In this case, pixels that are darker than the threshold value will be included in the mask. The selected value should be low enough to exclude as much background as possible without losing any part of the animals to be tracked. Below there is an example on how to do this.

If the objects to track are lighter than the background (e.g. if you are tracking fluorescent objects or using dark field illumination), un-check `Is Light Background?`.  In this case, pixels that are above the threshold value will be included in the mask.

![SetParameters](https://cloud.githubusercontent.com/assets/8364368/26410507/6df7ef54-409b-11e7-8139-9ce99daf69cb.gif)  

In some cases, even after adjusting the threshold there still remain large regions of background. If the tracked objects significatively change position during the movie you can enable the background subtraction as shown below. This method will consider anything that does not change within the specified frame range as background.  However, if any of your animals are immobile during the entire frame range will be lost. 

![SetBgndSubt](https://cloud.githubusercontent.com/assets/8364368/26410958/95a8c09a-409c-11e7-9fc9-14dafeabb467.gif)  

Other important parameters to set are:

* `Frame per Seconds` (fps) is the frame rate of your video. An important value since it is used to calculate several other parameters. If `Extract Timestamp` is set to `true`, the software will try to extract the frame rate from the video timestamp. However, keep in mind that it is not always possible to recover the correct timestamp, and therefore it is recommended that you provide the value here.
* `Frames to Average` is used to calculate the background mask. This value can significantly speed up the compression step. However, it will not work if the particles are highly motile. Use the buttons `Play` and `Next Chunk` to see how a selected value affects the mask. Note that the average is used only for the background mask, the foreground regions are kept intact for each individual frame. 
* `Microns per Pixels`. This value is only used to calculate the [skeleton features](EXPLANATION.md/#feat_create), but the results will be in pixels instead of micrometers if the conversion factor is not set.

You can access further parameters by clicking `Edit More Parameters`. The explanation of each parameter can be found by using the [contextual help](https://en.wikipedia.org/wiki/Tooltip). It is not always trivial to effectively adjust these other parameters, but if you believe you need too, I recommend using a small video (~100 frames) for testing.

When you are satisfied with the selected parameters select a file name and press `Save Parameters`. The parameters will be saved as a [JSON](http://json.org/) file that can be used in [Batch Processing Multiple Files](#batch-processing-multiple-files). If you need to further modify a parameter you can either use a text editor to change the JSON file directly or reload the file by dragging it into the Set Parameters widget.

## Batch Processing Multiple Files
![BatchProcessing](https://cloud.githubusercontent.com/assets/8364368/26605347/86ffb1e6-4585-11e7-9835-ffdc0751c67a.png)

This widget is used to execute [all steps](EXPLANATION.md) of tracking and feature extraction on each of the files on a given directory. The program allows a degree of parallelization by analyzing multiple files at the same time.  The number of processes to run in parallel (`Maximum Number of Processes`) should not exceed the number of processor cores available on your machine to avoid slowing down the analysis.


### Chosing the Files to be Analyzed
Tierpsy Tracker will analyse of the video files that are present in `Original Video Dir` including sub-directories.  Particular files are included if their names match the `File Pattern to Include`, but do not match the `File Pattern to Exclude`. 

* The patterns can use [Unix shell-style wildcards](https://docs.python.org/3.1/library/fnmatch.html). 
* In order to distinguish the [output files](OUTPUTS.md) that are saved during processing, any file that ends with any of the [reserved suffixes](https://github.com/ver228/tierpsy-tracker/blob/master/tierpsy/helper/misc/file_processing.py#L5) will be ignored. 
* To analyze a single file set `File Pattern to Include` to the entire file name.
* If the `Analysis Start Point` is set to a step after [`COMPRESS`](EXPLANATION.md/#compress) the `Original Videos Dir` is ignored and the `Masked Videos Dir` is used instead.

Alternatively, one can create a text file with the list of files to be analysed (one file per line). The path to this file can then be set in `Individual File List`. 

### Parameters Files
Parameters files created using the [Set Parameters](#set-parameters) widget can be select in the `Parameter Files` box. You can also select some previously created files using the drop-down list. If no file is selected the [default values](https://github.com/ver228/tierpsy-tracker/blob/dev/tierpsy/helper/params/docs_tracker_param.py) will be used. 

#### Worm Tracker 2.0 Option

You can analyse videos created by the [Worm Tracker 2.0](http://www.mrc-lmb.cam.ac.uk/wormtracker/) by selecting the parameters files [WT2_clockwise.json](https://github.com/ver228/tierpsy-tracker/blob/development/tierpsy/extras/param_files/WT2_clockwise.json) or
[WT2_anticlockwise.json](https://github.com/ver228/tierpsy-tracker/blob/development/tierpsy/extras/param_files/WT2_anticlockwise.json). Use the former if the ventral side in the videos is located in the clockwise direction from the worm head, and the later if it is in the anticlockwise direction. To select a subset of files with a particular orientation you can save each subset in a different root directory or include the orientation information in the file name and use use the `Pattern include` option . If you need to fine-tune the parameters you can edit the .json files either with a text editor or with `Set Parameters`.

Note that each of the video files `.avi` must have an additional pair of files with the extensions `.info.xml` and `.log.csv`. Additionally, keep in mind that if the stage aligment step fails, an error will be risen and the analysis of that video will be stopped. If you do not want to see the error messages untick the option `Print debug information`.


### Analysis Progress 

Tierpsy Tracker will determine which analysis steps have already been completed for the selected files and will only execute the analysis from the last completed step. Files that were completed or do not satisfy the next step requirements will be ignored. 

* To see only a summary of the files to be analysed without starting the analysis tick `Only Display Progress Summary`.

* You can start or end the analysis at specific points by using the `Analysis Start Point` and `Analysis End Point` drop-down menus. 

* If you want to re-analyse a file from an earlier step, delete or rename the output files that were created during the previous run. If you only want to overwrite a particular step, you have to delete the corresponding step in the `/provenance_tracking` node in the corresponding file.

### Directory to Save the Output Files
The masked videos created in the [compression step](EXPLANATION.md/#video-compression) are stored in `Masked Videos Dir`. The rest of the tracker results are stored in `Tracking Results Dir`. In both cases the subdirectory tree structure in `Original Videos Dir` is recreated. 

The reason for creating the parallel directory structure is to make it easy to delete the analysis outputs to re-run with different parameter values.  It also makes it easy to delete the original videos to save space once you've arrived at good parameter values. If you prefer to have the output files in the same directory as the original videos you can set `Masked Videos Dir` and `Tracking Results Dir` to the same value.

### Temporary directory
By default, Tierpsy Tracker creates files in the `Temporary Dir` and only moves them to the `Masked Videos Dir` or the `Tracking Results Dir` when the analysis has finished. The reasons to use a temporary directory are:

* Protect files from corruption due to an unexpected termination (crashes). HDF5 is particularly prone to get corrupted if a file was opened in write mode and not closed properly.
* Deal with unreliable connections. If you are using remote disks it is possible that the connection between the analysis computer and the data would be interrupted. A solution is to copy the required files locally before starting the analysis and copy the modified files back once is finished.

Some extra options:

* By default the original videos are not copied to the temporary directory for compression. These files can be quite large and since they would be read-only they do not require protection from corruption.  If you want copy the videos, tick `Copy Raw Videos to Temp Dir` box.

* In some cases the analysis will not finished correctly because some steps were not executed. If you still want to copy the files produced by the remaining steps to the final location tick the `Copy Unifnished Analysis` box.

### Command Line Tool

The same functions are accesible using the command line. You can see the available option by typing in the main tierpsy directory:
```
python cmd_scripts/processMultipleFiles.py -h
```

## Tierpsy Tracker Viewer

This widget is used to visualize the tracking results. You can move to a specific frame, zoom in/out, select specific trajectories, and visualize the skeletons overlayed on the compressed video, the trajectory paths or saved [unmasked frames](OUTPUTS.md#full_data). See below for an example on how to use it.

![MWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412511/eac27158-40a0-11e7-880c-5671c2c27099.gif)  

### Manually Joining Trajectories

You can manually correct the trajectories as shown below. Once you have finished open [Batch Processing Multiple Files](#batch-processing-multiple-files) and re-run the analysis selecting FEAT\_MANUAL\_CREATE from the Analysis Start Point drop menu. This will execute the step [FEAT\_MANUAL\_CREATE](EXPLANATION.md/#feat_manual_create), and create a file with the extension basename_feat_manual.hdf5 with the same contents as [`basename_features.hdf5`](OUTPUTS.md/#basename_features.hdf5) but with the manually joined indexes.

![TrackJoined](https://cloud.githubusercontent.com/assets/8364368/26412212/e0e112f8-409f-11e7-867b-512cf044d717.gif) 

### Viewer Shortcuts

`W` : label selected box as `Single Worm`.

`C` : label selected box as `Worm Cluster`.

`B` : label selected box as `Bad`.

`U` : label selected box as `Undefined`.

`J` : Join both trajectories in the zoomed windows.
	
`S` : Split the selected trajectory at the current time frame.

`Up key` : select the top zoomed window. 
	
`Down key` : select the bottom zoomed window. 

`[` : Move the the begining of the selected trajectory.
	
`]` : Move the the end of the selected trajectory.

`+` : Zoom out the main window.

`-` : Zoom in the main window.

`>` : Duplicated the frame step size.

`<` : Half the frame step size.

`Left key` : Increse the frame by step size.

`Right key` : Decrease the frame by step size.
    
### Plotting the Analysis Results
The analysis will produce a set of files described [here](https://github.com/ver228/tierpsy-tracker/blob/development/docs/OUTPUTS.md). The extracted features are store in the files that end with [features.hdf5](https://github.com/ver228/tierpsy-tracker/blob/development/docs/OUTPUTS.md#basename_featureshdf5). You can access to them using [pandas](http://pandas.pydata.org/) and [pytables](http://www.pytables.org/). There are examples on how to do it in MATLAB in the [tierpsy_tools](https://github.com/aexbrown/tierpsy_tools) repository.

## Features Summary
![FeatSummary](https://user-images.githubusercontent.com/8364368/41034550-d3665230-6981-11e8-97d9-63c74ff24661.png)
* `Root Directory`: Directory containing the previously calculate features files. 
* `Feature Type` : Select between the features calculated using the [OpenWorm Analysis Toolbox](https://github.com/openworm/open-worm-analysis-toolbox) or [Tierpsy Features](https://github.com/ver228/tierpsy-features).
* `Use manually edited features?` Tick if you want to collect data from manually edited trajectories. Only trajectories labelled as either worm or worm cluster are going to be used.
* `Summary Type` : Select what data is going to be collected from a video. Either a summary per video (`plate`), a summary for each trajectory available (`trajectory`) or multiple random subsamplings per video (`plate_augmented`).

The files will be located by doing a recursive search for matching the extension according to the table below.

| Feature Type | Is Manually Edited? | File Extension |
| -------- | -------- | ------ | 
| tierpsy | Ticked | featuresN.hdf5 |
| tierpsy | Unticked | featuresN.hdf5 |
| openworm | Ticked | feat_manual.hdf5 |
| openworm | Unticked | features.hdf5 |

The results are saved into two separated .csv file located in the root directory. The first file, `filenames_FEATURETYPE_SUMMARY_DATE.csv`, contains the names of all the files found in root the directory. The `is_good` column is set to `True` if the file is valid and used in the summary. The second file, `features_FEATURETYPE_SUMMARY_DATE.csv`, contains the corresponding features summarized as described in the [output files](https://github.com/ver228/tierpsy-tracker/blob/master/docs/OUTPUTS.md#features_summar) section. The two result files can be joined using the `file_id` column.
