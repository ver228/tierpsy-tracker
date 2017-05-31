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

![TierpsyTrackerConsole](https://cloud.githubusercontent.com/assets/8364368/26624637/64275e1c-45e9-11e7-8bd6-69a386007d89.png)   

## Set Parameters

The purpose of this widget is to setup the parameters used in [Batch Processing Multiple Files](#batch-processing-multiple-files). The graphic interface is designed to help to select the parameters for [video compression](EXPLANATION.md/#compress). These parameters should be changed under different imaginig conditions.

The most likely parameter to be modified is `Threshold`. The selected value should be enough to exclude as much background as possible without lossing any part of the animals to be tracked. Below there is an example on how to do this.

![SetParameters](https://cloud.githubusercontent.com/assets/8364368/26410507/6df7ef54-409b-11e7-8139-9ce99daf69cb.gif)  

In some cases, even after adjusting the threshold there still remain large regions of background. If the tracked objects significatively change position during the movie you can enable the background subtraction as shown below. This method will consider background anything that do not change within the specified frame range, therefore if some of your animals are inmobile they will be lost. 

![SetBgndSubt](https://cloud.githubusercontent.com/assets/8364368/26410958/95a8c09a-409c-11e7-9fc9-14dafeabb467.gif)  

Other important parameters to set are:

* `Frame per Seconds` (fps) of your video. An important value since it is used to calculate several other parameters. If `Extract Timestamp` is set to `true`, the software will try to extract the fps from the video timestamp. However, keep it mind that it is not always possible to recover the correct timestamp, and therefore it is recommended give a value for the fps.
* `Frames to Average` to calculate the background mask. This value can significatively speed up the compression step. However, it will not work if the particles are highly motile. Use the buttons `Play` and `Next Chunck` to see how a selected value affects the mask. Note that the averaged is used only for the background mask, the foreground regions are kept intact for each individual frame. 
* `Microns per Pixels`. This value is only used to calculate the [skeleton features](EXPLANATION.md/#feat_create), but the results will be in pixels instead of micrometers if the conversion factor is not setted.

You can access to all the parameters by clicking `Edit More Parameters`. The explanation of each parameter can be found by using the [contextual help](https://en.wikipedia.org/wiki/Tooltip). It is not trivial to adjust the parameters, but if you believe you need too, I recommend to use a small movie (few seconds) for testing.

When you are satisfied with the selected parameters select a file name and press `Save Parameters`. The parameters will be saved into a [JSON](http://json.org/) that can be used by [Batch Processing Multiple Files](#batch-processing-multiple-files). If you need to further modify a parameter you can either use a text editor or reload the file by dragging it to the Set Parameters widget.

## Batch Processing Multiple Files

We can analyze multiple files simultaneously by setting `Maximum Number of Processes`. 

![BatchProcessing](https://cloud.githubusercontent.com/assets/8364368/26605347/86ffb1e6-4585-11e7-9835-ffdc0751c67a.png)

### How to choose the files to analyze
The program will do a recursive search in `Original Video Dir` looking for files that match the value in `File Pattern to Include`, but do not match the partern `File Pattern to Exclude`. 

* The patterns can use [Unix shell-style wildcards](https://docs.python.org/3.1/library/fnmatch.html). 
* In order to distinguish the program [output files](OUTPUTS.md), any file that ends with any of the [reserved subfixes](https://github.com/ver228/tierpsy-tracker/blob/master/tierpsy/helper/misc/file_processing.py#L5) will be ignored. 
* To analyze a single file set `File Pattern to Include` to the file name.
* If the `Analysis Start Point` is set to be after [`COMPRESS`](EXPLANATION.md/#compress) the `Original Videos Dir` would be ignored and `Masked Videos Dir` would be used instead.

Alternatively one can create a text file with the list of files to be analysed. The path to this file can be set in `Individual File List`. 


### What happens if I have analyzed files in the same directory. 

The program will find the progress of all the files selected for the analysis, and will only execute the analysis from the last completed step. Files that were completed or do not satisfy the next step requirements will be ignored. 

* To see only a summary of the files to be analysed without starting the analysis tick `Only Display Progress Summary`.

* You can start or end the analysis at specific points by using the `Analysis Start Point` and `Analysis End Point` drop-down menus. 
* If you want to re-analyse a file you might have to delete or rename the previous files. If you only want to overwrite a particular step, you have to delete the corresponding step in the `/provenance_tracking` node in the corresponding file. 



### Where to save the results
The masked videos created in the [compression step](EXPLANATION.md/#video-compression) are stored in `Masked Videos Dir`. The rest of the tracker results are stored in `Tracking Results Dir`. In both cases the subdirectory tree structure in `Original Videos Dir` would be recreated. 

The reason because the tracking and the compression files are stored in different directories is because the compressed files are mean to replace the original videos, and should not be altered after compression. On the other hand you might want to re-run the analysis using a different parameters. In this way you could delete or rename the results directory and start the analysis again. If you do not want to store the files in separate directories you can assign `Masked Videos Dir` and `Tracking Results Dir` to the same value.

### Parameters Files
Parameters files created with the widget [Set Parameters](#set-parameters) can be select in the `Parameter Files` box. You can also select some previously created files using the drop-down list. If no file is selected the [default values](https://github.com/ver228/tierpsy-tracker/blob/dev/tierpsy/helper/params/docs_tracker_param.py) will be used. 

### Temporary directory
By default the program creates files into the `Temporary Dir` and only moves them to the `Masked Videos Dir` or the `Tracking Results Dir` when the analysis finished. The reasons to use a temporary directory are:

* Protect files from corruption due to an unexpected termination (crashes). HDF5 is particularly prone to get corrupted if a file was opened in write mode and not closed properly.
* Deal with unreliable connections. If you are using remote disks it is possible that the connection between the analysis computer and the data would be interrupted. A solution is to copy the required files locally before starting the analysis and copy the modified files back once is finished.

Some extra options:

* By default the original videos are not copied to the temporary directory for compression. This files can be quite large and since they would be read-only they do not require protection from corruption.  If you want copy the videos, *i.e.* you have connection problems, tick `Copy Raw Videos to Temp Dir` box.

* In some cases the analysis will not finished correctly because some steps were not executed. If you still want to copy to the final destination the files produced by remaining steps tick the `Copy Unifnished Analysis` box.


### Command Line Tool

The same functions are accesible using the command line. You can see the available option by typing in the main tierpsy directory:
```
python cmd_scripts/processMultipleFiles.py -h
```

## Tierpsy Tracker Viewer


![MWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412511/eac27158-40a0-11e7-880c-5671c2c27099.gif)  

Tracks can be joined
![TrackJoined](https://cloud.githubusercontent.com/assets/8364368/26412212/e0e112f8-409f-11e7-867b-512cf044d717.gif) 

### HotKeys
	W : label selected box as `Single Worm`.
	C : label selected box as `Worm Cluster`.
	B : label selected box as `Bad`.
	U : label selected box as `Undefined`.
	
	J : Join both trajectories in the zoomed windows.
	S : Split the selected trajectory at the current time frame.
	
	Up key : select the top zoomed window. 
	Down key : select the bottom zoomed window. 
       
    [ : Move the the begining of the selected trajectory.
    ] : Move the the end of the selected trajectory.
    
    + : Zoom out the main window.
    - : Zoom in the main window.
    
    > : Duplicated the frame step size.
    < : Half the frame step size.
    
    Left key : Increse the frame by step size.
    Right key : Decrease the frame by step size.
    
## Single Worm Viewer
It is a similar interface to the (Tierpsy Tracker Viewer)[#tierpsy-tracker-viewer] created specifically for the `SINGLE_WORM_SHAFER` case. It can be used as shown below.

![SWTrackerViewer](https://cloud.githubusercontent.com/assets/8364368/26412826/e608bfea-40a1-11e7-9d3e-d0b8bf482db2.gif) 
