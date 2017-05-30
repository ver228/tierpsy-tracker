# Software Explanation

## Video Compression

### COMPRESS

This step has the double function of identifing candidate regions for the tracking and zeroing the background in order to efficiently store data using lossless compression. 

The algorithm identifies dark particles on a lighter background or light particles on a darker background using [adaptative thresholding](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html) and filtering particles by size. The filter parameters should be adjusted manually for each different setup, however it should be possible to use the same parameters under similar experimental contions. More information on how to setup these parameters can be found in the [GUI manual](HOWTO.md/#set-parameters).

![COMPRESS](https://cloud.githubusercontent.com/assets/8364368/8456443/5f36a380-2003-11e5-822c-ea58857c2e52.png)

The masked images are stored into a HDF5 container using a gzip filter. Some advantages of this format are:

* It can significally reduce the size of a high resolution video since only the pixels corresponding to tracked particle are kept. This gain depends on codec used by the original video and the fraction of pixels in the image that correspond to the background. In some cases, *e.g* where the animal occupies a large region of the image or the original video was highly compressed, the gain in compression using the gzip hdf5 format can be low or even inexistent. 

* Rapid access to specific video frames. Typically it is slow access to a specific video frame, particularly in long videos. Most of the video readers do an approximative search for fast moving to a specific time point, and can miss the desired frame by a significative amount. To accurately retrieve a specific frame one has to read sequencially the whole video file, a slower process particuarly if the frame is at the end of the video. The HDF5 format indexes the data so it takes a similar amount of time to access to any frame in the video.

* Metadata can be stored in the same file as the video. HDF5 format allows to store all kind of binary data into the same file. This allows to store the video metadata, timestamp and experimental condtions, as well as the analysis progress in the same file.

### VID_SUBSAMPLE
After compression a low resolution avi file is generated. This is only for visualization purposes in case the [Tierpsy Tracker Viewer](HOWTO.md/#tierpsy-tracker-viewer) is not available.

## Create Trajectories

### TRAJ_CREATE

The first step is to identify possible particles. We divide the image in regions of non-zero connected pixels. For each candidate region we calculate a simple threshold and create a binary mask. We identify each individual object and calculate their centroids, areas and bounding boxes. Only objects with features within user-defined ranges are kept. This information is stored in the [/plate_worms](OUTPUTS.md/#plate_worms) table.

### TRAJ_JOIN

The second step is to join the identified particles to create the trajectories. We link the particles trajectories by using their closest neighbor in consecutive frames. The closest neighbor must less than `max_allowed_dist` and the change in area must be less than `area_ratio_lim` . One particle can only be joined to a single particle in consecutive frames, no splitted or merged trajectories are allowed. If this conditions are not satisfied it means that there was a problem in the trajectory *e.g.* two worms colided and therefore in the next frame the closest object is twice the area, or a worm disapear from the screen. If there is any conflict the trajectory will be broken and a new number will be assigned to any unassigned particle. 

In a subsequent step the program will try to join trajectories that have a small time gap between them *i.e.* the worm was lost for a couple of frames. Additionally we will remove any spurious trajectories shorter than `min_track_size` .

Below there is an example of how the trajectories look after tracking.

![trajectories](https://cloud.githubusercontent.com/assets/8364368/26301795/25eb72ac-3eda-11e7-8a52-99dd6c49bc07.gif)

### SKE_INIT
This is a refiment step to clean [/plate_worms](OUTPUTS.md/#plate_worms). For each trajectory we interpolate any time gap, calculate a fixed ROI size, and smooth the threshold and centroid over time. The purpose of this modifications is to make the thresholding more robust and the data suitable to for the [next step](#calculate-skeletons). The data is stored in the [/trajectories_data](OUTPUTS.md/#trajectories_data) table that becomes the central table. It contains the trajectories that would be used in the subsequent steps and displayed by the [viewer](HOWTO.md#tierpsy-tracker-viewer).


### BLOB_FEATS

We extract a set of features for each particle in each frame (corresponding to the individual rows in [/trajectories_data](#trajectories_data)). The results are stored in the [/blob_features](#blob_features) table.

## Calculate Skeletons

### SKE CREATE

In this step the multiworm data is transform to a single worm mask by extracting individual ROIs using the information in [/trajectories_data](OUTPUTS.md/#trajectories_data). Then the skeletons can be extracted using using the [segWorm](https://github.com/openworm/SegWorm) algorithm. To increase the speed the algorithm was implemented in Python and optimized using Cython and C. The output is stored as [`basename_skeletons.hdf5`](OUTPUTS.md/#basename_skeletonshdf5). An example of the result is shown below. 
![skeletons](https://cloud.githubusercontent.com/assets/8364368/26309647/a6b4402e-3ef5-11e7-96cd-4a037ee42868.gif)

### SKE_FILT
This step indentify skeletons that are likely to correspond to a wrong mask. There are two different filtering steps:

1. It is based on [segWorm](https://github.com/openworm/SegWorm) and looks for large changes in width or area between the midbody and the head or the tail. This changes are likely to correspond to coiled worms or cases were the worm is incontact with some debris. 
2. It looks for width or length outliers among all the skeletons in the video. The idea is that it is relatively common to find worms that are barely touching or parallel of each other. In these cases the corresponding binary mask will look like a valid worm except for having a disproportionate large length or width. This step might fail if the width or legth distributions are shifted due to tracked spurious particles.


### SKE_ORIENT
This step orient the skeleton head and tail by movement. The [SKE_CREATE](#ske_create) step produces a skeleton but do not determine which extreme of the curve is the head and which the tail. Since the head in one skeleton will not suddenly jump to the other side of the worm within a small amout of time, it is possible to assign "blocks" of skeletons with the same orientation as long as there is not a gap of missing skeletons larger than a few frames. For each of these blocks we can calculate motility of each of the curve extremes as the standard deviation of their angular speed. The part corresponding to the head should have a larger motility than the one corresponding to the tail. This approach should be able to correct most of the skeletons but further improvements can be archieve using the [image intensity](#int_ske_orient).

### STAGE_ALIGMENT
Only used in `SINGLE_WORM_SHAFER`. In the [Schafer's lab worm tracker](http://www.mrc-lmb.cam.ac.uk/wormtracker/) a single worm is follow by a camera around the plate. This requires the extra step to shift skeletons coordinates from the image frame of reference to the stage frame of reference. In this step the recorded stage positions saved as a `.log.csv` file are aligned to the the video time. This step requires MATLAB to run the corresponding function from the original [segWorm](https://github.com/openworm/SegWorm) code. Please raise an issue if you are interested in the development of a python only implementation.


### CONTOUR_ORIENT
Only used in `SINGLE_WORM_SHAFER`. This step switches the dorsal and ventral contours to match the `ventral_side` orientation specified in the `/experiment_info` field or in the JSON parameters file.

### INT_PROFILE
Straighten the worm by using the previously extracted skeleton (see below A) and interpolating it into a 2D map (see below B top). The intensity map can be further smoothed by averaging data along the worm to obtain an intensity profile as shown below in B bottom.  

![INT_SKE_ORIENT](https://cloud.githubusercontent.com/assets/8364368/26366191/089a6ca4-3fe2-11e7-91ef-77a7a78ee8ba.png)

### INT_SKE_ORIENT
This step uses the profile intensity extracted [before](#int_profile) to increase the accuracy of [head-tail orientation](#ske_orient). The intensity profile have a distinct pattern due to anatomical differences in the worm. Regions that are wrongly oriented can easily be observed when the intensity profiles are arranged by time frame (see above C bottom red dashed lines). These regions can be algorithmically detected by calculating a median intensity profile from all the frames, and obtaining the difference between the profile in each frame and the median profile or the median profile switched. In blocks of skeletons with the wrong orientation the difference with the switched profile will be less than with the original profile.

Using this algorithm the errors in head-tail identification decrease to 0.01% compared to 4.48% in the original [segWorm](https://github.com/openworm/SegWorm) implementation. Since this algorithm uses the median profile to identified switched regions, it can fail if [SKE_ORIENT](#ske_orient) previously did not correctly assigned most of the skeletons.

## Extract Features

### FEAT_CREATE
This step uses the [Open Worm Analysis Toolbox](https://github.com/openworm/open-worm-analysis-toolbox) to calculate the skeleton features explained in [`basename_features.hdf5`](OUTPUTS.md/#basename_features.hdf5).

### FEAT_MANUAL_CREATE
Same as [FEAT_CREATE](#feat_create) but it will only use the indexes that were manually identified as worms using the [Tierpsy Tracker Viewer](HOWTO.md#tierpsy-tracker-viewer). The results will be saved as `basename_feat_manual.hdf5`.

### WCON_EXPORT
Only used in `SINGLE_WORM_SHAFER`. Export skeletons data in [`basename_features.hdf5`](OUTPUTS.md/#basename_features.hdf5) using the [WCON format](https://github.com/openworm/tracker-commons). In the future this step should be available in the default analysis sequence.

