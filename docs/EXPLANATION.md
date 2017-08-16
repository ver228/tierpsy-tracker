# Software Explanation
This page explains all the steps executed in the analysis of a video file. See the [Output Files](OUTPUTS.md) section for a description of each of the files created by Tierpsy Tracker.


## Video Compression

### COMPRESS

This step has the double function of identifing candidate regions for tracking and zeroing the background in order to efficiently store data using lossless compression. 

The algorithm identifies dark particles on a lighter background or light particles on a darker background using [adaptative thresholding](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html) and filtering particles by size. The filter parameters should be adjusted manually for each different setup, however it should be possible to use the same parameters under similar experimental contions. More information on how to set these parameters can be found in the instructions for the widget [Set Parameters](HOWTO.md/#set-parameters).

![COMPRESS](https://cloud.githubusercontent.com/assets/8364368/8456443/5f36a380-2003-11e5-822c-ea58857c2e52.png)

The masked images are stored into a HDF5 container using a gzip filter. Some advantages of this format are:

* It can significally reduce the size of a high resolution video since only the pixels corresponding to tracked particle are kept. This gain depends on codec used by the original video and the fraction of pixels in the image that correspond to the background. In some cases, *e.g* where the animal occupies a large region of the image or the original video was highly compressed, the gain in compression using the gzip hdf5 format can be low or even negative. 

* Rapid access to specific video frames. Typically it is slow to access to a specific video frame, particularly in long videos. Most  video readers do an approximative search to a specific time point for fast seeking and can miss the desired frame. To accurately retrieve a specific frame one has to read sequencially the whole video file, a slower process particuarly if the frame is at the end of the video. The HDF5 format indexes the data so it takes a similar amount of time to access any frame in the video.

* Metadata can be stored in the same file as the video. HDF5 format allows to store all kinds of binary data in the same file, including video metadata, timestamp and experimental condtions, as well as the analysis progress.

### VID_SUBSAMPLE
After compression a low resolution avi file is generated. This is only for visualization purposes in case the [Tierpsy Tracker Viewer](HOWTO.md/#tierpsy-tracker-viewer) is not available.

## Create Trajectories

### TRAJ_CREATE

The first step is to identify possible particles. We divide the image into regions of non-zero connected pixels. For each candidate region we calculate a simple threshold and create a binary mask. We identify each individual object and calculate their centroids, areas and bounding boxes. Only objects with features within user-defined ranges are kept. This information is stored in the [/plate_worms](OUTPUTS.md/#plate_worms) table.

### TRAJ_JOIN

The second step is to join the identified particles into trajectories. We link the particles' trajectories by using their nearest neighbor in consecutive frames. The nearest neighbor must be less than `max_allowed_dist` away and the fractional change in area must be less than `area_ratio_lim`. One particle can only be joined to a single particle in consecutive frames, no split or merged trajectories are allowed. If these conditions are not satisfied it means that there was a problem in the trajectory *e.g.* two worms collided and therefore in the next frame the closest object is twice the area, or a worm disapeared from the field of view. If there is any conflict, the trajectory will be broken and a new label will be assigned to any unassigned particle. 

In a subsequent step, Tierpsy Tracker tries to join trajectories that have a small time gap between them *i.e.* the worm was lost for a few frames. Additionally we will remove any spurious trajectories shorter than `min_track_size` .

Additionally, if a valid [keras model](https://keras.io/) was trained. It is possible to use a neural network to filter worms from eggs and other spurious particles.

Below there is an example of how the trajectories look after tracking.

![trajectories](https://cloud.githubusercontent.com/assets/8364368/26301795/25eb72ac-3eda-11e7-8a52-99dd6c49bc07.gif)

### SKE_INIT
This is a refinement step to clean [/plate_worms](OUTPUTS.md/#plate_worms). For each trajectory we interpolate any time gap, calculate a fixed region of interest size, and smooth the threshold and centroid over time. The purpose of these modifications is to make the thresholding more robust and the data suitable for the [next step](#calculate-skeletons). The data is stored in the [trajectories_data](OUTPUTS.md/#trajectories_data) table. This is the main table since it contains the data of the trajectories that are used in the subsequent steps and displayed by the [viewer](HOWTO.md/#tierpsy-tracker-viewer).


### BLOB_FEATS

We extract a set of features for each particle in each frame (corresponding to the individual rows in [/trajectories_data](OUTPUTS.md/#trajectories_data)). The results are stored and explained in the [/blob_features](OUTPUTS.md/#blob_features) table.

## Calculate Skeletons

### SKE_CREATE

In this step the multiworm data is transformed to a single worm mask by extracting individual regions of interest using the information in [/trajectories_data](OUTPUTS.md/#trajectories_data). Then the skeletons can be extracted using the [segWorm](https://github.com/openworm/SegWorm) algorithm. The basis of this algorithm is to identify the contour points with the highest curvature. These points correspond to the head and the tail. Using these points the contour is then divided in two parts corresponding to the ventral and dorsal side. The skeleton is calculated as the set of middle points between oposite sides of the contour. To increase speed, the algorithm was re-implemented in python and optimized using cython and C. The output is stored as [`basename_skeletons.hdf5`](OUTPUTS.md/#basename_skeletonshdf5). An example of the result is shown below. 

![skeletons](https://cloud.githubusercontent.com/assets/8364368/26309647/a6b4402e-3ef5-11e7-96cd-4a037ee42868.gif)

### SKE_FILT
This step identifies skeletons that are likely to have been produced by an inaccurate mask. There are two different filtering steps:

1. The first step is based on [segWorm](https://github.com/openworm/SegWorm) and looks for large changes in width (set by `filt_bad_seg_thresh`) or area (set by `filt_max_area_ratio`) between the midbody and the head or the tail. These changes are likely to correspond to coiled worms or cases were the worm is in contact with some debris. 

2. The second step looks for width or length outliers among all the skeletons in the video. The idea is that it is relatively common to find worms that are barely touching or parallel to each other. In these cases, the corresponding binary mask will look like a valid worm except for having a disproportionately large length or width. This step can fail unless most of the tracked particles are properly segmented single worms.


### SKE_ORIENT
This step orients the skeleton head and tail by movement. The [SKE_CREATE](#ske_create) step produces a skeleton but does not determine which extreme of the curve is the head and which the tail. Since the head in one skeleton will not suddenly jump to the other side of the worm within a small amout of time, it is possible to assign "blocks" of skeletons with the same orientation as long as there is not a gap of missing skeletons larger than a few frames (set by `max_gap_allowed_block`). For each of these blocks we estimate the motility of each of the skeleton extremes as the standard deviation of their angular speed. The part corresponding to the head should have a larger motility than the one corresponding to the tail. This approach is able to correct most of the skeletons but further improvements can be archieve using the [image intensity](#int_ske_orient).

### STAGE_ALIGMENT
Only used in for data from [Worm Tracker 2.0](http://www.mrc-lmb.cam.ac.uk/wormtracker/) (`WT2`). In Worm Tracker 2.0, a single worm is follow by a camera around the plate. Analysis of these videos requires the extra step of shifting skeletons coordinates from the image frame of reference to the stage frame of reference. In this step the recorded stage positions saved as a `.log.csv` file are aligned to the the video time. This step requires MATLAB to run the corresponding function from the original [segWorm](https://github.com/openworm/SegWorm) code. Please raise an [issue](https://github.com/ver228/tierpsy-tracker/issues) if you are interested in the development of a python only implementation.


### CONTOUR_ORIENT
Only used in `WT2`. This step switches the dorsal and ventral contours to match the `ventral_side` orientation specified in the `/experiment_info` field or in the JSON parameters file.

### INT_PROFILE
This step uses worm intensity profiles to improve head-tail orientation detection.  Worms are first straightened using the previously extracted skeleton (see below A) and interpolating it into a 2D map (see below B top). The intensity map can be further smoothed by averaging data along the worm to obtain an intensity profile as shown below in B bottom.  

![INT_SKE_ORIENT](https://cloud.githubusercontent.com/assets/8364368/26366191/089a6ca4-3fe2-11e7-91ef-77a7a78ee8ba.png)

### INT\_SKE\_ORIENT
This step uses the profile intensity extracted in the [previous step](#int_profile) to increase the accuracy of [head-tail orientation](#ske_orient). The intensity profiles have a distinct pattern due to anatomical differences in the worm. Regions that are wrongly oriented can easily be observed when the intensity profiles are arranged by time frame (see above C bottom red dashed lines). These regions can be algorithmically detected by calculating the median intensity profile of all the frames, and obtaining the difference between the profile in each frame and the median profile or the median profile in the opposite orientation. In blocks of skeletons with the wrong orientation the difference with the switched profile will be less than with the original profile.

Using this algorithm the errors in head-tail identification decrease to 0.01% compared to 4.48% in the original [segWorm](https://github.com/openworm/SegWorm) implementation. Since this algorithm uses the median profile to identify switched regions, it can fail if [SKE_ORIENT](#ske_orient) previously did not correctly oriented most of the skeletons.

## Extract Features

### FEAT_CREATE
This step uses the [open worm analysis toolbox](https://github.com/openworm/open-worm-analysis-toolbox) to calculate the skeleton features explained in [`basename_features.hdf5`](OUTPUTS.md/#basename_features.hdf5).

### FEAT\_MANUAL\_CREATE
Same as [FEAT_CREATE](#feat_create) but it will only use the indexes that were manually identified as worms using the [Tierpsy Tracker Viewer](HOWTO.md#tierpsy-tracker-viewer). The results will be saved as `basename_feat_manual.hdf5`.

### WCON_EXPORT
Currently only used in `WT2`. Export skeletons data in [`basename_features.hdf5`](OUTPUTS.md/#basename_features.hdf5) using the [WCON format](https://github.com/openworm/tracker-commons). In the future this step should be available in the default analysis sequence.

