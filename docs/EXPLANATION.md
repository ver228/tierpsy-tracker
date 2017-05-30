# Software Explanation

## Video Compression

### COMPRESS

This step has the double function of identifing candidate regions for the tracking and zeroing the background in order to efficiently store data using lossless compression. 

The algorithm identifies dark particles on a lighter background or light particles on a darker background using [adaptative thresholding](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html) and filtering particles by size. The filter parameters should be adjusted manually for each different setup, however it should be possible to use the same parameters under similar experimental contions. More information on how to setup these parameters can be found in the [GUI manual](HOWTO.md#set-parameters).

![COMPRESS](https://cloud.githubusercontent.com/assets/8364368/8456443/5f36a380-2003-11e5-822c-ea58857c2e52.png)

The masked images are stored into a HDF5 container using a gzip filter. Some advantages of this format are:

* It can significally reduce the size of a high resolution video since only the pixels corresponding to tracked particle are kept. This gain depends on codec used by the original video and the fraction of pixels in the image that correspond to the background. In some cases, *e.g* where the animal occupies a large region of the image or the original video was highly compressed, the gain in compression using the gzip hdf5 format can be low or even inexistent. 

* Rapid access to specific video frames. Typically it is slow access to a specific video frame, particularly in long videos. Most of the video readers do an approximative search for fast moving to a specific time point, and can miss the desired frame by a significative amount. To accurately retrieve a specific frame one has to read sequencially the whole video file, a slower process particuarly if the frame is at the end of the video. The HDF5 format indexes the data so it takes a similar amount of time to access to any frame in the video.

* Metadata can be stored in the same file as the video. HDF5 format allows to store all kind of binary data into the same file. This allows to store the video metadata, timestamp and experimental condtions, as well as the analysis progress in the same file.

### VID_SUBSAMPLE
After compression a low resolution avi file is generated. This is only for visualization purposes in case the [Tierpsy Tracker Viewer](HOWTO.md#tierpsy-tracker-viewer) is not available.

## Creating trajectories

### TRAJ_CREATE

The first step is to identify possible particles. We divide the image in regions of non-zero connected pixels. For each candidate region we calculate a simple threshold and create a binary mask. We identify each individual object and calculate their centroids, areas and bounding boxes. Only objects with features within user-defined ranges are kept. This information is stored in the [/plate_worms](#plate_worms) table.

### TRAJ_JOIN

The second step is to join the identified particles to create the trajectories. We link the particles trajectories by using their closest neighbor in consecutive frames. The closest neighbor must less than `max_allowed_dist` and the change in area must be less than `area_ratio_lim` . One particle can only be joined to a single particle in consecutive frames, no splitted or merged trajectories are allowed. If this conditions are not satisfied it means that there was a problem in the trajectory *e.g.* two worms colided and therefore in the next frame the closest object is twice the area, or a worm disapear from the screen. If there is any conflict the trajectory will be broken and a new number will be assigned to any unassigned particle. 

In a subsequent step the program will try to join trajectories that have a small time gap between them *i.e.* the worm was lost for a couple of frames. Additionally we will remove any spurious trajectories shorter than `min_track_size` .

Below there is an example of how the trajectories look after tracking.

![trajectories](https://cloud.githubusercontent.com/assets/8364368/26301795/25eb72ac-3eda-11e7-8a52-99dd6c49bc07.gif)

### SKE_INIT
This is a refiment step to clean [/plate_worms](#plate_worms). For each trajectory, we interpolate any time gap, calculate a fixed ROI size, and smoothes the threshold and centroid over time. The purpose of this modifications is to make the thresholding more robust and the data suitable to for the (next step)[Extracting worm skeletons]. The data is stored in the [/trajectories_data](#trajectories_data) table that becomes the central table. It contains the trajectories that would be used in the subsequent steps and displayed by the [viewer](HOWTO.md#tierpsy-tracker-viewer).


### BLOB_FEATS

We extract a set of features for each particle in each frame (corresponding to the individual rows in [/trajectories_data](#trajectories_data)). The results are stored in the [/blob_features](#blob_features) table.

## Extracting worm skeletons

### SKE CREATE
![skeletons](https://cloud.githubusercontent.com/assets/8364368/26309647/a6b4402e-3ef5-11e7-96cd-4a037ee42868.gif)

This step extract the worm skeleton using a python implementation of [segWorm](https://github.com/openworm/SegWorm). 


Since one has to deal with multiworm at a time speed becomes an important issue, therefore the code was optimized using Cython and C. 

The skeletons and contours are normalized to have the same number of points in order to store them in a simple table. The output is store in a file with the extension [basename_skeletons.hdf5](#basename_skeletonshdf5).

### SKE_FILT
Filter "bad worms", meaning any particle indentified and analyzed for the tracker that it is not a worm, or any trajectory that corresponds to two or more worms in contact.


### SKE_ORIENT
In a second part of the code the head and tail are identified by movement. Althought it is hard to determine the head and the tail from the contour, it is possible to assign "blocks" with the same orientation for skeletons in contingous frames, since the head in one skeleton will not suddenly jump to the other side of the worm within a few frames. We can then assign the relative standard deviation (SD) of the angular movement for the first and last part of the segment. If the blocks are large enough the section with the higher SD would be the head.
 
### STAGE_ALIGMENT
### CONTOUR_ORIENT
'ventral_side':['','clockwise','anticlockwise', 'unknown'],

### INT_PROFILE
### INT_SKE_ORIENT
![INT_SKE_ORIENT](https://cloud.githubusercontent.com/assets/8364368/26366191/089a6ca4-3fe2-11e7-91ef-77a7a78ee8ba.png)


## Extracting worm features

### FEAT_CREATE
### FEAT_MANUAL_CREATE

[Open Worm Analysis Toolbox](https://github.com/openworm/open-worm-analysis-toolbox)
Uses the code in `obtainFeatures.py` in the `FeaturesAnalysis` directory, and the movement validation repository. This part is still in progress but basically creates a normalized worm object from the basename_skeletons.hdf5 tables, and extract features and mean features using the movement_validation functions. The motion data is stored in a large table with all the worms in it and with with the indexes frame_number and worm_index, where the event data is stored in individual tables for each worm. The seven hundred or so mean features are stored in another table where each worm corresponds to worm index.

### WCON_EXPORT
[Tracker Commons](https://github.com/openworm/tracker-commons)

# Output Files

## basename.hdf5
attributes: 
  * expected_fps := 1,
  * time_units := 'frames'
  * microns_per_pixel := 1
  * xy_units := 'pixels'
  * is_light_background := 1

#### /mask 
(tot_images, im_high, im_width)
Compressed array with the masked image.

#### /full_data
(tot_images/save_full_interval, im_high, im_width)
Frame without mask saved every `save_full_interval`. The saving interval is recommended to be adjusted every 5min. This field can be useful to identify changes in the background that are lost in the [/mask](#mask) dataset *e.g.* food depletion or contrast lost due to water condensation.

#### mean_intensity
(tot_images)
Mean intensity of a given frame. It is useful in optogenetic experiments to identify when the light is turned on.

#### timestamp/time
#### timestamp/raw

Timestamp extracted from the video if the `is_extract_metadata` flag set to `true`. If this fields exists and are valid (there are not nan values and they increase monotonically), they will be used to calculate the `fps` used in subsequent parts of the analysis. The extracting the timestamp can be a slow process since it uses [ffprobe](https://ffmpeg.org/ffprobe.html) to read the whole video. If you believe that your video does not have a significative number of drop frames and you know the frame rate, or simply realise that ffprobe cannot extract the timestamp correctly, I recommend to set `is_extract_metadata` to `false`.

### basename_subsample.avi


### basename_skeletons.hdf5

#### /plate_worms
  * worm_index_blob: Trajectory index given initially by the program. Since there can be several short spurious tracks identified this number can be very large and does not reflect the number of final trajectories.
  * worm_index_joined: Index after joining trajectories separated by a small time gap and filtering short spurious tracks, and invalid row will be assigned -1.
  * threshold: Threshold used for the image binarization.
  * frame_number: Video frame number.
  * coord_x, coord_y, box_length, box_width, angle: center coordinates, length, width and orientation of the [minimum rotated rectangle](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#minarearect).
  * area: blob area.
  * bounding_box_xmin, bounding_box_xmax, bounding_box_ymin, bounding_box_ymax: [bounding rectangle](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#boundingrect) coordinates.

#### /trajectories_data
table containing the smoothed data and the indexes to link each row in the others table, with the corresponding worm_index and frame_number

  * frame_number: F
  * worm_index_joined: F
  * plate_worm_id: F
  * skeleton_id: row in the trajectory_data, useful to quickly recover worm data.
  * coord_x, coord_y: Centroid coordinates after smoothing [plate_worms](#plate_worms). It is used to find the ROI to calculate the skeletons. If you want to calculate the centroid features use the corresponding field in [blob_features](#blob_features).
  * threshold: value used to segment the worm in the ROI.
  * has_skeleton: flag to mark is the skeletonization was succesful
  * roi_size: F
  * area: F
  * timestamp_raw: F
  * timestamp_time: F
  * is_good_skel: F
  * skel_outliers_flag: F
  * int_map_id: F

#### /blob_features
  * coord_x, coord_y, box_length, box_width, box_orientation
  * area: [area](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#contourarea)
  * perimeter: [perimeter](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#arclength)
  * quirkiness: sqrt(1 - box_width^2 / box_width^2)
  * compactness: 4 * pi * area / (perimeter^2)
  * solidity: area / ([convex hull](http://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/shapedescriptors/hull/hull.html#) area)
  * intensity_mean, intensity_std: mean and standard deviation inside the thresholded region.
  * hu0, hu1, hu2, hu3, hu4, hu5, hu6: [hu moments](http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#humoments)

#### /contour_area:

#### /contour_side1_length: 
#### /contour_side2_length:
#### /skeleton_length: 
length in pixels.

#### /skeleton:
#### /contour_side1:
#### /contour_side2: 
  normalized coordinates. head is the first index and tail the last. The contour side is assigned to keep a clockwise-orientation. There is still work to do to find what is the ventral and dorsal side.

#### /width_midbody:

#### /contour_width:
  contour width along the skeleton. I'm using the output from segworm, and resampling by interpolation It might be possible to improve this.

#### /intensity_analysis/switched_head_tail:
  * worm_index
  * ini_frame
  * last_frame

#### /timestamp/raw:
#### /timestamp/time:

### basename_features.hdf5

#### /coordinates/dorsal_contours:
#### /coordinates/ventral_contours:
#### /coordinates/skeletons:

#### /features_events/worm_*:
  * inter_backward_distance
  * inter_backward_time
  * inter_coil_distance
  * inter_coil_time
  * inter_forward_distance
  * inter_forward_time
  * inter_omega_distance
  * inter_omega_time
  * inter_paused_distance
  * inter_paused_time
  * inter_upsilon_distance
  * inter_upsilon_time
  * midbody_dwelling
  * omega_turn_time
  * omega_turns_frequency
  * omega_turns_time_ratio
  * paused_distance
  * paused_motion_distance_ratio
  * paused_motion_frequency
  * paused_motion_time_ratio
  * paused_time
  * tail_dwelling
  * upsilon_turn_time
  * upsilon_turns_frequency
  * upsilon_turns_time_ratio
  * worm_dwelling

#### /features_timeseries:
  * worm_index
  * timestamp
  * skeleton_id
  * motion_modes
  * length
  * head_width, midbody_width, tail_width
  * area
  * area_length_ratio
  * width_length_ratio
  * max_amplitude
  * amplitude_ratio
  * primary_wavelength, secondary_wavelength
  * track_length
  * eccentricity
  * bend_count
  * tail_to_head_orientation
  * head_orientation
  * tail_orientation
  * eigen_projection_1, eigen_projection_2, eigen_projection_3,  eigen_projection_4, eigen_projection_5, eigen_projection_6
  * head_bend_mean, neck_bend_mean, midbody_bend_mean, hips_bend_mean, tail_bend_mean
  * head_bend_sd, neck_bend_sd, midbody_bend_sd, hips_bend_sd, tail_bend_sd
  * head_tip_speed, head_speed, midbody_speed, tail_speed, tail_tip_speed
  * head_tip_motion_direction, head_motion_direction, midbody_motion_direction, tail_motion_direction, tail_tip_motion_direction
  * head_crawling_amplitude, midbody_crawling_amplitude, tail_crawling_amplitude
  * head_crawling_frequency, midbody_crawling_frequency, tail_crawling_frequency
  * foraging_amplitude
  * foraging_speed
  * path_range
  * path_curvature

#### /features_summary: 
  P10th_split, P90th_split

  * P10th
  * P90th
  * means
  * medians







