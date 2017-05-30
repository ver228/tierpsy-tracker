# Output Files
attributes: 
  * expected_fps := 1,
  * time_units := 'frames'
  * microns_per_pixel := 1
  * xy_units := 'pixels'
  * is_light_background := 1
  * 
## basename.hdf5
Contains the compressed hdf5 video data.

#### /mask 
`Shape (tot_images, im_high, im_width)`

Compressed array with the masked images.

#### /full_data
`Shape (tot_images/save_full_interval, im_high, im_width)`

Frame without mask saved every `save_full_interval` frames. By default the interval is adjusted to be saved every 5 min. This field can be useful to identify changes in the background that are lost in [/mask](#mask) *e.g.* food depletion or contrast lost due to water condensation.

#### mean_intensity
`Shape (tot_images,)`

Mean intensity of a given frame. It is useful in optogenetic experiments to identify when the light is turned on.

#### timestamp/time timestamp/raw

Timestamp extracted from the video if the `is_extract_metadata` flag set to `true`. If this fields exists and are valid (there are not `nan` values and they increase monotonically), they will be used to calculate the `fps` used in subsequent parts of the analysis. The extraction of the timestamp can be a slow process since it uses [ffprobe](https://ffmpeg.org/ffprobe.html) to read the whole video. If you believe that your video does not have a significative number of dropped frames and you know the frame rate, or simply realise that ffprobe cannot extract the timestamp correctly, I recommend to set `is_extract_metadata` to `false`.

### basename_subsample.avi
Low time and spatial resolution avi video generated using the data in [/mask](#mask).

### basename_skeletons.hdf5
Contains the results of the [tracking](#create-trajectories) and [skeletonization](#calculate-skeletons) steps.

#### /plate_worms
Table where the first results of [TRAJ_CREATE](EXPLANATION.md/#traj_create) and [TRAJ_JOIN](EXPLANATION.md/#traj_join). Do not use this table in further analysis, use instead [/trajectories_data](#trajectories_data).

  * `worm_index_blob`: trajectory index given by the program. Since there can be several short spurious tracks identified this number can be very large and does not reflect the number of final trajectories.
  * `worm_index_joined`: index after joining trajectories separated by a small time gap and filtering short spurious tracks, and invalid row will be assigned -1.
  * `threshold`: threshold used for the image binarization.
  * `frame_number`: video frame number.
  * `coord_x`, `coord_y`, `box_length`, `box_width`, `angle`: center coordinates, length, width and orientation of the [minimum rotated rectangle](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#minarearect).
  * `area`: blob area.
  * `bounding_box_xmin`, `bounding_box_xmax`, `bounding_box_ymin`, `bounding_box_ymax`: [bounding rectangle](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#boundingrect) coordinates.

#### /trajectories_data
Table containing the data of the trajectories used in the analysis and displayed by the [Tierpsy Tracker Viewer](HOWTO.md#tierpsy-tracker-viewer). Each row should have a unique pair of `worm_index_joined` and `frame_number` keys corresponding to each of the particles identified in each video frame.

  * `frame_number`: video frame number.
  * `worm_index_joined`: same as in [`/plate_worms`](#plate_worms).
  * `plate_worm_id`: row number in [`/plate_worms`](#plate_worms).
  * `skeleton_id`: row in this table. It is useful to recover data after slicing using pandas.
  * `coord_x`, `coord_y`: centroid coordinates after smoothing [/plate_worms](#plate_worms). It is used to find the ROI to calculate the skeletons. If you want to calculate the centroid features use the corresponding field in [/blob_features](#blob_features).
  * `threshold`: value used to binarize the ROI.
  * `has_skeleton`: `true` is the skeletonization was succesful.
  * `is_good_skel`: `true` if the skeleton passed the [filter step](#ske_filt). Only rows with this flag as `true` will be used to calculate the [skeleton features](#feat_create). 
  * skel_outliers_flag: internal used to identify why a skeleton was rejected in the [filter step](#ske_filt).
  * `roi_size`: size in pixels of the region of interest. Should be constant for a given trajectory.
  * `area`: expected blob area. Useful to filter spurious particles after the ROI binarization.
  * `timestamp_raw`: timestamp number. Useful to find droped frames.
  * `timestamp_time`: real time timestamp value.
  * `int_map_id`: corresponding row in the [`base_name_intensities.hdf5`](base_name_intensities.hdf5).

#### /blob_features
  * `coord_x`, `coord_y`, `box_length`, `box_width`, `box_orientation`. features calculated using [minAreaRect](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#minarearect).
  * `area`: [area](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#contourarea).
  * `perimeter`: [perimeter](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#arclength).
  * `quirkiness`: defined as `sqrt(1 - box_width^2 / box_width^2)`.
  * `compactness`: defined as `4 * pi * area / (perimeter^2)`.
  * `solidity`: `area / convex hull area` where the convex hull is calculated as [here](http://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/shapedescriptors/hull/hull.html#).
  * `intensity_mean`, `intensity_std`: mean and standard deviation inside the thresholded region.
  * `hu0`, `hu1`, `hu2`, `hu3`, `hu4`, `hu5`, `hu6`: [Hu moments](http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#humoments).


#### /skeleton /contour_side1 /contour_side2
Normalized coordinates (same number of points) for the skeletons and the contour in each side of the worm. The head should correspond to the first index and the tail to the last.


#### /contour_width
Contour width along the skeleton.

#### /width_midbody
Contour width of the midbody. Used to calculate the intensity maps in [INT_PROFILE](EXPLANATION.md/#int_profile).


#### /contour_side1_length /contour_side2_length /skeleton_length
Contours and skeleton length in pixels before normalization and smoothing. This value is likely to be larger than the length caculated in [FEAT_CREATE](EXPLANATION.md/#feat_create) due to the noiser contours and probably should be deprecated.

#### /contour_area
Area in pixels of the binary image used to calculate the skeletons. Probably should be deprecated.


#### /intensity_analysis/switched_head_tail
Internal. Table with the skeleton switched in [INT_SKE_ORIENT](EXPLANATION.md/#int_ske_orient).

#### /timestamp/raw /timestamp/time
Same as in [basename.hdf5](#basename.hdf5).

### base_name_intensities.hdf5

#### /trajectories_data_valid 
Same as [/trajectories_data](#trajectories_data) but only containing rows where `has_skeleton` is `true`.

#### /straighten_worm_intensity 
`Shape (tot_valid_skel, n_length, n_width)`

Intensity maps of the straigten worms described in [INT_PROFILE](EXPLANATION.md/#INT_PROFILE). Each index in the first dimension correspond to the same row in [/trajectories_data_valid](#trajectories_data_valid). Note that the data type is [`float16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format). Cast this data to float32 or float64 before doing any operation to avoid overflows.

#### /straighten_worm_intensity_median 
`Shape (tot_valid_skel, n_length)`

Averaged intensity along the skeleton. Calculated in [INT_PROFILE](EXPLANATION.md/#INT_PROFILE) and used by [INT_SKE_ORIENT](EXPLANATION.md/#INT_SKE_ORIENT). The data is organized as in [/straighten_worm_intensity](#straighten_worm_intensity).


### basename_features.hdf5

#### /coordinates/*
Contour and skeleton coordinates after smoothing. Each index in the first dimension correspond to a row in [`/features_timeseries`](#features_timeseries). 

#### /features_timeseries
Table containing the features that can be represented as timeseries. Each row corresponds to a single (`worm_index`, `timestamp`) pair.

  * `worm_index` : trajectory index. Same as `worm_index_joined` in [/trajectories_data](#trajectories_data).
  * `timestamp` : video timestamp indexes. Should be continous. The real space between indexes should be `1/frames per second`. 
  * `skeleton_id` : corresponding row in the [/trajectories_data](#trajectories_data) table. It should be -1 if there is not a match (dropped frames).
  * `motion_modes` : vector indicating if the worm is `moving forward (1)`, `backwards (-1)` or it is `paused (0)`.
  * `length` : skeleton length calculated using `/coordinates/skeleton`. 
  * `head_width`, `midbody_width`, `tail_width` : contour width for each worm region.
  * `area` : contour area calculated using the [shoelace formula](https://en.wikipedia.org/wiki/Shoelace_formula). 
  * `area_length_ratio` : `area/length`
  * `width_length_ratio` : `midbody_width/length`
  * `max_amplitude` : 
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



#### /features_summary: 
Set of features calculated by subdividing and reducing the features in [/features_timeseries](#features_timeseries) and the [/features_events/worm_* ](#features_eventsworm_).

The subdivisions can be done by movement type or/and by signed data. A feature was subdivided when the corresponding subfix is in its name. If a feature name does not contain any subdivision subfix then the reduction was applied on all the original data.

The movement subdivision are only valid for features in [/features_timeseries](#features_timeseries) and are:

* `_forward` : worm is moving forwards (`motion_modes == 1`).
* `_paused` : worm is paused (`motion_modes == 0`).
* `_backward`  : worm is moving backwards (`motion_modes == -1`).

The signed data subdivision are only applied to data where the sign have meaning, *e.g.* `midbody_speed` or `midbody_bend_mean` and not in features like `area` or `length`. The signed data subdivisions are:
* `_neg` : only the negative data is considered.
* `_pos` : only the positive data is considered.
* `_abs` : the absolute value of the data is taken before doing the reduction.

The reduction can be done by any of the following operations on a given subdivision. The data is saved in the corresponding table name.
  
  * `P10th` : 10th [percentile](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html).
  * `P90th` : 90th [percentile](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html).
  * `means` : mean value.
  * `medians` : median value (50th percentile).

Finally, there are some tables that have the subfix `_split`. For this tables long trajectories are splitted in shorter ones of at most `split_traj_time` seconds before calculating the features. This is done as an attempt to balance uneven track sizes. Otherwise a trajectory that was followed for the whole duration of the video will be counted as a single feature vector, while a trajectory that was lost and found several times will be counted as many feature vectors. This is because each time a worm is lost it would be assigned a new id when it is found again.


