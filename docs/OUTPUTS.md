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

#### /mean_intensity
`Shape (tot_images,)`

Mean intensity of a given frame. It is useful in optogenetic experiments to identify when the light is turned on.

#### /timestamp/time timestamp/raw

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

Intensity maps of the straigten worms described in [INT_PROFILE](EXPLANATION.md/#int_profile). Each index in the first dimension correspond to the same row in [/trajectories_data_valid](#trajectories_data_valid). Note that the data type is [`float16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format). Cast this data to float32 or float64 before doing any operation to avoid overflows.

#### /straighten_worm_intensity_median 
`Shape (tot_valid_skel, n_length)`

Averaged intensity along the skeleton. Calculated in [INT_PROFILE](EXPLANATION.md/#int_profile) and used by [INT_SKE_ORIENT](EXPLANATION.md/#int_ske_orient). The data is organized as in [/straighten_worm_intensity](#straighten_worm_intensity).


### basename_features.hdf5
This file contains the results of [FEAT_CREATE](EXPLANATION.md#feat_create). For a more detailed information of the features see the supplementary information of [Yemini et al](http://www.nature.com/nmeth/journal/v10/n9/full/nmeth.2560.html).

#### /coordinates/*
Contour and skeleton coordinates after smoothing. Each index in the first dimension correspond to a row in [`/features_timeseries`](#features_timeseries). 

#### /features_timeseries
Table containing the features that can be represented as timeseries. Each row corresponds to a single (`worm_index`, `timestamp`) pair.

`tail_to_head_orientation` This feature needs to be changed since its value is with respect to the image frame of reference that does not have a real meaning unless we are using chemotaxis experiments.

  * `worm_index` : trajectory index. Same as `worm_index_joined` in [/trajectories_data](#trajectories_data).
  * `timestamp` : video timestamp indexes. Should be continous. The real space between indexes should be `1/frames per second`. 
  * `skeleton_id` : corresponding row in the [/trajectories_data](#trajectories_data) table. It should be -1 if there is not a match (dropped frames).
  * `motion_modes` : vector indicating if the worm is `moving forward (1)`, `backwards (-1)` or it is `paused (0)`.
  * `length` : `(microns)` skeleton length calculated using `/coordinates/skeleton`. 
  * `head_width`, `midbody_width`, `tail_width` : `(microns)` contour width for each worm body region.
  * `area` : `(microns^2)` contour area calculated using the [shoelace formula](https://en.wikipedia.org/wiki/Shoelace_formula). 
  * `area_length_ratio` : `(microns)` `area/length`
  * `width_length_ratio` : `(no units)` `midbody_width/length`
  * `max_amplitude` : `(microns)` maximum amplitude of the skeleton along its major axis.
  * `amplitude_ratio` : `(no units)` ratio between the maximum amplitudes on each side of the skeleton along its major axis. The smaller amplitude is the numerator and the larger the denominator.
  * `primary_wavelength`, `secondary_wavelength` : `(microns)` larger and second larger peaks of the fourier transform of the skeleton rotated over its major axis.
  * `track_length` : `(microns)` length of the line extending from head to its tail.
  * `eccentricity` : `(no units)` [eccentricity](https://en.wikipedia.org/wiki/Eccentricity_(mathematics)) of the worm's body calculated using the [contour moments](http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html).
  * `bend_count` : `(no units)` the number of kinks (bends) in the skeleton.
  * `tail_to_head_orientation` `(degrees)` direction the worm is facing is measured as the angle between the head and the tail. 
  * `head_orientation` `(degrees)`  direction of the head measured as the angle between 1/6 of the worm body from the head and the head tip.
  * `tail_orientation` `(no units)`  direction of the tail measured as the angle between 1/6 of the worm body from the tail and the tail tip.
  * `eigen_projection_1`, `eigen_projection_2`, `eigen_projection_3`,  `eigen_projection_4`, `eigen_projection_5`, `eigen_projection_6` : `(degrees)` eigenworm coefficients calculated using the [Stephens et al., 2008](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000028) moethod. The eigenworms are calculated by applying [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to the angles between each subsequent skeleton point (an orientation-invariant representation). The PCA components are calculated on wild-type skeletons, and account for roughly 95% of the variance in N2 shapes. Another way of looking at this is that the worm shape is compressed with roughly 5% loss. 
  * `head_bend_mean`, `neck_bend_mean`, `midbody_bend_mean`, `hips_bend_mean`, `tail_bend_mean` : `(degrees)` mean of the angles along the skeleton for each worm body region.
  * `head_bend_sd`, `neck_bend_sd`, `midbody_bend_sd`, `hips_bend_sd`, `tail_bend_sd` : `(degrees)` standard deviation of the angles along the skeleton for each worm body region. The sign is given by the corresponding `_bend_mean`.
  * `head_crawling_amplitude`, `head_crawling_frequency`, `midbody_crawling_amplitude`, `midbody_crawling_frequency` `tail_crawling_amplitude`, `tail_crawling_frequency`: Frequency and amplitude of the largest peak of the fourier transform over a time window of the body part bend_mean. 
  * `head_tip_speed`, `head_speed`, `midbody_speed`, `tail_speed`, `tail_tip_speed` : `(micrometers/seconds)` body part speed. It is signed accordingly to the segment change of direction.
  * `head_tip_motion_direction`, `head_motion_direction`, `midbody_motion_direction`, `tail_motion_direction`,  `tail_tip_motion_direction` : `(degrees/seconds)` angular speed of the respective body part.
  * `foraging_amplitude` : the largest foraging angle measured (nose bend) prior to returning to a straight, unbent position. 
  * `foraging_speed` : `(degrees/seconds)` foraging angular speed. It quantifies how fast the nose is moving.
  * `path_range` : `(micrometers)` distance of the worm’s midbody from the path centroid.
  * `path_curvature` : `(radians/micrometers)` the angle of the worm's path divided by the distance travelled.

#### /features_events/worm_*
  * `worm_dwelling`, `head_dwelling`, `midbody_dwelling`, `tail_dwelling` : `(seconds)` time duration a body part spends in a specific region of the plate. The worm path is subdivided in uniform grids where each element diagonal is equal to the body part mean width. The stored vector contains the time spend in each grid, excluding the grids that the body part never visited.

Each time frame can be labeled according to any of the following events definitions (modified from the Supplementary Material of [Yemini et al.](http://www.nature.com/nmeth/journal/v10/n9/full/nmeth.2560.html):

* `forward` : the worm is moving in forward motion. It is defined over a period of 0.5 seconds where the worm travels at least 5% of its mean length and its speed is at least 5% of its length per second. The worm must maintain this conditions almost continuously with
interruptions not longer than 0.25 seconds (the interruptions allow for quick contradictory movements such as head withdrawal, body contractions, and segmentation noise).

* `backward` : the worm is moving in backward motion. Same as `forward` except the midbody speed sign must be negative.

* `paused` : the worm is moving motion is paused. It is defined over a period of 0.5 seconds where the forward and backward speed does not exceed 2.5% of the worms length per second. Similarly to `forward` the maximum permissible interruption must be less than 0.25 seconds.

* `omega_turn` : the worm bends are used to find a contiguous sequence of frames wherein a large bend travels from the worm’s head, through its midbody, to its tail. The worm’s body is separated into three equal parts from its head to its tail.  The mean supplementary angle is measured along each third. For omega turns, this angle must initially exceed 30° at the first but not the last third of the body (the head but not the tail). The middle third must then exceed 30°. And finally, the last but not the first third of the body must exceed 30° (the tail but not the head). This sequence of a 30° mean supplementary angle, passing continuously along the worm from head to tail, is labeled an omega turn event.

* `upsilon_turn` : computed nearly identically to the `omega_turn` but they capture all events that escaped being labeled omega turns, wherein the mean supplementary angle exceeded 15° on one side of the worm (the first or last third of the body) while not exceeding 30° on the opposite end. 


* `coil` : this feature is currently deactivated. It should use the annotations produced by [segWorm](https://github.com/openworm/SegWorm) during skeletonization. The original algorithm is based on the ratio between the contour midbody and head/tail width.

For each of the event each of the following features are calculated:

* `_distance` : `(micromenters)` distance travelled during the events.
* `inter_*_distance` : `(micromenters)` distance travelled between different events.
* `_time` : `(seconds)` time durations of each event.
* `inter_*_time` : `(seconds)` time between different events.
* `_frequency` : `(1/seconds)` how often an event occurs per time unit `(n_events/total_time)`.
* `_time_ratio` :  `(no units)` ratio between the time spend at the event over the total trajectory time.
* `_distance_ratio` : `(no units)` ratio between the total distance travelled during an event and the total distance travelled during the whole trajectory.

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


