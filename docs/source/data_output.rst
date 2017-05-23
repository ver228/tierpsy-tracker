############
Output Files
############

BASENAME.hdf5
#############

**/mask** *(tot_images, im_high, im_width)*

**/full_data** *(tot_images/full_image, im_high, im_width)*

**/mean_intensity** *(tot_images)*

**/timestamp/time**

**/timestamp/raw**

BASENAME_subsample.avi
######################

BASENAME_skeletons.hsf5
#######################

**/plate_worms**
 * worm_index_blob
 * worm_index_joined
 * frame_number
 * coord_x
 * coord_y
 * box_length
 * box_width
 * angle
 * area
 * bounding_box_xmin, bounding_box_xmax
 * bounding_box_ymin, bounding_box_ymax
 * threshold

**/blob_features**
 * coord_x, coord_y
 * area
 * perimeter
 * box_length
 * box_width
 * quirkiness
 * compactness
 * box_orientation
 * solidity
 * intensity_mean
 * intensity_std
 * hu0, hu1, hu2, hu3, hu4, hu5, hu6

**/trajectories_data**
 * frame_number
 * worm_index_joined
 * plate_worm_id
 * skeleton_id
 * coord_x
 * coord_y
 * threshold
 * has_skeleton
 * roi_size
 * area
 * timestamp_raw
 * timestamp_time
 * is_good_skel
 * skel_outliers_flag
 * int_map_id

**/contour_area**

**/contour_side1_length**
**/contour_side2_length**
**/skeleton_length**

**/skeleton**
**/contour_side1**
**/contour_side2**

**/width_midbody**

**/contour_width**

**/intensity_analysis/switched_head_tail**
 * worm_index
 * ini_frame
 * last_frame

**/timestamp/raw**

**/timestamp/time**

BASENAME_features.hdf5
#######################
**/coordinates/dorsal_contours**

**/coordinates/ventral_contours**

**/coordinates/skeletons**


**/features_events/worm_***
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

**/features_timeseries**
 * worm_index
 * timestamp
 * skeleton_id
 * motion_modes
 * length
 * head_width
 * midbody_width
 * tail_width
 * area
 * area_length_ratio
 * width_length_ratio
 * head_bend_mean
 * neck_bend_mean
 * midbody_bend_mean
 * hips_bend_mean
 * tail_bend_mean
 * head_bend_sd
 * neck_bend_sd
 * midbody_bend_sd
 * hips_bend_sd
 * tail_bend_sd
 * max_amplitude
 * amplitude_ratio
 * primary_wavelength
 * secondary_wavelength
 * track_length
 * eccentricity
 * bend_count
 * tail_to_head_orientation
 * head_orientation
 * tail_orientation
 * eigen_projection_1
 * eigen_projection_2
 * eigen_projection_3
 * eigen_projection_4
 * eigen_projection_5
 * eigen_projection_6
 * head_tip_speed
 * head_speed
 * midbody_speed
 * tail_speed
 * tail_tip_speed
 * head_tip_motion_direction
 * head_motion_direction
 * midbody_motion_direction
 * tail_motion_direction
 * tail_tip_motion_direction
 * foraging_amplitude
 * foraging_speed
 * head_crawling_amplitude
 * midbody_crawling_amplitude
 * tail_crawling_amplitude
 * head_crawling_frequency
 * midbody_crawling_frequency
 * tail_crawling_frequency
 * path_range
 * path_curvature

**/features_summary**
P10th_split, P90th_split
 * P10th
 * P90th
 * means
 * medians



attributes: 
  * expected_fps := 1,
  * time_units := 'frames'
  * microns_per_pixel := 1
  * xy_units := 'pixels'
  * is_light_background := 1
