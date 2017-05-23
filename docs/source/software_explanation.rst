####################
Software Explanation
####################


Video Compression
#################

This step has the double function identifing candidate regions for the tracking and to zero the background to increase the lossless compression efficiency. The algorithm aims to identify dark particles on a light background or light particles on a dark background using `adaptative thresholding <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html>`_ and paticle size filter. The filter parameters must be adjusted manually for a different setup but typically it is possible to use the same parameters under similar experimental contions. More information on how to setup this parameters can be found in the `GUI manual <https://github.com/ver228/tierpsy-tracker/edit/dev/docs/source/GUI_manual.rst>`_ .

.. image:: https://cloud.githubusercontent.com/assets/8364368/8456443/5f36a380-2003-11e5-822c-ea58857c2e52.png

The data is stored into a hdf5 container using a gzip filter. Some advantages of this format are:

- This format can significally reduce the size of a high resolution video since only the pixels corresponding to tracked particle are kept. This gain depends on codec used by the original video, and can be less important or even inexistent.

- Rapid access to specific video frames. Typically it is hard to rapidly access to a specific video frame. Most of the video readers do an approximative search when looking for a specific frame. To accurately retrive a video frame one has to read sequencially the whole video file, a process particuarly slow for large video files. The HDF5 format indexes the data in a way that makes trivial to access to specific frames.

- Metadata can be stored in the same file as the video. HDF5 format allows to store all kind of binary data into the same file. This allows to store the video metadata and analysis progress in the same file.


Creating worm trajectories
##########################

Trajectories are linked by its closest neighbor in a consecutive area. The closest neighbor must have a similar area and be closer than a specified distance, additionally the algorithm filters for large or smaller particles. 

In a second step, trajectories are joined that have a small time and spatial gap between their end and beginning, as well as similar area. Finally for visualization purposes, a video is created showing a speedup and low resolution version of the masks where trajectories are drawed over time. 

.. image:: https://cloud.githubusercontent.com/assets/8364368/26301795/25eb72ac-3eda-11e7-8a52-99dd6c49bc07.gif


 
Extracting worm skeletons
##########################

Uses the code in `getSkeletonsTables.py`, `checkHeadOrientation.py` and `WormClass.py` in the `trackWorms` directory as well as all the code in the `segWormPython` directory. 

Firstly, the center of mass and the threshold for each of the trajectories is smoothed.  This improves the estimation of the worm threshold, fills gaps where the trajectory might have been lost, and helps to produce videos where the ROI displaces gradually following individual worms.

Secondly, a ROI is thresholded, a contour is calculated, and the worm is skeletonized. The key part of this step is the skeletonization code based on `segWorm <https://github.com/openworm/SegWorm>`_. Since one has to deal with multiworm at a time speed becomes an important issue, therefore the code was optimized using Cython and C. The skeletons and contours are normalized to have the same number of points in order to store them in a simple table. The output is store in a file with the extension `basename_skeletons.hdf5`_ .

In a second part of the code the head and tail are identified by movement. Althought it is hard to determine the head and the tail from the contour, it is possible to assign "blocks" with the same orientation for skeletons in contingous frames, since the head in one skeleton will not suddenly jump to the other side of the worm within a few frames. We can then assign the relative standard deviation (SD) of the angular movement for the first and last part of the segment. If the blocks are large enough the section with the higher SD would be the head.
 
Finally, for visualization purposes movies for each individual worm trajectory are created. In frames where segworm was succesful the skeleton and contours are drawn. In fraws where segworm fail the overlay of the thresholded mask is drawn.

.. image:: https://cloud.githubusercontent.com/assets/8364368/26309647/a6b4402e-3ef5-11e7-96cd-4a037ee42868.gif


.. image:: https://cloud.githubusercontent.com/assets/8364368/26366191/089a6ca4-3fe2-11e7-91ef-77a7a78ee8ba.png


Extracting worm features
########################
Uses the code in `obtainFeatures.py` in the `FeaturesAnalysis` directory, and the movement validation repository. This part is still in progress but basically creates a normalized worm object from the '_skeletons.hdf5' tables, and extract features and mean features using the movement_validation functions. The motion data is stored in a large table with all the worms in it and with with the indexes frame_number and worm_index, where the event data is stored in individual tables for each worm. The seven hundred or so mean features are stored in another table where each worm corresponds to worm index.

TODO: 
- Filter "bad worms", meaning any particle indentified and analyzed for the tracker that it is not a worm, or any trajectory that corresponds to two or more worms in contact.

- Indentify all the trajectories that correspond to the same worm along the video. This might be a bit challenging, but I think that by extracting morphological features or even intensity maps it might be possible to identify all the trajectories segments for the same worm, even after a collision event.
- Test the feature extraction. I haven't check that the features are stored appropiately. There might be some bugs in this part.
- Explain the parameters in the `tracker_param.py`
- Explain output of each file.



Output Files
############

basename.hdf5
===============================

attributes: 
  * expected_fps := 1,
  * time_units := 'frames'
  * microns_per_pixel := 1
  * xy_units := 'pixels'
  * is_light_background := 1

**/mask** *(tot_images, im_high, im_width)*
Compressed array with the masked image.

**/full_data** *(tot_images/save_full_interval, im_high, im_width)*
Frame without mask saved every ``save_full_interval``. The saving interval is recommended to be adjusted every 5min. This field can be useful to identify changes in the background that are lost in the **/mask** dataset *e.g.* food depletion or contrast lost due to water condensation.

**/mean_intensity** *(tot_images)*
Mean intensity of a given frame. It is useful in optogenetic experiments to identify when the light is turned on.

**/timestamp/time** || **/timestamp/raw**

Timestamp extracted from the video if the ``is_extract_metadata`` flag set to ``true``. If this fields exists and are valid (there are not nan values and they increase monotonically), they will be used to calculate the ``fps`` used in subsequent parts of the analysis. The extracting the timestamp can be a slow process since it uses `ffprobe <https://ffmpeg.org/ffprobe.html>`_ to read the whole video. If you believe that your video does not have a significative number of drop frames and you know the frame rate, or simply realise that ffprobe cannot extract the timestamp correctly, I recommend to set ``is_extract_metadata`` to ``false``.

basename_subsample.avi
======================

basename_skeletons.hdf5
========================

:/plate_worms:
  * worm_index_blob: Trajectory index given initially by the program. Since there can be several short spurious tracks identified this number can be very large and does not reflect the number of final trajectories.
  * worm_index_joined: Index after joining trajectories separated by a small time gap and filtering short spurious tracks, and invalid row will be assigned ``-1``.
  * threshold: Threshold used for the image binarization.
  * frame_number: Video frame number.
  * coord_x, coord_y, box_length, box_width, angle: center coordinates, length, width and orientation of the `minimum rotated rectangle <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#minarearect>`_.
  * area: blob area.
  * bounding_box_xmin, bounding_box_xmax, bounding_box_ymin, bounding_box_ymax: `bounding rectangle <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#boundingrect>`_ coordinates.

:/trajectories_data: table containing the smoothed data and the indexes to link each row in the others table, with the corresponding worm_index and frame_number

  * frame_number: F
  * worm_index_joined: F
  * plate_worm_id: F
  * skeleton_id: row in the trajectory_data, useful to quickly recover worm data.
  * coord_x, coord_y: Centroid coordinates after smoothing **/plate_worms data**. It is used to find the ROI to calculate the skeletons. If you want to calculate the centroid features use the corresponding field in **/blob_features**.
  * threshold: value used to segment the worm in the ROI.
  * has_skeleton: flag to mark is the skeletonization was succesful
  * roi_size: F
  * area: F
  * timestamp_raw: F
  * timestamp_time: F
  * is_good_skel: F
  * skel_outliers_flag: F
  * int_map_id: F

:/blob_features:
  * coord_x, coord_y, box_length, box_width, box_orientation
  * area: `area <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#contourarea>`_
  * perimeter: `perimeter <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#arclength>`_
  * quirkiness: sqrt(1 - box_width^2 / box_width^2)
  * compactness: 4 * pi * area / (perimeter^2)
  * solidity: area / (`convex hull <http://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/shapedescriptors/hull/hull.html#>`_ area)
  * intensity_mean, intensity_std: mean and standard deviation inside the thresholded region.
  * hu0, hu1, hu2, hu3, hu4, hu5, hu6: `hu moments <http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#humoments>`_

:/contour_area:

:/contour_side1_length: 
:/contour_side2_length:
:/skeleton_length: length in pixels.

:/skeleton:
:/contour_side1:
:/contour_side2: 
  normalized coordinates. head is the first index and tail the last. The contour side is assigned to keep a clockwise-orientation. There is still work to do to find what is the ventral and dorsal side.

:/width_midbody:

:/contour_width:
  contour width along the skeleton. I'm using the output from segworm, and resampling by interpolation It might be possible to improve this.

:/intensity_analysis/switched_head_tail:
  * worm_index
  * ini_frame
  * last_frame

:/timestamp/raw:

:/timestamp/time:

basename_features.hdf5
===============================

:/coordinates/dorsal_contours:

:/coordinates/ventral_contours:

:/coordinates/skeletons:

:/features_events/worm_*:
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

:/features_timeseries:
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

:/features_summary: 
  P10th_split, P90th_split

  * P10th
  * P90th
  * means
  * medians







