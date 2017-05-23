The Multiworm Tracker consists of four steps:

*Note: the tracker uses numpy arrays with the C ordering (the last dimension is the fast changing one).*

1. **Video Compression:** Includes all the code in the `videoCompression` directory. This step identifies regions from the individual worms using local thresholding, and saves a masked image where only the pixels from the worm regions are saved and the background is set to zero. This allow to efficiently use gzip to obtain a lossless compression of the image. The output file is a hdf5 file with 3 datasets:
  - `/mask` compressed array with the masked image.
  - `/full_data` full frame (no mask) saved every large interval of frames. This value is given in the group attribute `save_interval`.
  - `/im_diff` difference between the non-zeros pixels in consecutive frames in `/mask`. This can be useful to identify corrupted frames.

 ![video_compression](https://cloud.githubusercontent.com/assets/8364368/8456443/5f36a380-2003-11e5-822c-ea58857c2e52.png)

 Notes:
 - This step is important for compression, but also as the first step (segmentation) in the tracking as worm candidates regions are identified. 
- If the worm density is high, the worm occupies a large area of the field of view or the raw video is already heavily compressed, the output hdf5 file can be larger than the original video.
 - The hdf5 storage of the masked images is important in our setup: the high resolution and high-througput make even jpg compressed videos too large to be kept for long time storage. However, in the future this step might be done in real time in our system. 


2. **Creating worm trajectories:** Uses the code in the `trackWorms` directory to obtain an estimate of the worm trajectories. In the first step (`getWormTrajectories.py`), trajectories are linked by its closest neighbor in a consecutive area. The closest neighbor must have a similar area and be closer than a specified distance, additionally the algorithm filters for large or smaller particles. In a second step, trajectories are joined that have a small time and spatial gap between their end and beginning, as well as similar area. Finally for visualization purposes, a video is created showing a speedup and low resolution version of the masks where trajectories are drawed over time. 
![trajectories](https://cloud.githubusercontent.com/assets/8364368/26301795/25eb72ac-3eda-11e7-8a52-99dd6c49bc07.gif)
 
 The main output of the program is a file with the extension '_trajectories.hdf5'. The hdf5 pytables file with a table named `\plate_worms` with the fields *worm_index, worm_index_joined, frame_number, coord_x, coord_y, area, perimeter, box_lenght, box_width, box_orientation, quirkiness, compactness, solidity, intensity_mean, intensity_std, threshold, bounding_box_xmin, bounding_box_xmax, bounding_box_ymin, bounding_box_ymax, segworm_id*.
Most of these fields are features with a self-explanatory name, here we only elaborate the ones considered requiring clarification.
   - *worm_index:* index of the trajectory given by the program, since lots of partial spurious trajectories can be identified, this number can be very large, but the index value does not reflect the number of final trajectories returned by the program.
   - *worm_index_joined:* the trajectory index after joining close trajectories and filtering for short spurious tracks. This is the number that must be used in subsequent analysis.
   - *threshold:* worm threshold over the background calculated by finding an abrupt change in the cumulative of the intensity distribution. This number is can be noisy for some frames, but it is later improved by averaging over a large number of frames before the skeletonization step.
   - *box_lenght, box_width, box_orientation, quirkiness:* refer to the values of the rotated min bounding box calculated by openCV minAreaRect. Quirkiness is eccentricity but using box_lenght and box_width instead of an ellipse major and minor axis.
   - *bounding_box_xmin, bounding_box_xmax, bounding_box_ymin:* refers to the rectangular, not rotated, bounding of the worm.
   - *segworm_id* depecrated, used before to quickly related data from the skeletons table.
 
3. **Extracting worm skeletons:** Uses the code in `getSkeletonsTables.py`, `checkHeadOrientation.py` and `WormClass.py` in the `trackWorms` directory as well as all the code in the `segWormPython` directory. 
  Firstly, the center of mass and the threshold for each of the trajectories is smoothed.  This improves the estimation of the worm threshold, fills gaps where the trajectory might have been lost, and helps to produce videos where the ROI displaces gradually following individual worms.
  Secondly, a ROI is thresholded, a contour is calculated, and the worm is skeletonized. The key part of this step is the skeletonization code based on [segWorm](https://github.com/openworm/SegWorm). Since one has to deal with multiworm at a time speed becomes an important issue, therefore the code was optimized using Cython and C. The skeletons and contours are normalized to have the same number of points in order to store them in a simple table. The output is store in a file with the extension '_skeletons.hdf5', and contain the following datasets:
  - *trajectories_data* table containing the smoothed data and the indexes to link each row in the others table, with the corresponding worm_index and frame_number:
    - worm_index_joined, frame_number: same as in plate_worm_id.
    - coord_x, coord_y: x and y coordinates of the ROI center.
    - threshold: value used to segment the worm in the ROI.
    - plate_worm_id: row in the trajectories plate_worm table.
    - skeleton_id: row in the trajectory_data, useful to quickly recover worm data.
    - has_skeleton: flag to mark is the skeletonization was succesful
  - skeleton, contour_side1, contour_side2: normalized coordinates. head is the first index and tail the last. The contour side is assigned to keep a clockwise-orientation. There is still work to do to find what is the ventral and dorsal side.
  - skeleton_length, contour_side1_length, contour_side2_length: length in pixels.
  - contour_width: contour width along the skeleton. I'm using the output from segworm, and resampling by interpolation. It might be possible to improve this.

 In a second part of the code the head and tail are identified by movement. Althought it is hard to determine the head and the tail from the contour, it is possible to assign "blocks" with the same orientation for skeletons in contingous frames, since the head in one skeleton will not suddenly jump to the other side of the worm within a few frames. We can then assign the relative standard deviation (SD) of the angular movement for the first and last part of the segment. If the blocks are large enough the section with the higher SD would be the head.
 


 
 Finally, for visualization purposes movies for each individual worm trajectory are created. In frames where segworm was succesful the skeleton and contours are drawn. In fraws where segworm fail the overlay of the thresholded mask is drawn.
 ![skeleton](https://cloud.githubusercontent.com/assets/8364368/26309647/a6b4402e-3ef5-11e7-96cd-4a037ee42868.gif)

4. **Extracting worm features:** Uses the code in `obtainFeatures.py` in the `FeaturesAnalysis` directory, and the movement validation repository. This part is still in progress but basically creates a normalized worm object from the '_skeletons.hdf5' tables, and extract features and mean features using the movement_validation functions. The motion data is stored in a large table with all the worms in it and with with the indexes frame_number and worm_index, where the event data is stored in individual tables for each worm. The seven hundred or so mean features are stored in another table where each worm corresponds to worm index.

TODO: 
- Filter "bad worms", meaning any particle indentified and analyzed for the tracker that it is not a worm, or any trajectory that corresponds to two or more worms in contact.
- Determine ventral and dorsal orientation. This could be done by the worm intensity [worm_on_progress/Intensity_analysis](https://github.com/ver228/Multiworm_Tracking/tree/master/work_on_progress/Intensity_analysis).
- Indentify all the trajectories that correspond to the same worm along the video. This might be a bit challenging, but I think that by extracting morphological features or even intensity maps it might be possible to identify all the trajectories segments for the same worm, even after a collision event.
- Test the feature extraction. I haven't check that the features are stored appropiately. There might be some bugs in this part.
- Explain the parameters in the `tracker_param.py`
- Explain output of each file.
