#Multiworm Tracker

The Multiworm Tracker consist in 4 steps:

*Note: the tracker uses numpy arrays with the C ordering (the last dimension is the fast changing).*

1. **Video Compression:** Includes all the code in the `videoCompression` directory. This steps consist identify regions from the individual worms using local thresholding, and save a masked image where only the pixels from the worm regions are saved and the background is set to zero. This allow to efficiently used gzip to obtain a lossless compression of the image (hence the name). The output file is a hdf5 file with 3 datasets:
  - `/mask` compressed array with the masked image.
  - `/full_data` full frame (no mask) saved every large interval of frames. This value is given in the group attribute `save_interval`.
  - `/im_diff` difference between the nonzeros pixels in consecutive frames in `/mask`. This can be useful to identify corrupted frames.

 ![video_compression](https://cloud.githubusercontent.com/assets/8364368/8456443/5f36a380-2003-11e5-822c-ea58857c2e52.png)

 Additionally, for visualization purposes a speedup (25x) and lower resolution (1/4) video of the masked images is created, as well as a tiff stack with lower resolution (1/8) of the stored full frames.

 Notes:
 - This step is important for compression, but also as the first step in the tracking as worm candidates regions are identified.
 - The hdf5 storage of the masked images is important in our setup: the high resolution and the high-througput make even the jpg compressed videos too large to be kept for long time storage. However, in the future this step might be done in real time in our system. Even more, if the worm density is high or the raw video is already heavily compressed, the output hdf5 file can be larger than the original video. Therefore a latter version of the program could make the mask storage an optional parameter.

2. **Creating worm trajectories:** Uses the code in `getDrawTrajectories.py` and `getWormTrajectories.py` on the `trackWorms` directory.This step consists in obtain an estimate of the trajectories and some of the features for possible worms. In the first step, trajectories are linked by its closest neighbor in a consecutive area. The closest neighbor must have a similar area and be closer than an specified distance, additionally the algorithm filter for large or smaller particles. In a second step, trajectories that have a small time and spatial gap between their ending and begging, as well as similar area are joined. All this thresholds are specified by the user in config_param.py. Finally for visualization purposes, a video is created showing a seepdup and low resolution version of the masks where trajectories are drawed over time. 

 ![trajectories](https://cloud.githubusercontent.com/assets/8364368/8456555/1b7fe600-2004-11e5-9905-59a77187aef5.png)
 The main output of the program is a file with the extension '_trajectories.hdf5'. The hdf5 pytables file with a table named `\plate_worms` with the fields *worm_index, worm_index_joined, frame_number, coord_x, coord_y, area, perimeter, box_lenght, box_width, box_orientation, quirkiness, compactness, solidity, intensity_mean, intensity_std, threshold, bounding_box_xmin, bounding_box_xmax, bounding_box_ymin, bounding_box_ymax, segworm_id*.
Most of this fields are features with a self-explanatory name I would only to the ones I considered required clarification.
   - *worm_index:* index of the trajectory given by the program, since lots of partial spurious trajectories can be identified, this number can be very large, but the index value does not reflect the number of final trajectories returned by the program.
   - *worm_index_joined:* the trajectory index after joining close trajectories and filter for short spurious tracks. This is the number that must be used in subsequent analysis.
   - *threshold:* worm threshold over the background calculated by using finding an abrupt change in the cumulative of the intensity distribution. This number is can be noisy for some frames, but it is later improved by averaging over a large number of frames before the skeletonization step.
   - *box_lenght, box_width, box_orientation, quirkiness:* refer to the values of the rotated min bounding box calculated by openCV minAreaRect. Quirkiness is eccentricity but using box_lenght and box_width instead of an ellipse major and minor axis.
   - *bounding_box_xmin, bounding_box_xmax, bounding_box_ymin:* refers to the rectangular, not rotated, bounding of the worm.
   - *segworm_id* depecrated, used before to quickly related data from the skeletons table.
 
3. **Extracting worm skeletons:**

 ![skeleton](https://cloud.githubusercontent.com/assets/8364368/8456643/a99e69c0-2004-11e5-936e-91c0ab1120b0.png)

 ![head_tail_identification](https://cloud.githubusercontent.com/assets/8364368/8456652/b80fc1fc-2004-11e5-8d06-e52a58b493ef.png)

4. **Extracting worm features:** 

