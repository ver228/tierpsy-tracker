#Multiworm Tracker

The Multiworm Tracker consist in 4 steps:

*Note: the tracker uses numpy arrays with the C ordering (the last dimension is the fast changing).*

1. **Video Compression:** This steps consist identify regions from the individual worms using local thresholding, and save a masked image where only the pixels from the worm regions are saved and the background is set to zero. This allow to efficiently used gzip to obtain a lossless compression of the image (hence the name). The output file is a hdf5 file with 3 datasets:
  - `/mask` compressed array with the masked image.
  - `/full_data` full frame (no mask) saved every large interval of frames. This value is given in the group attribute `save_interval`.
  - `/im_diff` difference between the nonzeros pixels in consecutive frames in `/mask`. This can be useful to identify corrupted frames.

Notes:
- This step is important for compression, but also as the first step in the tracking as worm candidates regions are identified.
- The hdf5 storage of the masked images is important in our setup: the high resolution and the high-througput make even the jpg compressed videos too large to be kept for long time storage. However, in the future this step might be done in real time in our system. Even more, if the worm density is high or the raw video is already heavily compressed, the output hdf5 file can be larger than the original video. Therefore a latter version of the program could make the mask storage an optional parameter.


2. **Creating worm trajectories:** 
