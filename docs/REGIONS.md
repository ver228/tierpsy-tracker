# Motion mode and food region detection

The `motion_mode` event flag vector indicates whether a worm is moving `forward (1)`, `backwards (-1)`, or is `paused (0)`.
In a similar fashion, the `food_region` event flag vector indicates the position of a worm with respect to a food patch, which can be classified as `inside (1)`, `outside(-1)`, or `edge (0)` (Note: `food_region` is only available on [AEX](https://www.tierpsy.com) data for the time being).

The `motion_mode` and `food_region` event flags are determined using the same algorithm, which ultimately assigns a `lower (-1)`, `central (0)`, `upper (1)` flag to each entry of the appropriate array. For `motion_mode`, this array is be `speed`, while for `food_region` we use `dist_from_food_edge`.

## The strategy
Here is a quick summary of how the algorithm works. This example shows the `motion_mode`, but the algorithm is the same for `food_region`.

1. Fill `NaN` values in the input `vec` array (`speed` here) with the nearest non-NaN neighbour

2. Smooth `vec` using a window of `smooth_window` frames
<img src="https://user-images.githubusercontent.com/33106690/84380490-50ceac80-abdf-11ea-93c0-f96ad384016a.png" width=800>

3. Find regions where the smoothed `vec` being between `(-central_th, +central_th)`, and last more than `min_frame_range` frames. These regions are certainly `central` regions, and are shown in white in the figure below. Regions not `central` are considered `active` and are in orange below.
<img src="https://user-images.githubusercontent.com/33106690/84387659-461a1480-abeb-11ea-8dd7-aa6e849f72e7.png" width=800>

4. Find entries that are certainly `lower` (smoothed `vec` is `< -extrema_th`) and `upper` (smoothed `vec` is `> +extrema_th`). Entries belonging to `lower` and `upper` regions are respectively in pink and green markers in the figure below
<img src="https://user-images.githubusercontent.com/33106690/84387661-474b4180-abeb-11ea-99fb-eaf61f6fc17e.png" width=800>

5. Entries in an `active` region but not certainly `upper` or `lower` (grey dots in orange region below) are labelled as `upper` or `lower` according to these rules:
    * if in an `active` region there are only *either* certainly `upper` (or certainly `lower`) entries, the "uncertain" entries are labelled as `upper` (or `lower`)
    * if an `active` region contains both uncertain entries *and both* certainly `upper` *and* certainly `lower` entries, "uncertain" entries are given the label of the previous certain entries (e.g. an uncertain entry following a certain `upper` will be classified as `upper` as well). Any entry which is still "uncertain" is given the label of the following certain entry. Examples can be seen in the figure above: "uncertain" points labelled in this step will show as a grey dot surrounded by a green or pink circle
    <img src="https://user-images.githubusercontent.com/33106690/84387670-49150500-abeb-11ea-96f3-a11e8c660242.png" width=800>

6. Since the event flag is based on the values of `vec`, entries where `vec` is `NaN` are ill-defined. An approach would be to just set the event flag to be `NaN` at all entries where `vec` was `NaN` (see example in figure below). However this would bring to an artificial increase of the subdivision in regions, that can reflect in features like the median length of a `paused` bout being underestimated. To prevent excessive fragmentation of `motion_mode` and `food_region` states, we only set the entries of an event flag vector to `NaN` if the corresponding entry of `vec` was a both a `NaN` and more than `smooth_window` frames away by a valid `vec` entry.
<img src="https://user-images.githubusercontent.com/33106690/84387684-4ca88c00-abeb-11ea-97bf-97a681453cf4.png" width=800>

The final result can be seen below: since the worm's `speed` was only undefined for short times, the `motion_mode` flag assumes a valid value throughout the worm's trajectory.

<img src="https://user-images.githubusercontent.com/33106690/84387692-4f0ae600-abeb-11ea-85ca-209e5e065c27.png" width=800>
