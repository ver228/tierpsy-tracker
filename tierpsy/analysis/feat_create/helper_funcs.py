

def read_fps(fname, min_allowed_fps=1, dflt_fps=25):
        # try to infer the fps from the timestamp
    try:
        with tables.File(fname, 'r') as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]

            if np.all(np.isnan(timestamp_time)):
                raise ValueError
            fps = 1 / np.nanmedian(np.diff(timestamp_time))

            if np.isnan(fps) or fps < 1:
                raise ValueError
            is_default_timestamp = 0

    except (tables.exceptions.NoSuchNodeError, IOError, ValueError):
        with tables.File(fname, 'r') as fid:
            #the expected fps might be in a different node depending on the file
            if '/mask' in fid:
                node_name = '/mask'
            else:
                node_name = '/trajectories_data'


            node = fid.get_node(node_name)
            if 'expected_fps' in node._v_attrs:
                fps = node._v_attrs['expected_fps']
            else:
                fps = dflt_fps #default in old videos
            is_default_timestamp = 1

    return fps, is_default_timestamp

def read_microns_per_pixel(skeletons_file):
    # these function are related with the singleworm case it might be necesary to change them in the future
    try:
        with tables.File(skeletons_file, 'r') as fid:
            microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
    except (KeyError, tables.exceptions.NoSuchNodeError):
        return 1

    if microns_per_pixel_scale.size == 2:
        assert np.abs(
            microns_per_pixel_scale[0]) == np.abs(
            microns_per_pixel_scale[1])
        microns_per_pixel_scale = np.abs(microns_per_pixel_scale[0])
        return microns_per_pixel_scale
    else:
        return 1