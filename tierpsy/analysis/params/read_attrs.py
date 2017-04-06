import tables
import numpy as np


def _read_expected_attr(fname, attr_name, dflt):
    with tables.File(fname, 'r') as fid:
        #the expected fps might be in a different node depending on the file
        if '/mask' in fid:
            node_name = '/mask'
        else:
            node_name = '/trajectories_data'


        node = fid.get_node(node_name)
        if attr_name in node._v_attrs:
            attr = node._v_attrs[attr_name]
        else:
            attr = dflt #default in old videos
        
        return attr

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
        fps = _read_expected_attr(fname, 'expected_fps', dflt_fps)
        is_default_timestamp = 1

    return fps, is_default_timestamp

def read_microns_per_pixel(skeletons_file):
    # these function are related with the singleworm case it might be necesary to change them in the future
    try:
        #this for the single worm case...
        with tables.File(skeletons_file, 'r') as fid:
            microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
            if microns_per_pixel_scale.size == 2:
                assert np.abs(
                    microns_per_pixel_scale[0]) == np.abs(
                    microns_per_pixel_scale[1])
                microns_per_pixel = np.abs(microns_per_pixel_scale[0])
        
    except (KeyError, tables.exceptions.NoSuchNodeError):
        try:
            microns_per_pixel =_read_expected_attr(skeletons_file, 'microns_per_pixel', dflt=1)
        except (KeyError, tables.exceptions.NoSuchNodeError):
            return 1



    return microns_per_pixel