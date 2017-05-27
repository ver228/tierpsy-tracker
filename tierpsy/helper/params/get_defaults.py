
import tables
import numpy as np
from .read_attrs import read_unit_conversions

def _correct_dflts(params, convertions):
    '''
    Replace a value in params by convertions only if a "bad" value was given.
    '''
    for key in convertions:
        if key in params:
            param = params[key]
            if param is None or param <=0:
                params[key] = convertions[key]
    return params


def _read_correct_fps(fname, expected_fps=-1):
    DFLT_FPS = 25 #to be used in case no valid FPS can be read for the file

    try:
        #try to read the fps from the file
        fps_out = read_unit_conversions(fname)[0]
        fps, expected_fps, time_units = fps_out
    except OSError:
        #likely file does not exists yet
        fps, time_units = -1, 'frames'

    
    if fps <= 0 or time_units == 'frames': 
        #if it is not valid try to use the user provided value
        if expected_fps > 0:
            fps = expected_fps
        else:
            #if it is still bad value the hardcoded DFLT_FPS
            fps = DFLT_FPS

    assert fps > 0
    return fps

def _read_resampling_N(fname):
    with tables.File(fname, 'r') as fid:
        resampling_N = fid.get_node('/skeleton').shape[1]
        return resampling_N


def compress_defaults(fname, expected_fps=-1, **params):
    fps = _read_correct_fps(fname, expected_fps)
    
    convertions = dict(
        buffer_size = int(round(fps)),
        save_full_interval = int(200 * fps)
    )
    
    return _correct_dflts(params, convertions)

def traj_create_defaults(fname, buffer_size):
    output = compress_defaults(fname, buffer_size=buffer_size)
    return output['buffer_size']

def ske_init_defaults(fname, **params):
    fps = _read_correct_fps(fname)

    convertions = dict(
        displacement_smooth_win = int(round(fps+1)),
        threshold_smooth_win = int(round((fps*20)+1))
    )
    return _correct_dflts(params, convertions)
    
def head_tail_defaults(fname, **params):
    fps = _read_correct_fps(fname)
    resampling_N = _read_resampling_N(fname)

    convertions = dict(
        max_gap_allowed = max(1, int(fps//2)),
        window_std = int(round(fps)),
        min_block_size = int(round(10 * fps)),
        segment4angle = int(round(resampling_N / 10))
    )
    return _correct_dflts(params, convertions)

def head_tail_int_defaults(fname, **params):
    
    fps = _read_correct_fps(fname)

    convertions = dict(
        smooth_W = max(1, int(round(fps / 5))),
        gap_size = max(1, int(fps//4)),
        min_block_size = int(round(2*fps/5)),
        local_avg_win =  int(round(10 * fps))
    )
    return _correct_dflts(params, convertions)

def min_num_skel_defaults(fname, **params):
    fps = _read_correct_fps(fname)
    convertions = dict(
        min_num_skel = int(round(4 * fps))
        )
    output = _correct_dflts(params, convertions)
    
    if len(output) == 1:
        return output['min_num_skel']
    else:
        return output



