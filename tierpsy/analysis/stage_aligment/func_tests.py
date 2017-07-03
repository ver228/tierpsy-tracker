# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""
import os
import matplotlib.pylab as plt
import tables
import numpy as np
import pandas as pd
from tierpsy.helper.params import read_fps

from tierpsy.analysis.stage_aligment.findStageMovement import getFrameDiffVar, findStageMovement

def test_var_diff(masked_file, skeletons_file):
    
    with tables.File(skeletons_file, 'r') as fid:
        frame_diff_o = fid.get_node('/stage_movement/frame_diffs')[:]
        frame_diff_o = np.squeeze(frame_diff_o)
        video_timestamp_ind = fid.get_node('/timestamp/raw')[:]
        
    
    frame_diffs_d = getFrameDiffVar(masked_file);
    
    # The shift makes everything a bit more complicated. I have to remove the first frame, before resizing the array considering the dropping frames.
    if video_timestamp_ind.size > frame_diffs_d.size + 1:
        #%i can tolerate one frame (two with respect to the frame_diff)
        #%extra at the end of the timestamp
        video_timestamp_ind = video_timestamp_ind[:frame_diffs_d.size + 1];
    
    frame_diffs = np.full(np.max(video_timestamp_ind), np.nan);
    dd = video_timestamp_ind - np.min(video_timestamp_ind)-1; #shift data
    dd = dd[dd>=0];
    if frame_diffs_d.size != dd.size: 
        #the number of frames and time stamps do not match, nothing to do here
        return
    frame_diffs[dd] = frame_diffs_d;
    
    assert np.max(np.abs(frame_diff_o-frame_diffs)) < 1e-6
    
    return frame_diffs

masked_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/SCHAFER_LAB_SINGLE_WORM/MaskedVideos/L4_19C_1_R_2015_06_24__16_40_14__.hdf5'
skeletons_file = masked_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')

#frame_diffs = test_var_diff(masked_file, skeletons_file)

with pd.HDFStore(masked_file, 'r') as fid:
    stage_log = fid['/stage_log']
    xml_info = fid.get_node('/xml_info').read().decode()
        

with tables.File(skeletons_file, 'r') as fid:
    frame_diffs = fid.get_node('/stage_movement/frame_diffs')[:]
    frame_diffs = np.squeeze(frame_diffs)
    video_timestamp_ind = fid.get_node('/timestamp/raw')[:]
    

mediaTimes = stage_log['stage_time'].values;
locations = stage_log[['stage_x', 'stage_y']].values;
fps = read_fps(skeletons_file)
#%this is not the cleaneast but matlab does not have a xml parser from
#%text string
delay_str = xml_info.partition('<delay>')[-1].partition('</delay>')[0]
delay_time = float(delay_str) / 1000;
delay_frames = np.ceil(delay_time * fps);
    
#%%
frameDiffs = frame_diffs
delayFrames = delay_frames

is_stage_move, movesI, stage_locations = \
findStageMovement(frame_diffs, mediaTimes, locations, delay_frames, fps);
##%%
# 
#from tierpsy.analysis.stage_aligment.findStageMovement import \
#_norm_frame_diffs, graythreshmat, _get_small_otsu, _initial_checks, \
#_init_search, maxPeaksDistHeight
#
#mediaTimes, locations, delayFrames, fps, spareZeroTimeLocation  = \
#_initial_checks(mediaTimes, locations, delayFrames, fps)
#frameDiffs = _norm_frame_diffs(frameDiffs)
##%%
## Compute the global Otsu and small frame-difference thresholds.
## Note 1: we use the Otsu to locate frame-difference peaks corresponding to
## stage movement.
## Note 2: we use the small threshold to locate the boundaries between
## frame differences corresponding to stage movement and frame differences
## corresponding to a non-movement interval.
#gOtsuThr = graythreshmat(frameDiffs);
#
#gSmallDiffs, gSmallThr = _get_small_otsu(frameDiffs, gOtsuThr)   
##
#maxMoveFrames = delayFrames + 1; #% maximum frames a movement takes
#frames, movesI, prevPeakI, prevPeakEndI, maxMoveTime, timeOff  = \
#_init_search(frameDiffs, gOtsuThr, gSmallDiffs, gSmallThr, 
#                 mediaTimes, maxMoveFrames, fps)
## Match the media time-stage movements to the frame-difference peaks.
#mediaTimeOff = 0; #% the offset media time
#prevOtsuThr = gOtsuThr; #% the previous small threshold
#prevSmallThr = gSmallThr; #% the previous small threshold
#isShifted = False; #% have we shifted the data to try another alignment?       