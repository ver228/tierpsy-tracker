# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""

import numpy as np
import tables
import pandas as pd
import warnings

from tierpsy.helper.params import read_fps
from tierpsy.helper.misc import print_flush
from tierpsy.analysis.stage_aligment.findStageMovement import getFrameDiffVar, findStageMovement, shift2video_ref

def isGoodStageAligment(skeletons_file):
    with tables.File(skeletons_file, 'r') as fid:
        try:
            good_aligment = fid.get_node('/stage_movement')._v_attrs['has_finished'][:]
            print(good_aligment)
        except (KeyError, IndexError, tables.exceptions.NoSuchNodeError):
            good_aligment = 0

        return good_aligment in [1, 2]

def _h_get_stage_inv(skeletons_file, timestamp):
    first_frame = timestamp[0]
    last_frame = timestamp[-1]

    with tables.File(skeletons_file, 'r') as fid:
        stage_vec_ori = fid.get_node('/stage_movement/stage_vec')[:]
        timestamp_ind = fid.get_node('/timestamp/raw')[:].astype(np.int)
        rotation_matrix = fid.get_node('/stage_movement')._v_attrs['rotation_matrix']
        microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
        #2D to control for the scale vector directions
            
    # let's rotate the stage movement
    dd = np.sign(microns_per_pixel_scale)
    rotation_matrix_inv = np.dot(
        rotation_matrix * [(1, -1), (-1, 1)], [(dd[0], 0), (0, dd[1])])

    # adjust the stage_vec to match the timestamps in the skeletons
    good = (timestamp_ind >= first_frame) & (timestamp_ind <= last_frame)

    ind_ff = timestamp_ind[good] - first_frame
    stage_vec_ori = stage_vec_ori[good]

    stage_vec = np.full((timestamp.size, 2), np.nan)
    stage_vec[ind_ff, :] = stage_vec_ori
    # the negative symbole is to add the stage vector directly, instead of
    # substracting it.
    stage_vec_inv = -np.dot(rotation_matrix_inv, stage_vec.T).T


    return stage_vec_inv

def alignStageMotion(masked_file, skeletons_file):
    #%%
    fps = read_fps(skeletons_file)
    
    with tables.File(skeletons_file, 'r+') as fid:
        
        # delete data from previous analysis if any
        if not '/stage_movement' in fid:
            g_stage_movement = fid.create_group('/', 'stage_movement')
        else:
            g_stage_movement = fid.get_node('/stage_movement')

        for field in ['stage_vec', 'is_stage_move', 'frame_diffs']:
            if field in g_stage_movement:
                fid.remove_node(g_stage_movement, field)

        g_stage_movement._v_attrs['has_finished'] = 0
        
        video_timestamp_ind = fid.get_node('/timestamp/raw')[:]
        #%%
        #I can tolerate a nan in the last position
        if np.isnan(video_timestamp_ind[-1]):
            video_timestamp_ind[-1] = video_timestamp_ind[-2] 
    
        if np.any(np.isnan(video_timestamp_ind)):
            exit_flag = 80;
            warnings.warn('The timestamp is corrupt or do not exist.\n No stage correction processed. Exiting with has_finished flag {}.'.format(exit_flag))
            #turn on the has_finished flag and exit
            g_stage_movement._v_attrs['has_finished'] = exit_flag
            return
    
        video_timestamp_ind = video_timestamp_ind.astype(np.int)
        
    #%%
    # Open the information file and read the tracking delay time.
    # (help from segworm findStageMovement)
    # 2. The info file contains the tracking delay. This delay represents the
    # minimum time between stage movements and, conversely, the maximum time it
    # takes for a stage movement to complete. If the delay is too small, the
    # stage movements become chaotic. We load the value for the delay.
    with tables.File(masked_file, 'r') as fid:
        xml_info = fid.get_node('/xml_info').read().decode()
        g_mask = fid.get_node('/mask')
        # Read the scale conversions, we would need this when we want to convert the pixels into microns
        pixelPerMicronX = 1/g_mask._v_attrs['pixels2microns_x']
        pixelPerMicronY = 1/g_mask._v_attrs['pixels2microns_y']

    with pd.HDFStore(masked_file, 'r') as fid:
        stage_log = fid['/stage_log']
    
    #%this is not the cleaneast but matlab does not have a xml parser from
    #%text string
    delay_str = xml_info.partition('<delay>')[-1].partition('</delay>')[0]
    delay_time = float(delay_str) / 1000;
    delay_frames = np.ceil(delay_time * fps);
    
    normScale = np.sqrt((pixelPerMicronX ** 2 + pixelPerMicronX ** 2) / 2);
    pixelPerMicronScale =  normScale * np.array((np.sign(pixelPerMicronX), np.sign(pixelPerMicronY)));
    
    #% Compute the rotation matrix.
    #%rotation = 1;
    angle = np.arctan(pixelPerMicronY / pixelPerMicronX);
    if angle > 0:
        angle = np.pi / 4 - angle;
    else:
        angle = np.pi / 4 + angle;
    
    cosAngle = np.cos(angle);
    sinAngle = np.sin(angle);
    rotation_matrix = np.array(((cosAngle, -sinAngle), (sinAngle, cosAngle)));
    #%%
    #% Ev's code uses the full vectors without dropping frames
    #% 1. video2Diff differentiates a video frame by frame and outputs the
    #% differential variance. We load these frame differences.
    frame_diffs_d = getFrameDiffVar(masked_file);

    #%% Read the media times and locations from the log file.
    #% (help from segworm findStageMovement)
    #% 3. The log file contains the initial stage location at media time 0 as
    #% well as the subsequent media times and locations per stage movement. Our
    #% algorithm attempts to match the frame differences in the video (see step
    #% 1) to the media times in this log file. Therefore, we load these media
    #% times and stage locations.
    #%from the .log.csv file
    mediaTimes = stage_log['stage_time'].values;
    locations = stage_log[['stage_x', 'stage_y']].values;
    
    #%% The shift makes everything a bit more complicated. I have to remove the first frame, before resizing the array considering the dropping frames.
    if video_timestamp_ind.size > frame_diffs_d.size + 1:
        #%i can tolerate one frame (two with respect to the frame_diff)
        #%extra at the end of the timestamp
        video_timestamp_ind = video_timestamp_ind[:frame_diffs_d.size + 1];
    
    dd = video_timestamp_ind - np.min(video_timestamp_ind) - 1; #shift data
    dd = dd[dd>=0];
    #%%
    if frame_diffs_d.size != dd.size:
        exit_flag = 81;
        warnings.warn('Number of timestamps do not match the number read movie frames.\n No stage correction processed. Exiting with has_finished flag {}.'.format(exit_flag))
        #%turn on the has_finished flag and exit
        
        with tables.File(skeletons_file, 'r+') as fid:
             fid.get_node('/stage_movement')._v_attrs['has_finished'] = exit_flag
        return
    
    frame_diffs = np.full(int(np.max(video_timestamp_ind)), np.nan);
    frame_diffs[dd] = frame_diffs_d;
    
    
    #%% try to run the aligment and return empty data if it fails 
    try:
        is_stage_move, movesI, stage_locations = \
        findStageMovement(frame_diffs, mediaTimes, locations, delay_frames, fps);
        exit_flag = 1;
    except:
        exit_flag = 82;
        warnings.warn('Returning all nan stage vector. Exiting with has_finished flag {}'.format(exit_flag))
        
        with tables.File(skeletons_file, 'r+') as fid:
             fid.get_node('/stage_movement')._v_attrs['has_finished'] = exit_flag
        
        #%remove the if we want to create an empty 
        is_stage_move = np.ones(frame_diffs.size+1);
        stage_locations = [];
        movesI = [];
    
    #%% 
    stage_vec_d, is_stage_move_d = shift2video_ref(is_stage_move, movesI, stage_locations, video_timestamp_ind)
    
    #%% save stage data into the skeletons.hdf5
    with tables.File(skeletons_file, 'r+') as fid:
        g_stage_movement = fid.get_node('/stage_movement')
        
        fid.create_carray(g_stage_movement, 'frame_diffs', obj=frame_diffs_d)
        fid.create_carray(g_stage_movement, 'stage_vec', obj=stage_vec_d)
        fid.create_carray(g_stage_movement, 'is_stage_move', obj=is_stage_move_d)
        
        g_stage_movement._v_attrs['fps'] = fps
        g_stage_movement._v_attrs['delay_frames'] = delay_frames
        g_stage_movement._v_attrs['microns_per_pixel_scale'] = pixelPerMicronScale
        g_stage_movement._v_attrs['rotation_matrix'] = rotation_matrix
        g_stage_movement._v_attrs['has_finished'] = 1
    
    
    print_flush('Finished.')
    

if __name__ == '__main__':
    #masked_file = '/Users/ajaver/OneDrive - Imperial College London/Local_Videos/miss_aligments/trp-2 (ok298) off food_2010_04_30__13_03_40___1___8.hdf5'
    masked_file = '/Users/ajaver/Tmp/Results/N2_A_24C_R_6_2015_06_16__19_40_00__.hdf5'
    skeletons_file = masked_file.replace(
        'MaskedVideos',
        'Results').replace(
        '.hdf5',
        '_skeletons.hdf5')
    alignStageMotion(masked_file, skeletons_file)