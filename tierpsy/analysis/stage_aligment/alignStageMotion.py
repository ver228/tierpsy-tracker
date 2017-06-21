# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""

import os
import subprocess as sp
import tempfile
import numpy as np
import tables
import pandas as pd
import warnings
from tierpsy.helper.params import read_fps
from tierpsy.helper.misc import print_flush, get_base_name

def alignStageMotion(
        masked_image_file,
        skeletons_file,
        tmp_dir=os.path.expanduser('~/Tmp')):

    assert os.path.exists(masked_image_file)
    assert os.path.exists(skeletons_file)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    base_name = os.path.split(masked_image_file)[1].partition('.hdf5')[0]
    # check if it was finished before
    # with tables.File(skeletons_file, 'r+') as fid:
    #     try:
    #         has_finished = fid.get_node('/stage_movement')._v_attrs['has_finished'][:]
    #     except (KeyError, IndexError, tables.exceptions.NoSuchNodeError):
    #         has_finished = 0
    # if has_finished > 0:
    #     print_flush('%s The stage motion was previously aligned.' % base_name)
    #     return

    # get the current to add as a matlab path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    start_cmd = ('matlab -nojvm -nosplash -nodisplay -nodesktop <').split()

    script_cmd = "addpath('{}'); " \
        "try, alignStageMotionSegwormFun('{}', '{}'); " \
        "catch ME, disp(getReport(ME)); " \
        "end; exit; "

    script_cmd = script_cmd.format(
        current_dir, masked_image_file, skeletons_file)

    # create temporary file to read as matlab script, works better than
    # passing a string in the command line.
    tmp_fid, tmp_script_file = tempfile.mkstemp(
        suffix='.m', dir=tmp_dir, text=True)
    with open(tmp_script_file, 'w') as fid:
        fid.write(script_cmd)

    matlab_cmd = start_cmd + [tmp_script_file]

    # call matlab and align the stage motion
    print_flush('%s Aligning Stage Motion.' % base_name)
    sp.call(matlab_cmd)
    print_flush('%s Alignment finished.' % base_name)

    # delete temporary file.
    os.close(tmp_fid)
    os.remove(tmp_script_file)

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
#%%
def alignStageMotion_new(masked_image_file, skeletons_file):
    base_name = get_base_name(masked_image_file)
    #is_prev_frame_diffs = False;
    
    fps = read_fps(skeletons_file)
    with tables.File(skeletons_file, 'r+') as fid:
        # delete data from previous analysis if any
        if not '/stage_movement':
            g_stage_movement = fid.create_group('/', 'stage_movement')
        else:
            g_stage_movement = fid.get_node('/stage_movement')

        for field in ['stage_vec', 'is_stage_move', 'frame_diffs']:
            if field in g_stage_movement:
                fid.remove_node(g_stage_movement, field)

        g_stage_movement._v_attrs['has_finished'] = 0
        
        video_timestamp_ind = fid.get_node('/timestamp/raw')[:]
        if np.any(np.isnan(video_timestamp_ind)):
            exit_flag = 80;
            warnings.warns('The timestamp is corrupt or do not exist.\n No stage correction processed. Exiting with has_finished flag %i.' , exit_flag)
            #turn on the has_finished flag and exit
            g_stage_movement._v_attrs['has_finished'] = exit_flag
            return

    
    # Open the information file and read the tracking delay time.
    # (help from segworm findStageMovement)
    # 2. The info file contains the tracking delay. This delay represents the
    # minimum time between stage movements and, conversely, the maximum time it
    # takes for a stage movement to complete. If the delay is too small, the
    # stage movements become chaotic. We load the value for the delay.
    with tables.File(masked_image_file, 'r') as fid:
        xml_info = fid.get_node('/xml_info').read()
        g_mask = fid.get_node('/mask')
        #%% Read the scale conversions, we would need this when we want to convert the pixels into microns
        pixelPerMicronX = 1/g_mask._v_attrs['pixels2microns_x']
        pixelPerMicronY = 1/g_mask._v_attrs['pixels2microns_y']

    with pd.HDFStore(masked_image_file, 'r') as fid:
        stage_log = fid['/stage_log']
    
    #%this is not the cleaneast but matlab does not have a xml parser from
    #%text string
    delay_str = xml_info.partition('<delay>')[-1].partition('</delay>')[0]
    delay_time = float(delay_str) / 1000;
    delay_frames = np.ceil(delay_time * fps);
    
    normScale = np.sqrt((pixelPerMicronX ^ 2 + pixelPerMicronX ^ 2) / 2);
    pixelPerMicronScale =  normScale * np.array((np.sign(pixelPerMicronX), np.sign(pixelPerMicronY)));
    
    #% Compute the rotation matrix.
    #%rotation = 1;
    angle = np.atan(pixelPerMicronY / pixelPerMicronX);
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
    frame_diffs_d = getFrameDiffVar(masked_image_file);

    #%% Read the media times and locations from the log file.
    #% (help from segworm findStageMovement)
    #% 3. The log file contains the initial stage location at media time 0 as
    #% well as the subsequent media times and locations per stage movement. Our
    #% algorithm attempts to match the frame differences in the video (see step
    #% 1) to the media times in this log file. Therefore, we load these media
    #% times and stage locations.
    #%from the .log.csv file
    mediaTimes = stage_log['stage_time'];
    locations = stage_log[['stage_x', 'stage_y']];
    
    #%% The shift makes everything a bit more complicated. I have to remove the first frame, before resizing the array considering the dropping frames.
    if video_timestamp_ind.size > frame_diffs_d.size + 1:
        #%i can tolerate one frame (two with respect to the frame_diff)
        #%extra at the end of the timestamp
        video_timestamp_ind = video_timestamp_ind[:frame_diffs_d.size + 1];
    
    frame_diffs = np.full(np.max(video_timestamp_ind)-1, np.nan);
    dd = video_timestamp_ind - np.min(video_timestamp_ind);
    dd = dd[dd>0];

    if frame_diffs_d.size != dd.size:
        exit_flag = 81;
        warnings.warn('Number of timestamps do not match the number read movie frames.\n No stage correction processed. Exiting with has_finished flag %i.', exit_flag)
        #%turn on the has_finished flag and exit
        
        with tables.File(skeletons_file, 'r+') as fid:
             fid.get_node('/stage_movement')._v_attrs['has_finished'] = exit_flag
        return
    
    frame_diffs[dd] = frame_diffs_d;
    
    #%% try to run the aligment and return empty data if it fails 
    try:
        [is_stage_move, movesI, stage_locations] = findStageMovement(frame_diffs, mediaTimes, locations, delay_frames, fps);
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
    
    stage_vec = np.full((is_stage_move.size,2), np.nan);
    if movesI.size == 2 and np.all(movesI==0):
        #%there was no movements
        stage_vec[:,0] = stage_locations[0];
        stage_vec[:,1] = stage_locations[1];
        
    else:
        #%convert output into a vector that can be added to the skeletons file to obtain the real worm displacements
        for kk in range(stage_locations.shape[0]):
            bot = max(1, movesI[kk,1]+1);
            top = min(is_stage_move.size, movesI[kk+1,0]-1);
            stage_vec[bot:top, 0] = stage_locations[kk,0];
            stage_vec[bot:top, 1] = stage_locations[kk,1];
        
    #%the nan values must match the spected video motions
    #assert(all(isnan(stage_vec(:,1)) == is_stage_move))
    
    #%% prepare vectors to save into the hdf5 file.
    #%Go back to the original movie indexing. I do not want to include the missing frames at this point.
    is_stage_move_d = is_stage_move[video_timestamp_ind].astype(np.int8);
    stage_vec_d = stage_vec[video_timestamp_ind, :];
    
    #%% save stage data into the skeletons.hdf5
    with tables.File(skeletons_file, 'r+') as fid:
        g_stage_movement = fid.get_node('/stage_movement')
        
        g_stage_movement.create_carray(g_stage_movement, 'frame_diffs', obj=frame_diffs_d)
        g_stage_movement.create_carray(g_stage_movement, 'stage_vec', obj=stage_vec_d)
        g_stage_movement.create_carray(g_stage_movement, 'is_stage_move', obj=is_stage_move_d)
        
        g_stage_movement._v_atttrs['fps'] = fps
        g_stage_movement._v_atttrs['delay_frames'] = delay_frames
        g_stage_movement._v_atttrs['microns_per_pixel_scale'] = pixelPerMicronScale
        g_stage_movement._v_atttrs['rotation_matrix'] = rotation_matrix
        g_stage_movement._v_attrs['has_finished'] = 1
    
    
    print_flush('Finished.')
    

if __name__ == '__main__':
    file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_1/MaskedVideos/unc-7 (cb5) on food R_2010_09_10__12_27_57__4.hdf5'
    file_skel = file_mask.replace(
        'MaskedVideos',
        'Results').replace(
        '.hdf5',
        '_skeletons.hdf5')
    alignStageMotion(file_mask, file_skel)
