# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

min_area = 50 #minimum area in pixels allowed
max_area = 5000 #maximum area in pixels allowed
thresh_C = 15 #use 15 for worm rig and 60 for zeiss dissecting microscope
expected_frames = 15000 #expected number of frames in the video
fps = 25 #frame rate
roi_size = 128 #region of interest size (pixels) used for the skeletonization and individual worm videos
bad_seg_thresh = 0.5

#absolute path for the movement validation repository
movement_validation_dir = '/Users/ajaver/GitHub_repositories/movement_validation'
#movement_validation_dir = '/Users/ajaver/Documents/GitHub/movement_validation'

import os
assert os.path.exists(movement_validation_dir)

if not isinstance(fps, int):
    fps = int(fps)

#getROIMask
mask_param = {'min_area': min_area*2, 'max_area': max_area, 'has_timestamp': True, 
'thresh_block_size':61, 'thresh_C':thresh_C}

#compressVideo
compress_vid_param =  {'buffer_size' : fps, 'save_full_interval' : 200*fps, 
                       'max_frame' : 1e32, 'useVideoCapture': True, 
                       'expected_frames':expected_frames, 'mask_param':mask_param}
#getWormTrajectories
get_trajectories_param = {'initial_frame':0, 'last_frame': -1,
                      'min_area':min_area, 'min_length':5, 'max_allowed_dist':20, 
                      'area_ratio_lim': (0.5, 2), 'buffer_size': fps}

#joinTrajectories
join_traj_param = {'min_track_size': fps*2, 'max_time_gap' : fps*4, 'area_ratio_lim': (0.67, 1.5)}

#getSmoothTrajectories
smoothed_traj_param = {'min_displacement' : 0, 'displacement_smooth_win': fps*4 + 1, 
                       'threshold_smooth_win' : fps*20 + 1}

#trajectories2Skeletons
get_skeletons_param = {'roi_size' : roi_size, 'resampling_N' : 49, 
                       'min_mask_area' : min_area, 'smoothed_traj_param' : smoothed_traj_param}

#correctHeadTail
head_tail_param = {'max_gap_allowed' : fps//2, 'window_std' : fps, 'segment4angle' : 5, 
                   'min_block_size' : fps*10}


#writeIndividualMovies
ind_mov_param = {'roi_size' : roi_size, 'fps' : fps, 
                  'bad_seg_thresh' : bad_seg_thresh, 'save_bad_worms': False}

#getWormFeatures
features_param = {'bad_seg_thresh' : bad_seg_thresh, 'fps' : fps}