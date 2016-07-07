# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
deprecated_fields = ['has_timestamp']


# ('is_single_worm', False, 
# ('min_area', 50, 'minimum allowed area in pixels.'),
# ('max_area', 1e10, 'maximum allowed area in pixels.'),
# ('thresh_C', 15,  'threshold used by the adaptative thresholding in the mask calculation.'),
# ('keep_corner_time', True, 'keep the pixels in the top left corner that correspond to the video timestamp (used only in the lab setup)'),
# ('thresh_block_size', 61, 'block size used by the adaptative thresholding'),
# ('dilation_size', 9, 'size of the structural element used in morphological operations.'),
# ('compression_buff', 25, 
# ('keep_border_data':False, 'is_invert_thresh':False, 'expected_fps':25, 'threshold_factor':1.05,
# ('resampling_N':49,  'max_gap_allowed_block':10, 'fps_filter':-1, 'min_displacement':0, 'strel_size':5, 
# 'filt_bad_seg_thresh':0.8, 'filt_min_displacement':10, 'filt_critical_alpha':0.01, 
# 'filt_max_width_ratio':2.25, 'filt_max_area_ratio':6,
# 'save_int_maps':False, 'int_avg_width_frac':0.3, 'int_width_resampling':15, 
# 'int_length_resampling':131, 'int_max_gap_allowed_block':-1
# '''
#         keep_border_data - set it to false if you want to remove any connected component that touches the border.
#         is_invert_thresh - set to true to indentify bright worms over a dark background.
        
#         expected_fps - expected frame rate
#         fps_filter - frame per second used to calcular filters for trajectories. As default it will have the same value as expected_fps. Set to zero to eliminate filtering.
        
#         min_displacement - minimum total displacement of a trajectory to be included in the analysis
#         filt_min_displacement - minimum total displacement of a trajectory to be included in calculation of the limit to determine if a skeleton is an outlier
#         filt_bad_seg_thresh - minimum fraction of succesfully skeletonized frames in a worm trajectory to be considered valid
        
#         resampling_N = number of segments used to renormalize the worm skeleton and contours
#         '''

default_param = {'is_single_worm':False, 'min_area':50, 'max_area':1e10, 'thresh_C':15, 
'has_timestamp':True, 'thresh_block_size':61, 'dilation_size':9, 'compression_buff':25, 
'keep_border_data':False, 'is_invert_thresh':False, 'expected_fps':25, 'threshold_factor':1.05,
'resampling_N':49,  'max_gap_allowed_block':10, 'fps_filter':-1, 'min_displacement':0, 'strel_size':5, 
'filt_bad_seg_thresh':0.8, 'filt_min_displacement':10, 'filt_critical_alpha':0.01, 
'filt_max_width_ratio':2.25, 'filt_max_area_ratio':6,
'save_int_maps':False, 'int_avg_width_frac':0.3, 'int_width_resampling':15, 
'int_length_resampling':131, 'int_max_gap_allowed_block':-1}

class tracker_param:
    def __init__(self, source_file=''):
        input_param = default_param.copy()
        if source_file:
            with open(source_file) as fid:
                param_in_file = json.load(fid)
            for key in param_in_file:
                if key in deprecated_fields:
                    continue
                input_param[key] = param_in_file[key]
                
        self._get_param(**input_param)
        #print(input_param)
    def _get_param(self, is_single_worm, min_area, max_area, thresh_C, 
        has_timestamp, thresh_block_size, dilation_size, compression_buff, 
        keep_border_data, is_invert_thresh, expected_fps, threshold_factor,
        resampling_N, max_gap_allowed_block, fps_filter, min_displacement, strel_size, 
        filt_bad_seg_thresh, filt_min_displacement, filt_critical_alpha, 
        filt_max_width_ratio, filt_max_area_ratio,
        save_int_maps, int_avg_width_frac, int_width_resampling, 
        int_length_resampling, int_max_gap_allowed_block):
        
        if not isinstance(expected_fps, int):
            expected_fps = int(expected_fps)
        
        self.expected_fps = expected_fps
        
        #getROIMask
        self.mask_param = {'min_area': min_area, 'max_area': max_area, 'has_timestamp': has_timestamp, 
        'thresh_block_size':thresh_block_size, 'thresh_C':thresh_C, 'dilation_size':dilation_size, 
        'keep_border_data':keep_border_data, 'is_invert_thresh': is_invert_thresh}
        
        #compressVideo
        self.compress_vid_param =  {'buffer_size' : compression_buff, 'save_full_interval' : 200*expected_fps, 
                               'max_frame' : 1e32, 'mask_param' : self.mask_param, 'expected_fps' : expected_fps}
        #getWormTrajectories
        min_track_lenght = max(1, fps_filter/5)
        max_allowed_dist = max(1, expected_fps)
        
        self.trajectories_param = {'initial_frame' : 0, 'last_frame': -1,
                              'min_area': min_area/2, 'min_length' : min_track_lenght, 'max_allowed_dist' : max_allowed_dist, 
                              'area_ratio_lim': (0.5, 2), 'buffer_size': compression_buff, 'threshold_factor' : threshold_factor,
                              'strel_size' : (strel_size, strel_size)}
        
        #joinTrajectories
        min_track_size = max(1, fps_filter*2)
        max_time_gap = max(0, fps_filter*4)
        self.join_traj_param = {'min_track_size': min_track_size, 'max_time_gap' : max_time_gap, 'area_ratio_lim': (0.67, 1.5)}
        
        #getSmoothTrajectories
        self.smoothed_traj_param = {'min_track_size' : min_track_size, 'min_displacement' : min_displacement, 
        'displacement_smooth_win': expected_fps + 1, 'threshold_smooth_win' : expected_fps*20 + 1, 'roi_size' : -1}
        
        #trajectories2Skeletons
        self.skeletons_param = {'resampling_N' : resampling_N, 'worm_midbody' : (0.33, 0.67), 
                               'min_mask_area' : min_area/2, 'smoothed_traj_param' : self.smoothed_traj_param,
                               'strel_size' : strel_size}
        
        #correctHeadTail
        if max_gap_allowed_block <0: expected_fps//2
        self.head_tail_param = {'max_gap_allowed' : max_gap_allowed_block, 'window_std' : expected_fps, 'segment4angle' : round(resampling_N/10), 
                           'min_block_size' : expected_fps*10}
        
        min_num_skel = 4*expected_fps
        #getWormFeatures
        self.feat_filt_param = {'min_num_skel' : min_num_skel, 'bad_seg_thresh' : filt_bad_seg_thresh, 
        'min_displacement' : filt_min_displacement, 'critical_alpha' : filt_critical_alpha, 
        'max_width_ratio' : filt_max_width_ratio, 'max_area_ratio' : filt_max_area_ratio}


        self.int_profile_param = {'width_resampling' : int_width_resampling, 'length_resampling' : int_length_resampling, 'min_num_skel' : min_num_skel,
                     'smooth_win' : 11, 'pol_degree' : 3, 'width_percentage' : int_avg_width_frac, 'save_int_maps' : save_int_maps}
        
        if int_max_gap_allowed_block < 0:
            int_max_gap_allowed_block = max_gap_allowed_block/2

        smooth_W = max(1, round(expected_fps/5))
        self.head_tail_int_param = {'smooth_W' : smooth_W, 'gap_size' : int_max_gap_allowed_block, 'min_block_size' : (2*expected_fps//5), 
        'local_avg_win' : 10*expected_fps, 'min_frac_in' : 0.85, 'head_tail_param' : self.head_tail_param}
