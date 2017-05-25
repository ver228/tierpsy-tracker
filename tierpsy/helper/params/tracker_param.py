# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
import os

#get default parameters files
from tierpsy import AUX_FILES_DIR, DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from tierpsy.helper.docs.tracker_param_docs import default_param, info_param, valid_options

#deprecated variables that will be ignored
deprecated_fields = [
                    'has_timestamp', 
                    'min_displacement', 
                    'fps_filter', 
                    'traj_bgnd_buff_size', 
                    'traj_bgnd_frame_gap'
                    ]

#the dict key are the old names and the value the new name
deprecated_alias = {
    'fps': 'expected_fps',
    'threshold_factor': 'worm_bw_thresh_factor',
    'is_invert_thresh' : 'is_light_background',
    'is_fluorescence' : 'is_light_background',
    'min_length' : 'traj_min_box_width',
    'min_box_width': 'traj_min_box_width',
    'max_area' : 'mask_max_area',
    'save_int_maps': 'int_save_maps',
    'is_extract_metadata':'is_extract_timestamp',
    }


def _correct_filter_model_name(filter_model_name):
    if filter_model_name:
        if not os.path.exists(filter_model_name):
            #try to look for the file in the AUX_FILES_DIR
            filter_model_name = os.path.join(AUX_FILES_DIR, filter_model_name)
        assert  os.path.exists(filter_model_name)

    return filter_model_name

def _get_all_prefix(input_params, prefix):
    return {x.replace(prefix, ''):input_params[x] for x in input_params if x.startswith(prefix)}


def _correct_json_path(json_file):
    if json_file in DFLT_PARAMS_FILES:
        json_file = os.path.join(DFLT_PARAMS_PATH, json_file)
    return json_file

def _correct_bgnd_param(bgnd_param):
    if bgnd_param['buff_size']>0 and bgnd_param['frame_gap']>0:
        return bgnd_param
    else:
        return {}

class TrackerParams:

    def __init__(self, source_file=''):
        p = self._read_clean_input(source_file)

        self.analysis_type = p['analysis_type']
        self.is_single_worm = self.analysis_type == 'SINGLE_WORM_SHAFER'
        self.use_skel_filter = True #useless but important in other functions        
        
        # getROIMask
        mask_param_f = ['mask_min_area', 'mask_max_area', 'thresh_block_size', 
        'thresh_C', 'dilation_size', 'keep_border_data', 'is_light_background']
        self.mask_param = {x.replace('mask_', ''):p[x] for x in mask_param_f}

        # compressVideo
        bgnd_param_mask_f = ['mask_bgnd_buff_size', 'mask_bgnd_frame_gap', 'is_light_background']
        self.bgnd_param_mask = {x.replace('mask_bgnd_', ''):p[x] for x in bgnd_param_mask_f}
        self.bgnd_param_mask = _correct_bgnd_param(self.bgnd_param_mask)

        self.compress_vid_param = {
            'buffer_size': p['compression_buff'],
            'save_full_interval': p['save_full_interval'],
            'mask_param': self.mask_param,
            'bgnd_param': self.bgnd_param_mask,
            'expected_fps': p['expected_fps'],
            'microns_per_pixel' : p['microns_per_pixel'],
            'is_extract_timestamp': p['is_extract_timestamp']
        }

        # createSampleVideo
        self.subsample_vid_param = {
            'time_factor' : 8, 
            'size_factor' : 5, 
            'dflt_fps' : p['expected_fps']
        }

        # getBlobsTable
        trajectories_param_f = ['traj_min_area', 'traj_min_box_width',
        'worm_bw_thresh_factor', 'strel_size', 'analysis_type', 'thresh_block_size',
        'n_cores_used']
        self.trajectories_param = {x.replace('traj_', ''):p[x] for x in trajectories_param_f}
        self.trajectories_param['buffer_size'] = p['compression_buff']
        
        # joinTrajectories
        self.join_traj_param = {
            'max_allowed_dist': p['traj_max_allowed_dist'],
            'min_track_size': 0, 
            'max_time_gap': 0, 
            'area_ratio_lim': p['traj_area_ratio_lim'],
            'is_single_worm': self.is_single_worm}

        # getSmoothTrajectories
        self.smoothed_traj_param = {
            'min_track_size': 0,
            'displacement_smooth_win': -1,
            'threshold_smooth_win': -1,
            'roi_size': p['roi_size']}

        
        self.init_skel_param = {
            'smoothed_traj_param': self.smoothed_traj_param,
            'filter_model_name' : _correct_filter_model_name(p['filter_model_name'])
            }


        self.blob_feats_param =  {'strel_size' : p['strel_size']}

        if self.analysis_type == 'ZEBRAFISH':
            skel_args = _get_all_prefix(p, 'zf_')
        else:
            skel_args = {'num_segments' : p['w_num_segments'],
                         'head_angle_thresh' : p['w_head_angle_thresh']}


        # trajectories2Skeletons

        self.skeletons_param = {
            'resampling_N': p['resampling_N'],
            'worm_midbody': (0.33, 0.67),
            'min_blob_area': p['traj_min_area'],
            'strel_size': p['strel_size'],
            'analysis_type': self.analysis_type,
            'skel_args' : skel_args}
        
        self.head_tail_param = {
            'max_gap_allowed': p['max_gap_allowed_block'], 
            'window_std': -1,
            'segment4angle': p['ht_orient_segment'],
            'min_block_size': -1}

        self.feat_filt_param = _get_all_prefix(p, 'filt_')
        self.feat_filt_param['min_num_skel'] = -1


        self.int_profile_param = {
            'width_resampling': p['int_width_resampling'],
            'length_resampling': p['int_length_resampling'],
            'min_num_skel': -1,
            'smooth_win': 11,
            'pol_degree': 3,
            'width_percentage': p['int_avg_width_frac'],
            'save_maps': p['int_save_maps']}

        
        
        self.head_tail_int_param = {
            'smooth_W': -1,
            'gap_size': p['int_max_gap_allowed_block'],
            'min_block_size': -1,
            'local_avg_win': -1,
            'min_frac_in': 0.85,
            'head_tail_param': self.head_tail_param,
            'head_tail_int_method': p['head_tail_int_method']}
        

        # getWormFeatures
        self.feats_param = {
            'feat_filt_param': self.feat_filt_param,
            'split_traj_time' : p['split_traj_time'],
            'is_single_worm': self.is_single_worm
        }

        self.p_dict = p


    def _read_clean_input(self, source_file):
        self.json_file = _correct_json_path(source_file)

        input_param = default_param.copy()
        if self.json_file:
            with open(self.json_file) as fid:
                param_in_file = json.load(fid)
            for key in param_in_file:

                if key in deprecated_fields:
                    #ignore deprecated fields
                    continue
                elif key in deprecated_alias:
                    #rename deprecated alias
                    input_param[deprecated_alias[key]] = param_in_file[key]
                
                elif key == 'min_area':
                    #special case of deprecated alias
                    input_param['mask_min_area'] = param_in_file['min_area']
                    input_param['traj_min_area'] = param_in_file['min_area']/2

                elif key in input_param:
                    input_param[key] = param_in_file[key]
                else:
                    raise ValueError('Parameter {} is not a valid parameter. Change its value in file {}'.format(key, self.json_file))
                
                if key in valid_options:
                    assert param_in_file[key] in valid_options[key]

        return input_param


if __name__=='__main__':
    json_file = ''
    params = TrackerParams(json_file)
    

