# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
import os

#get default parameters files
from tierpsy import DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from .docs_tracker_param import default_param, info_param, valid_options
from .docs_analysis_points import dflt_analysis_points

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

def get_dflt_sequence(analysis_type):
    assert analysis_type in valid_options['analysis_type']
    if analysis_type in dflt_analysis_points:
        analysis_checkpoints = dflt_analysis_points[analysis_type]
    else:
        analysis_checkpoints = dflt_analysis_points['DEFAULT']
        
    return analysis_checkpoints

def read_params(json_file=''):
    '''
    Read and clean input.
    '''
    
    #basically look for deprecated or changed names and fix them.
    input_param = default_param.copy()
    if json_file:
        with open(json_file) as fid:
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
                raise ValueError('Parameter {} is not a valid parameter. Change its value in file {}'.format(key, json_file))
            
            if key in valid_options:
                if not param_in_file[key] in valid_options[key]:
                    raise ValueError('Parameter {} is not in the list of valid options {}'.format(param_in_file[key],valid_options[key]))


        if not input_param['analysis_checkpoints']:
            input_param['analysis_checkpoints'] = get_dflt_sequence(input_param['analysis_type'])


    return input_param    



#AFTER THE LAST MODIFICATION I DON'T THINK THIS SHOULD BE A OBJECT, 
#BUT I WOULD LEAVE IT LIKE THIS FOR THE MOMENT FOR BACK COMPATIBILITY
class TrackerParams:

    def __init__(self, json_file=''):
        #If the json_file is in the extras/param_file directory add the full path
        if json_file in DFLT_PARAMS_FILES:
            json_file = os.path.join(DFLT_PARAMS_PATH, json_file)

        self.json_file = json_file
        self.p_dict = read_params(json_file)    



if __name__=='__main__':
    json_file = ''
    params = TrackerParams(json_file)
    

