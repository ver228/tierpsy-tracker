# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
import os
import warnings

#get default parameters files
from tierpsy import DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from .docs_tracker_param import default_param, info_param, valid_options
from .docs_analysis_points import dflt_analysis_points, dlft_analysis_type, deprecated_analysis_alias

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
    analysis_checkpoints = dflt_analysis_points[analysis_type].copy()
    return analysis_checkpoints


def fix_deprecated(param_in_file):
    '''
    basically look for deprecated or changed names and corrected with the newest versions them.
    '''
    corrected_params = {}
    for key in param_in_file:
        if key in deprecated_fields:
            #ignore deprecated fields
            continue
        elif key in deprecated_alias:
            #rename deprecated alias
            corrected_params[deprecated_alias[key]] = param_in_file[key]

        elif key == 'min_area':
            #special case of deprecated alias
            corrected_params['mask_min_area'] = param_in_file['min_area']
            corrected_params['traj_min_area'] = param_in_file['min_area']/2
        elif key == 'analysis_type':
            #correct the name of a deprecated analysis types
            vv = param_in_file['analysis_type']
            corrected_params['analysis_type'] = deprecated_analysis_alias[vv] if vv in deprecated_analysis_alias else vv

        elif key == 'filter_model_name':
            corrected_params['use_nn_filter'] = len(param_in_file['filter_model_name']) > 0 #set to true if it is not empty
        else:
            corrected_params[key] = param_in_file[key]

    return corrected_params


def fix_types(param_in_file):
    '''
    Using the GUI to set parameters leads to MWP_total_n_wells to be a str
    rather than an int. This function fixes this problem and can be used for
    any other that we may encounter
    '''
    if 'MWP_total_n_wells' in param_in_file.keys():
        if isinstance(param_in_file['MWP_total_n_wells'], str):
            param_in_file['MWP_total_n_wells'] = int(
                    param_in_file['MWP_total_n_wells'])

    return param_in_file


def read_params(json_file=''):
    '''
    Read input, and assign defults for the missing values.
    '''
    input_param = default_param.copy()
    if json_file:
        with open(json_file) as fid:
            param_in_file = json.load(fid)
        param_in_file = fix_deprecated(param_in_file)
        param_in_file = fix_types(param_in_file)

        for key in param_in_file:
            if key in input_param:
                input_param[key] = param_in_file[key]
            else:
                raise ValueError('Parameter "{}" is not a valid parameter. Change its value in file "{}"'.format(key, json_file))

            if key in valid_options:
                if input_param[key] not in valid_options[key] \
                and (int(input_param[key]) not in valid_options[key]):
                    raise ValueError('Parameter "{}" is not in the list of valid options {}'.format(param_in_file[key],valid_options[key]))

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

    @property
    def is_WT2(self):
        return self.p_dict['analysis_type'].endswith('WT2')

    @property
    def is_one_worm(self):
        analysis_type = self.p_dict['analysis_type']
        return analysis_type.endswith('WT2') or analysis_type.endswith('SINGLE')

    @property
    def use_nn_filter(self):
        _use_nn_filter = self.p_dict['use_nn_filter']
        if not _use_nn_filter and 'AEX' in self.p_dict['analysis_type']:
            warnings.warn('the use_filter_flag would be set to True since the analysis type contains AEX.')
            _use_nn_filter = True

        return _use_nn_filter



if __name__=='__main__':
    json_file = ''
    params = TrackerParams(json_file)


