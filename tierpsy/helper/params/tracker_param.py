# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
import os

#get default parameters files
from tierpsy import AUX_FILES_DIR, DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from tierpsy.helper.misc import repack_dflt_list


#deprecated variables that will be ignored
deprecated_fields = ['has_timestamp', 'min_displacement', 'fps_filter', 'traj_bgnd_buff_size', 'traj_bgnd_frame_gap']
#('fps_filter', 0,  '''PROBALY USELESS (Used in joinTrajectories).  Frame per second used to calculate filters for trajectories. Set to zero to eliminate filtering.''')

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

dflt_param_list = [
    ('analysis_type', 
        'WORM', 
        'Flag that defines the type of data expected and the consequent code that would be executed.'
        ),
    ('mask_min_area', 
        50, 
        'Minimum area in pixels for an object to be included in the compression mask.'
        ),
    ('mask_max_area', 
        int(1e8), 
        'Maximum area in pixels for an object to be included in the compression mask.'
        ),
    ('thresh_C', 
        15, 
        '''
        Used to calculate Mask.  Used by the adaptative thresholding algorithm. 
        The threshold for a pixel is the block mean minus thresh_C. 
        <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>
        '''
        ),
    ('thresh_block_size', 
        61, 
        '''
        Used to calculate Mask. 
        The size of a pixel neighborhood (block) that is used to calculate 
        a threshold value for the pixel by the adaptative thresholding.
        <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>
        '''
        ),
    ('dilation_size', 
        9, 
        'Size of the structural element used by morphological operations to calculate the worm mask.'
        ),
        
    ('save_full_interval', 
        -1, 
        '''
        Frequence in frames that an unprocessed frame is going to be saved in /full_data.
        If the value is negative, it would be calculated as 200*expected_fps.
        '''
        ),
    
    ('compression_buff', 
        -1, 
        '''Number of images "min-averaged" used to calculate the image mask. 
        If the value is negative it would be set using the value of expected_fps.
        '''
        ),
    ('keep_border_data', 
        False, 
        '''
        Set it to false if you want to remove any 
        binary blob that touches the border.
        '''),
    

    ('is_light_background', 
        True, 
        '''
        Set to **true** for dark objects over a light ground 
        and **false** for light objects over a dark background.
        '''
        ),

    ('is_extract_timestamp', 
        True, 
        '''
        Set to **true** to extract metadata (timestamps) 
        from the original video file. The timestamp can be used
        to calculate the FPS and to identify drop frames.
        Extract the timestamp is a slow step since it uses ffprobe to read whole video.  
        If you believe that your video does not have a significative number of drop 
        frames and you know the frame rate, or simply realise that 
        ffprobe cannot extract the timestamp correctly, 
        it is recommended to set this value to **false**.
        '''
        ),
    ('expected_fps', 
        -1, 
        '''
        Expected frames per seconds. If the value is negative it would be set to 1 
        and the units to *frames* to calculate the worm features.
        This value will be superseded if there is a valid timestamp in the video 
        (there are not nan values and they increase monotonically), but it can be used
        to identify problems with the timestamp.
        '''
        ),
    ('microns_per_pixel', -1., 
        '''
        Pixel size in micrometers. 
        If the value is negative it would be set to 1 
        and the units to *pixels* to calculate the worm features.
        '''
        ),

    ('mask_bgnd_buff_size', 
        -1, 
        '''
        Number of images used to estimate the background subtracted during compression. 
        If it is a negative number the background subtraction is deactivated. 
        '''
        ),
    ('mask_bgnd_frame_gap', 
        -1, 
        '''
        Frame gap between the images used to estimate the background subtracted during compression. 
        If it is a negative number the background subtraction is deactivated. 
        '''
        ),
    ('worm_bw_thresh_factor', 
        1.05, 
        '''
        Factor multiplied by the threshold used to create invidual binary images 
        used to create the trajectories and calculate the skeletons.
        If the particle mask is too big after tracking increase this value, if it is too small decrease it.
        '''
        ),
    ('strel_size', 
        5, 
        'Structural element size. Used to calculate the binary masks used for the skeletons and trajectories.'
        ),
    ('traj_min_area', 
        25, 
        '''
        Minimum area in pixels for an object to be considered as a part of a trajectory.
        '''
        ),
    ('traj_min_box_width', 5, 
        'Minimum width of bounding box in pixels for an object to be considered as a part of a trajectory.'
        ),
    
    ('traj_max_allowed_dist', 
        25, 
        'Maximum displacement between frames for two particles to consider part of the same track.'
        ),
    ('traj_area_ratio_lim', 
        2, 
        'Area ratio between blob areas in different frames to be considered part of the same trajectory.'
        ),

    ('n_cores_used', 
        1, 
        '''
        EXPERIMENTAL. Number of core used. 
        Currently it is only suported by TRAJ_CREATE and it is only recommended at high particle densities.
        '''),

    ('filter_model_name', 
        '', 
        'Path to the NN Keras model used to filter worms from spurious particles. If it is empty this step will be skiped.'
        ),

    ('roi_size', 
        -1, 
        '''
        Size of the Region Of Interest used to calculate the skeleton. 
        If it is set to -1 it would be calculated from the data.
        '''
        ),
    
    ('w_num_segments', 
        24, 
        '''
        Number of segments used to calculate the skeleton curvature 
        (or half the number of segments used for the contour curvature).  
        Reduced for rounder objects and decreased for sharper organisms.'
        '''
        ),
    ('w_head_angle_thresh', 
        60, 
        'Angle threshold for a peak in the contour curvature to be considered as the head or tail.'
        ),
    
    ('resampling_N', 
        49, 
        'Number of segments used to normalize the worm skeleton and contours.'
        ),
    
    ('max_gap_allowed_block', 
        -1, 
        '''
        Maximum number of missing frames for a group of skeletons to be considered 
        part of the same group in the head/tail correction using MOVEMENT.
        If this value is negative it would be set as fps/2.
        '''
        ),
    ('ht_orient_segment', 
        -1, 
        '''
        Segment size used to calculate the head/tail angular speed 
        used to orient the head/tail by MOVEMENT.
        If this value is negative it would be set as round(resampling_N/10).
        '''
        ),
    ('filt_bad_seg_thresh', 
        0.8, 
        'minimum fraction of succesfully skeletonized frames in a worm trajectory to be considered valid.'
        ),
    ('filt_max_width_ratio', 
        2.25, 
        '''
        Maximum width radio between midbody and head or tail to be considered as a valid skeleton. 
        If the worm more than double its width from the head/tail it might be coiled.
        '''
        ),
    ('filt_max_area_ratio',
        6, 
        '''
        Maximum area ratio between head+tail and the rest of the body to be a valid skeleton. 
        Find if the head and tail too small or the body too large to be a realistic worm.'
        '''
        ),
    
    ('filt_min_displacement', 
        10, 
        '''
        Minimum total displacement of a trajectory to be used to calculate the threshold to detect bad skeletons. 
        Useful to detect pair of worms detected as a single particle. 
        '''
        ),
    ('filt_critical_alpha', 
        0.01, 
        '''Critical chi2 alpha used in the mahalanobis distance to considered a skeleton a global outlier (bad skeleton).
        Useful to detect pair of worms detected as a single particle.
        '''
        ),
    
    
    ('int_save_maps', 
        False, 
        '**true** to save the intensity maps and not only the profile along the worm major axis.'
        ),
    ('int_avg_width_frac', 
        0.3, 
        '''
        Width fraction of the intensity maps used to calculate the 
        profile along the worm major axis in the head/tail correction by intensity.
        '''
        ),
    ('int_width_resampling', 
        15, 
        'Width in pixels of the intensity maps used for the head/tail correction by intensity.'
        ),
    ('int_length_resampling', 
        131, 
        'Length in pixels of the intensity maps used for the head/tail correction by intensity.'
        ),
    ('int_max_gap_allowed_block', 
        -1, 
        '''
        Maximum number of missing frames for a group of skeletons to be considered 
        part of the same group in the head/tail correction using INTENSITY.
        If this value is negative it would be set as fps/4.
        '''
        ),
    
    ('head_tail_int_method', 
        'MEDIAN_INT', 
        'Method to used to correct head/tail based on the intensity profile.'
        ),
    ('split_traj_time', 
        300, 
        'Time in SECONDS that a worm trajectory will be subdivided to calculate the splitted features.'
        ),

    ('ventral_side', 
        '', 
        'Ventral side orientation. Used in SINGLE_WORM_SHAFER if the dataset **/experiment_info** does not exists.'
        ),
    ]



# #not tested (used for the zebra fish)
# ('zf_num_segments', 12, 'Number of segments to use in tail model.'),
# ('zf_min_angle', -90, 'The lowest angle to test for each segment. Angles are set relative to the angle of the previous segment.'),
# ('zf_max_angle', 90, 'The highest angle to test for each segment.'),
# ('zf_num_angles', 90, 'The total number of angles to test for each segment. Eg., If the min angle is -90 and the max is 90, setting this to 90 will test every 2 degrees, as follows: -90, -88, -86, ...88, 90.'),
# ('zf_tail_length', 60, 'The total length of the tail model in pixels.'),
# ('zf_tail_detection', 'MODEL_END_POINT', 'Algorithm to use to detect the fish tail point.'),
# ('zf_prune_retention', 1, 'Number of models to retain after scoring for each round of segment addition. Higher numbers will be much slower.'),
# ('zf_test_width', 2, 'Width of the tail in pixels. This is used only for scoring the model against the test frame.'),
# ('zf_draw_width', 2, 'Width of the tail in pixels. This is used for drawing the final model.'),
# ('zf_auto_detect_tail_length', True, 'Flag to determine whether zebrafish tail length detection is used. If set to True, values for zf_tail_length, zf_num_segments and zf_test_width are ignored.')

valid_options = {
    'analysis_type':['WORM', 'SINGLE_WORM_SHAFER', 'PHARYNX', 'ZEBRAFISH'],
    'ventral_side':['','clockwise','anticlockwise', 'unknown'],
    'head_tail_int_method':['MEDIAN_INT', 'HEAD_BRIGHTER']
}

default_param, info_param = repack_dflt_list(dflt_param_list, valid_options)

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
    

