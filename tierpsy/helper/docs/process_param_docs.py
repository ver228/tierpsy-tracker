'''
List of default values and description of the tierpsy.processing.progressMultipleFilesFun
'''

import os
from tierpsy.helper.misc import repack_dflt_list

dflt_args_list = [
    ('video_dir_root', 
        '', 
        'Root directory with the raw videos.'
        ),
    ('mask_dir_root', 
        '', 
        'Root directory with the masked videos. It must the hdf5 from a previous compression step.'
        ),
    ('results_dir_root', 
        '', 
        'Root directory where the tracking results will be stored. If not given it will be estimated from the mask_dir_root directory.'
        ),
    ('tmp_dir_root',
        os.path.join(os.path.expanduser("~"), 'Tmp'),
        'Temporary directory where files are going to be stored.'
        ),
    ('videos_list',
        '',
        'File containing the full path of the videos to be analyzed, otherwise there will be search from root directory using pattern_include and pattern_exclude.'
        ),
    
    ('json_file',
        '',
        'File (.json) containing the tracking parameters.'
        ),
    ('max_num_process',
        6,
        'Max number of process to be executed in parallel.'
        ),

    ('pattern_include',
        '*.hdf5',
        'Pattern used to find the valid video files in video_dir_root'
        ),
    ('pattern_exclude',
        '',
        'Pattern used to exclude files in video_dir_root'
        ),
    
    ('is_copy_video',
        False,
        'The raw video file would be copied to the temporary directory.'
        ),
    ('copy_unfinished',
        False,
        'Copy files from an uncompleted analysis in the temporary directory.'
        ),
    ('force_start_point',
        '',
        'Force the program to start at a specific point in the analysis.'
        ),
    ('end_point',
        '',
        'End point of the analysis.'
        ),
    
    ('only_summary',
        False,
        'Use this flag if you only want to print a summary of the files in the directory.'
        ),
    ('unmet_requirements',
        False,
        'Use this flag if you only want to print the unmet requirements of the invalid source files.'
        ),
    ('refresh_time',
        10.,
        'Refresh time in seconds of the process screen.'
        ),

    ]

process_valid_options = {}
proccess_args_dflt, proccess_args_info = repack_dflt_list(dflt_args_list, valid_options=process_valid_options)
