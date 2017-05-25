import argparse
import os

from tierpsy.processing.helper import get_dflt_sequence
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
        os.path.join(os.path.expanduser("~")),
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

    ('analysis_sequence',
        '',
        'Sequence of analysis to be processed.'
        ),
    ('force_start_point',
        '',
        'Force the program to start at a specific point in the analysis.'
        ),
    ('end_point',
        '',
        'End point of the analysis.'
        ),
    #('analysis_checkpoints',
    #    '',
    #    'List of the points to be processed.'
    #    ),

    ('only_summary',
        False,
        'Use this flag if you only want to print a summary of the files in the directory.'
        ),
    ('unmet_requirements',
        False,
        'Use this flag if you only want to print the unmet requirements in the invalid source files.'
        ),
    ('refresh_time',
        10.,
        'Refresh time in seconds of the process screen.'
        ),

    ]

#I am choising this because it currently has all the available points. I would have to change it in the feature.
all_available_checkpoints = get_dflt_sequence('SINGLE_WORM_SHAFER', add_manual_feats=True)

proccess_args_dflt, proccess_args_info = repack_dflt_list(dflt_args_list, valid_options={})

class ProcessMultipleFilesParser(argparse.ArgumentParser):
    def __init__(self):
        description = "Process worm video in the local drive using several parallel processes"
        super().__init__(description=description)
        
        for name, dflt_val, help in dflt_args_list:
            
            args_d = {'help' : help}
            if isinstance(dflt_val, bool):
                args_d['action'] = 'store_true'
            else:
                args_d['default'] = dflt_val
                if isinstance(dflt_val, (int, float)):
                    args_d['type'] = type(dflt_val)

            if isinstance(dflt_val, (list, tuple)):
                args_d['nargs'] = '+'

            if name in process_valid_options:
                args_d['choices'] = process_valid_options[name]

            self.add_argument('--' + name, **args_d)



        # group = self.add_mutually_exclusive_group()

        # group.add_argument(
        #     '--analysis_type',
        #     default = 'all',
        #     choices = ['compress', 'track', 'all'],
        #     help='Type of analysis to be processed.',
        #     )

        # group.add_argument(
        #     '--analysis_checkpoints',
        #     default=dflt_vals['analysis_checkpoints'],
        #     nargs='+',
        #     help='List of the points to be processed.')
        

        # self.add_argument(
        #     '--force_start_point',
        #     default=dflt_vals['force_start_point'],
        #     choices = dflt_vals['checkpoints2process'],
        #     help='Force the program to start at a specific point in the analysis.')
        
        # self.add_argument(
        #     '--end_point',
        #     default=dflt_vals['end_point'],
        #     choices = dflt_vals['checkpoints2process'],
        #     help='End point of the analysis.')


if __name__ == '__main__':
    print(dflt_args_list)
        