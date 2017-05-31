# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:26:10 2016

@author: ajaver
"""
import os
import argparse

from tierpsy.helper.params import TrackerParams
from tierpsy.helper.misc import RunMultiCMD
from tierpsy.helper.params.docs_process_param import dflt_args_list, process_valid_options

from tierpsy.processing.helper import get_dflt_sequence, find_valid_files, remove_border_checkpoints
from tierpsy.processing.CheckFilesForProcessing import CheckFilesForProcessing
from tierpsy.processing.ProcessLocal import ProcessLocalParser

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

def processMultipleFilesFun(
        video_dir_root,
        mask_dir_root,
        results_dir_root,
        tmp_dir_root,
        json_file,
        videos_list,
        pattern_include,
        pattern_exclude,
        max_num_process,
        refresh_time,
        only_summary,
        force_start_point='',
        end_point='',
        is_copy_video=False,
        analysis_checkpoints=[],
        unmet_requirements = False,
        copy_unfinished = False):

    # calculate the results_dir_root from the mask_dir_root if it was not given
    if not results_dir_root:
        results_dir_root = getResultsDir(mask_dir_root)

    if not video_dir_root:
        video_dir_root = mask_dir_root

    param = TrackerParams(json_file)

    json_file = param.json_file
    
    if not analysis_checkpoints:
      analysis_checkpoints = get_dflt_sequence(param.p_dict['analysis_type'])
    
    
    remove_border_checkpoints(analysis_checkpoints, force_start_point, 0)
    remove_border_checkpoints(analysis_checkpoints, end_point, -1)

    walk_args = {'root_dir': video_dir_root, 
                 'pattern_include' : pattern_include,
                  'pattern_exclude' : pattern_exclude}
    
    check_args = {'video_dir_root': video_dir_root,
                  'mask_dir_root': mask_dir_root,
                  'results_dir_root' : results_dir_root,
                  'tmp_dir_root' : tmp_dir_root,
                  'json_file' : json_file,
                  'analysis_checkpoints': analysis_checkpoints,
                  'is_copy_video': is_copy_video,
                  'copy_unfinished': copy_unfinished}
    
    #get the list of valid videos
    if not videos_list:
        valid_files = find_valid_files(**walk_args)
    else:
        with open(videos_list, 'r') as fid:
            valid_files = fid.read().split('\n')
            #valid_files = [os.path.realpath(x) for x in valid_files]
            
    files_checker = CheckFilesForProcessing(**check_args)

    cmd_list = files_checker.filterFiles(valid_files, print_cmd=True)
    
    if unmet_requirements:
         files_checker._printUnmetReq()
    elif not only_summary:
        RunMultiCMD(
            cmd_list,
            local_obj = ProcessLocalParser,
            max_num_process = max_num_process,
            refresh_time = refresh_time)



def getResultsDir(mask_dir_root):
    # construct the results dir on base of the mask_dir_root
    subdir_list = mask_dir_root.split(os.sep)

    for ii in range(len(subdir_list))[::-1]:
        if subdir_list[ii] == 'MaskedVideos':
            subdir_list[ii] = 'Results'
            break
    # the counter arrived to zero, add Results at the end of the directory
    if ii == 0:
        subdir_list.append('Results')

    return (os.sep).join(subdir_list)


