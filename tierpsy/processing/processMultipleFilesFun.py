# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:26:10 2016

@author: ajaver
"""
import os

from tierpsy.helper.params import TrackerParams
from tierpsy.helper.misc import RunMultiCMD
from tierpsy.processing.CheckFilesForProcessing import CheckFilesForProcessing
from tierpsy.processing.ProcessWormsLocal import ProcessWormsLocalParser
from tierpsy.processing.batchProcHelperFunc import getDefaultSequence, walkAndFindValidFiles

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
        analysis_sequence='',
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
      analysis_checkpoints = getDefaultSequence(analysis_sequence, is_single_worm=param.is_single_worm)
    
    
    _removePointFromSide(analysis_checkpoints, force_start_point, 0)
    _removePointFromSide(analysis_checkpoints, end_point, -1)

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
        valid_files = walkAndFindValidFiles(**walk_args)
    else:
        with open(videos_list, 'r') as fid:
            valid_files = fid.read().split('\n')
            #valid_files = [os.path.realpath(x) for x in valid_files]
            
    files_checker = CheckFilesForProcessing(**check_args)

    cmd_list = files_checker.filterFiles(valid_files, print_cmd=True)
    
    if unmet_requirements:
         files_checker._printUnmetReq()
    
    if not only_summary:
        RunMultiCMD(
            cmd_list,
            local_obj = ProcessWormsLocalParser,
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

def _removePointFromSide(list_of_points, point, index):
    assert (index == 0) or (index == -1)
    if point:
        #move points until 
        while list_of_points and \
        list_of_points[index] != point:
            list_of_points.pop(index)
    if not list_of_points:
        raise ValueError("Point {} is not valid.".format(point))
