# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import argparse


from MWTracker.helperFunctions.runMultiCMD import runMultiCMD, print_cmd_list
from .compressSingleLocal import compressLocal_parser
from .compressMultipleFilesHelper import checkVideoFiles, exploreDirs


compress_dflt_vals = {
    'tmp_dir_root': os.path.join(
        os.path.expanduser("~"),
        'Tmp'),
    'json_file': '',
    'pattern_include': '*.mjpg',
    'pattern_exclude': '',
    'max_num_process': 6,
    'refresh_time': 10,
    'is_single_worm': False,
    'only_summary': False,
    'clean_previous': False,
    'is_copy_video': False,
    'videos_list': ''}

compress_parser = argparse.ArgumentParser(
    description="Compress video files in the local drive using several parallel processes")

compress_parser.add_argument('video_dir_root',
                             help='Root directory with the raw videos.')
compress_parser.add_argument(
    'mask_dir_root',
    help='Root directory where the masked hdf5 files are going to be saved.')

compress_parser.add_argument(
    '--videos_list',
    default='',
    help='File containing the full path of the videos to be analyzed, otherwise there will be search from video_dir_root using pattern_include and pattern_exclude.')

# name of the scripts used
# compress_parser.add_argument('--script_abs_path', default=compress_dflt_vals['script_abs_path'], \
#	help='Full path of the script to analyze single files.')

compress_parser.add_argument(
    '--json_file',
    default=compress_dflt_vals['json_file'],
    help='File (.json) containing the tracking parameters.')

compress_parser.add_argument(
    '--tmp_dir_root',
    default=compress_dflt_vals['tmp_dir_root'],
    help='Temporary directory where files are going to be stored')

compress_parser.add_argument(
    '--pattern_include',
    default=compress_dflt_vals['pattern_include'],
    help='Pattern used to find the valid video files in video_dir_root')
compress_parser.add_argument(
    '--pattern_exclude',
    default=compress_dflt_vals['pattern_exclude'],
    help='Pattern used to exclude files in video_dir_root')
compress_parser.add_argument(
    '--max_num_process',
    default=compress_dflt_vals['max_num_process'],
    type=int,
    help='Max number of process to be executed in parallel.')
compress_parser.add_argument(
    '--refresh_time',
    default=compress_dflt_vals['refresh_time'],
    type=float,
    help='Refresh time in seconds of the process screen.')

compress_parser.add_argument(
    '--is_single_worm',
    action='store_true',
    help='This flag indicates if the video corresponds to the single worm case.')
compress_parser.add_argument(
    '--only_summary',
    action='store_true',
    help='Use this flag if you only want to print a summary of the files in the directory.')
compress_parser.add_argument(
    '--clean_previous',
    action='store_true',
    help='Use this flag if you want to delete invalid proccesed files from previous analysis.')
compress_parser.add_argument(
    '--is_copy_video',
    action='store_true',
    help='The video file would be copied to the temporary directory.')


def compressMultipleFilesFun(
        video_dir_root,
        mask_dir_root,
        tmp_dir_root,
        json_file,
        pattern_include,
        pattern_exclude,
        max_num_process,
        refresh_time,
        is_single_worm,
        only_summary,
        clean_previous,
        is_copy_video,
        videos_list):

    cvf = checkVideoFiles(
        video_dir_root,
        mask_dir_root,
        tmp_dir_root=tmp_dir_root,
        json_file=json_file,
        is_single_worm=is_single_worm,
        is_copy_video=is_copy_video)

    if not videos_list:
        valid_files = exploreDirs(
            video_dir_root,
            pattern_include=pattern_include,
            pattern_exclude=pattern_exclude)
    else:
        with open(videos_list, 'r') as fid:
            valid_files = fid.read().split('\n')

    cvf.filterFiles(valid_files)

    # delete any previous invalid finished file.
    if clean_previous and self.filtered_files['FINISHED_BAD']:
        # let's ask if you really want to delete this files.
        print('%%%%%%%%%%')
        for _, masked_image_file in cvf.filtered_files['FINISHED_BAD']:
            print(masked_image_file)
        answer = input(
            r'The previous files were labeled as incorrectly finished and are going to be remove. \nDo you want to continue (y/n)')
        if answer == 'y':
            cvf.cleanPrevious()

    # print summary
    print(
        'Total number of files that match the pattern search: %i' %
        len(valid_files))
    print('Files to be proccesed : %i' %
          len(cvf.filtered_files['SOURCE_GOOD']))
    print('Invalid source files: %i' % len(cvf.filtered_files['SOURCE_BAD']))
    print('Files that were succesfully finished: %i' %
          len(cvf.filtered_files['FINISHED_GOOD']))
    print('Invalid finished files: %i' %
          len(cvf.filtered_files['FINISHED_BAD']))

    # print(cvf.filtered_files['SOURCE_BAD'])
    if not only_summary:
        cmd_list = cvf.getCMDlist()
        # run all the commands
        print_cmd_list(cmd_list)

        runMultiCMD(
            cmd_list,
            local_obj=compressLocal_parser,
            max_num_process=max_num_process,
            refresh_time=refresh_time)
