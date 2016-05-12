# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import argparse
import sys

from MWTracker.helperFunctions.runMultiCMD import runMultiCMD, print_cmd_list
from .trackSingleLocal import trackLocal_parser
from .trackMultipleFilesHelper import checkTrackFiles, exploreDirs
from .trackSingleWorker import checkpoint_label

parser = argparse.ArgumentParser(description = "Track worm's hdf5 files in the local drive using several parallel processes")

parser.add_argument('mask_dir_root', help = 'Root directory with the masked videos. It must contain only the hdf5 from the previous compression step.')

parser.add_argument('--videos_list', default='', help='File containing the full path of the masked videos to be analyzed, otherwise there will be search from video_dir_root using pattern_include and pattern_exclude.')

parser.add_argument('--script_abs_path', default = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trackSingleLocal.py'), \
	help='Full path of the script to analyze single files.')

parser.add_argument('--json_file', default = '', help='File (.json) containing the tracking parameters.')

parser.add_argument('--tmp_dir_root', default = os.path.join(os.path.expanduser("~"), 'Tmp'), \
	help='Temporary directory where files are going to be stored')

parser.add_argument('--is_single_worm', action='store_true', help = 'This flag indicates if the video corresponds to the single worm case.')

parser.add_argument('--force_start_point', default='', choices = checkpoint_label, help = 'Force the program to start at a specific point in the analysis.')
parser.add_argument('--end_point', default='END', choices = checkpoint_label, help='End point of the analysis.')

parser.add_argument('--max_num_process', default = 6, type = int, help = 'Max number of process to be executed in parallel.')
parser.add_argument('--refresh_time', default = 10, type = float, help = 'Refresh time in seconds of the process screen.')

parser.add_argument('--pattern_include', default = '*.hdf5', help = 'Pattern used to find the valid video files in video_dir_root')
parser.add_argument('--pattern_exclude', default = '', help = 'Pattern used to exclude files in video_dir_root')

parser.add_argument('--only_summary', action='store_true', help='Use this flag if you only want to print a summary of the files in the directory.')

parser.add_argument('--no_prev_check', action='store_true', help='Use this flag to do not check the files for completion before starting the process.')

parser.add_argument('--use_manual_join', action='store_true', help = 'Use this flag to calculate features on manually joined data.')
parser.add_argument('--no_skel_filter', action='store_true', help = 'Use this flag to do NOT filter valid skeletons using the movie robust averages.')

args = parser.parse_args()

def trackMultipleFilesFun(mask_dir_root, tmp_dir_root, json_file, script_abs_path, \
	pattern_include, pattern_exclude, \
	max_num_process, refresh_time, force_start_point, end_point, is_single_worm, 
	only_summary, no_prev_check, use_manual_join, no_skel_filter, videos_list):

	#we want to deal with absolute paths

	ctf = checkTrackFiles(mask_dir_root, tmp_dir_root = tmp_dir_root, \
		is_single_worm = is_single_worm, json_file = json_file, force_start_point = force_start_point, end_point = end_point, \
		script_abs_path = script_abs_path, use_manual_join= use_manual_join, no_skel_filter = no_skel_filter)
	
	pattern_exclude = [pattern_exclude] + ctf.invalid_ext
	
	if not videos_list:
		valid_files = exploreDirs(mask_dir_root, pattern_include = pattern_include, pattern_exclude = pattern_exclude)
	else:
		with open(videos_list, 'r') as fid:
			valid_files = fid.read().split('\n')
			
	if not no_prev_check:
		ctf.filterFiles(valid_files)
	else:
		ctf.filtered_files['SOURCE_GOOD'] = valid_files
	
	#print summary
	print('Total number of files that match the pattern search: %i' % len(valid_files))
	
	if not no_prev_check:
		print('Files to be proccesed : %i' % len(ctf.filtered_files['SOURCE_GOOD']))
		print('Invalid source files: %i' % len(ctf.filtered_files['SOURCE_BAD']))
		print('Files that were succesfully finished: %i' % len(ctf.filtered_files['FINISHED_GOOD']))
		print('Invalid finished files: %i' % len(ctf.filtered_files['FINISHED_BAD']))

	if not only_summary:
		cmd_list = ctf.getCMDlist()
		#run all the commands
		print_cmd_list(cmd_list)
		runMultiCMD(cmd_list, local_obj=trackLocal_parser, max_num_process = max_num_process, refresh_time = refresh_time)
