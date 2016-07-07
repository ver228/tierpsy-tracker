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


track_dflt_vals = {'results_dir_root':'', 'tmp_dir_root': os.path.join(os.path.expanduser("~"), 'Tmp'),
	'videos_list':'','json_file':'', 'pattern_include': '*.hdf5', 'pattern_exclude' : '',
	'script_abs_path' : os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trackSingleLocal.py'),
	'max_num_process':6, 'refresh_time':10, 'force_start_point':'', 'end_point':'END',
	'is_single_worm':False, 'only_summary':False, 'no_prev_check':False, 'use_manual_join':False, 
	'no_skel_filter':False }

track_parser = argparse.ArgumentParser(description = "Track worm's hdf5 files in the local drive using several parallel processes")

track_parser.add_argument('mask_dir_root', 
	help = 'Root directory with the masked videos. It must contain only the hdf5 from the previous compression step.')

track_parser.add_argument('--results_dir_root', default =track_dflt_vals['results_dir_root'], 
	help = 'Root directory with the results of the tracking will be stored videos. If not given it will be estimated from the mask_dir_root directory.')

track_parser.add_argument('--videos_list', default=track_dflt_vals['videos_list'], 
	help='File containing the full path of the masked videos to be analyzed, otherwise there will be search from video_dir_root using pattern_include and pattern_exclude.')

track_parser.add_argument('--script_abs_path', default=track_dflt_vals['script_abs_path'] , \
	help='Full path of the script to analyze single files.')

track_parser.add_argument('--json_file', default=track_dflt_vals['json_file'], 
	help='File (.json) containing the tracking parameters.')

track_parser.add_argument('--tmp_dir_root', default=track_dflt_vals['tmp_dir_root'], \
	help='Temporary directory where files are going to be stored')

track_parser.add_argument('--force_start_point', default=track_dflt_vals['force_start_point'], 
	choices = checkpoint_label, help = 'Force the program to start at a specific point in the analysis.')
track_parser.add_argument('--end_point', default=track_dflt_vals['end_point'], 
	choices = checkpoint_label, help='End point of the analysis.')
track_parser.add_argument('--max_num_process', default=track_dflt_vals['max_num_process'], 
	type = int, help = 'Max number of process to be executed in parallel.')
track_parser.add_argument('--refresh_time', default=track_dflt_vals['refresh_time'], 
	type = float, help = 'Refresh time in seconds of the process screen.')

track_parser.add_argument('--pattern_include', default=track_dflt_vals['pattern_include'] , 
	help = 'Pattern used to find the valid video files in video_dir_root')
track_parser.add_argument('--pattern_exclude', default=track_dflt_vals['pattern_exclude'] , 
	help = 'Pattern used to exclude files in video_dir_root')

track_parser.add_argument('--is_single_worm', action='store_true', 
	help = 'This flag indicates if the video corresponds to the single worm case.')
track_parser.add_argument('--only_summary', action='store_true', 
	help='Use this flag if you only want to print a summary of the files in the directory.')
track_parser.add_argument('--no_prev_check', action='store_true', 
	help='Use this flag to do not check the files for completion before starting the process.')
track_parser.add_argument('--use_manual_join', action='store_true', 
	help = 'Use this flag to calculate features on manually joined data.')
track_parser.add_argument('--no_skel_filter', action='store_true', 
	help = 'Use this flag to do NOT filter valid skeletons using the movie robust averages.')

def getResultsDir(mask_dir_root):
	#construct the results dir on base of the mask_dir_root
	subdir_list = mask_dir_root.split(os.sep)

	for ii in range(len(subdir_list))[::-1]:
		if subdir_list[ii] == 'MaskedVideos':
			 subdir_list[ii] = 'Results'
			 break
	#the counter arrived to zero, add Results at the end of the directory
	if ii == 0: subdir_list.append('Results') 
	
	return (os.sep).join(subdir_list)

def trackMultipleFilesFun(mask_dir_root, results_dir_root, tmp_dir_root, json_file, script_abs_path, \
	pattern_include, pattern_exclude, \
	max_num_process, refresh_time, force_start_point, end_point, is_single_worm, 
	only_summary, no_prev_check, use_manual_join, no_skel_filter, videos_list):

	#calculate the results_dir_root from the mask_dir_root if it was not given
	if not results_dir_root:
		results_dir_root = getResultsDir(mask_dir_root)

	ctf = checkTrackFiles(mask_dir_root, results_dir_root, tmp_dir_root = tmp_dir_root, \
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
