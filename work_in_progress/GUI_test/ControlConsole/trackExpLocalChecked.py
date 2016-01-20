# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import argparse

from start_console import runMultiCMD, print_cmd_list
from helperExpLocalChecked import checkTrackFiles, exploreDirs

def getTrackCommands(mask_dir_root, results_dir_root, tmp_dir_root='', json_file = '', end_point = '', is_single_worm = False,
	script_abs_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/trackSingleLocal.py',
	invalid_ext = ['_skeletons', '_trajectories', '_features', '_feat_ind']):
	
	mask_dir_root = os.path.abspath(mask_dir_root)
	#if masked_movies_dir[-1] != os.sep: masked_movies_dir += os.sep #add the path separator at the end the main directory 

	cmd_list_track = []
	for dpath, dnames, fnames in os.walk(mask_dir_root):
		for fname in fnames:
			if fname.endswith('.hdf5') and not any(fname.endswith(ff + '.hdf5') for ff in invalid_ext):
				masked_image_file = os.path.abspath(os.path.join(dpath, fname))
				assert(os.path.exists(masked_image_file))

				if isBadFile(masked_image_file):
					print('BAD', masked_image_file)
					continue
					#import stat
					#os.chflags(masked_image_file, not stat.UF_IMMUTABLE)
					#os.remove(masked_image_file)

	return cmd_list_track
	


def main(mask_dir_root, tmp_dir_root, json_file, script_abs_path, \
	pattern_include, pattern_exclude, \
	max_num_process, refresh_time, end_point, is_single_worm, only_summary):

	ctf = checkTrackFiles(mask_dir_root, tmp_dir_root = tmp_dir_root, \
		is_single_worm = is_single_worm, json_file = json_file, end_point = end_point, \
		script_abs_path = script_abs_path)
	
	
	pattern_exclude = [pattern_exclude] + ctf.invalid_ext
	valid_files = exploreDirs(mask_dir_root, pattern_include = pattern_include, pattern_exclude = pattern_exclude)
	
	ctf.filterFiles(valid_files)
	
	#print summary
	print('Total number of files that match the pattern search: %i' % len(valid_files))
	print('Files to be proccesed : %i' % len(ctf.filtered_files['SOURCE_GOOD']))
	print('Invalid source files: %i' % len(ctf.filtered_files['SOURCE_BAD']))
	print('Files that were succesfully finished: %i' % len(ctf.filtered_files['FINISHED_GOOD']))
	print('Invalid finished files: %i' % len(ctf.filtered_files['FINISHED_BAD']))

	if not only_summary:
		cmd_list = ctf.getCMDlist()
		#run all the commands
		print_cmd_list(cmd_list)
		runMultiCMD(cmd_list, max_num_process = max_num_process, refresh_time = refresh_time)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Track worm's hdf5 files in the local drive using several parallel processes")
	
	parser.add_argument('mask_dir_root', help = 'Root directory with the masked worm videos. It must contain only the hdf5 from the previous compression step.')
	parser.add_argument('--script_abs_path', default = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/trackSingleLocal.py', \
		help='Full path of the script to analyze single files.')
	
	parser.add_argument('--json_file', default = '', help='File (.json) containing the tracking parameters.')
	
	parser.add_argument('--tmp_dir_root', default = os.path.join(os.path.expanduser("~"), 'Tmp'), \
		help='Temporary directory where files are going to be stored')
	
	parser.add_argument('--is_single_worm', action='store_true', help = 'This flag indicates if the video corresponds to the single worm case.')

	checkpoint_list = ['TRAJ_CREATE', 'TRAJ_JOIN', 'TRAJ_VID', 'SKE_CREATE', 'SKE_ORIENT', 'FEAT_CREATE', 'FEAT_IND', 'END']
	parser.add_argument('--end_point', default='END', choices = checkpoint_list, help='End point of the analysis.')
	
	parser.add_argument('--max_num_process', default = 6, type = int, help = 'Max number of process to be executed in parallel.')
	parser.add_argument('--refresh_time', default = 10, type = float, help = 'Refresh time in seconds of the process screen.')

	parser.add_argument('--pattern_include', default = '*.hdf5', help = 'Pattern used to find the valid video files in video_dir_root')
	parser.add_argument('--pattern_exclude', default = '', help = 'Pattern used to exclude files in video_dir_root')
	
	parser.add_argument('--only_summary', action='store_true', help='Use this flag if you only want to print a summary of the files in the directory.')
	
	args = parser.parse_args()

	main(**vars(args))