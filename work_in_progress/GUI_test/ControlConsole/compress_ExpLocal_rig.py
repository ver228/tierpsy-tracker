# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys
from start_console import runMultiCMD, print_cmd_list
import fnmatch
import argparse

def checkMaskPrefix(fdir):
	#check if the root dir has a subfolder MaskedVideos otherwise add it to the end

	N = sum('MaskedVideos' == part for part in fdir.split(os.sep))

	if N > 1: 
		print('ERROR: Only one subdirectory is allowed to be named "MaskedVideos"')
		raise

	if N == 0:
		fdir =  os.path.join(fdir, 'MaskedVideos')

	return os.path.abspath(fdir)


def getCompCommands(video_dir_root, mask_dir_root, tmp_dir_root, json_file = '', \
	video_ext = '.mjpg', is_single_worm = False, \
	script_abs_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/compressSingleLocal.py'):
	
	assert os.path.exists(video_dir_root)
	video_dir_root = os.path.abspath(video_dir_root)
	
	mask_dir_root = checkMaskPrefix(mask_dir_root)
	tmp_dir_root = checkMaskPrefix(tmp_dir_root) if tmp_dir_root else mask_dir_root
	
	print('A')
	cmd_list_compress = []
	for dpath, dnames, fnames in os.walk(video_dir_root):
		for fname in fnames:
			if fnmatch.fnmatch(fname, video_ext):
				video_file = os.path.abspath(os.path.join(dpath, fname))
				assert(os.path.exists(video_file))

				subdir_path = dpath.replace(video_dir_root, '')
				if subdir_path and subdir_path[0] == os.sep: 
					subdir_path = subdir_path[1:] if len(subdir_path) >1 else ''

				mask_dir = os.path.join(mask_dir_root, subdir_path)
				if not os.path.exists(mask_dir): os.makedirs(mask_dir)

				tmp_mask_dir = os.path.join(tmp_dir_root, subdir_path)
				if not os.path.exists(tmp_mask_dir): os.makedirs(tmp_mask_dir)
				
				#create a command line with the required arguments
				cmd = ['python3', script_abs_path, video_file, mask_dir]
				#add the optional arguments if they are present
				for arg in ['tmp_mask_dir', 'json_file']:
					if eval(arg): cmd += ['--' + arg, eval(arg)]

				if is_single_worm: cmd.append('--is_single_worm')

				cmd_list_compress.append(cmd)

	return cmd_list_compress



def main(video_dir_root, mask_dir_root, tmp_dir_root, json_file, video_ext, script_abs_path, max_num_process, refresh_time, is_single_worm):

	cmd_list_compress = getCompCommands(video_dir_root = video_dir_root, 
		mask_dir_root = mask_dir_root, tmp_dir_root = tmp_dir_root, 
		json_file = json_file, video_ext = video_ext, is_single_worm = is_single_worm, 
		script_abs_path = script_abs_path)
	#cmd_list_compress = cmd_list_compress
	#display commands to be executed
	print_cmd_list(cmd_list_compress)

	#run all the commands
	runMultiCMD(cmd_list_compress, max_num_process = max_num_process, refresh_time = refresh_time)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Compress video files in the local drive using several parallel processes")
	
	parser.add_argument('video_dir_root', help='Root directory with the raw videos.')
	parser.add_argument('mask_dir_root', help='Root directory with the where the masked hdf5 files are going to be saved.')
	
	#name of the scripts used
	parser.add_argument('--script_abs_path', default='/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/compressSingleLocal.py', \
		help='Full path of the script to analyze single files.')
	
	parser.add_argument('--json_file', default='', help='File (.json) containing the tracking parameters.')
	
	parser.add_argument('--tmp_dir_root', default=os.path.join(os.path.expanduser("~"), 'Tmp'), \
		help='Temporary directory where files are going to be stored')
	
	parser.add_argument('--video_ext', default='*.mjpg', help='Extention used to find the valid video files video_dir_root')
	parser.add_argument('--is_single_worm', action='store_true', help = 'This flag indicates if the video corresponds to the single worm case.')

	parser.add_argument('--max_num_process', default=6, type=int, help='Max number of process to be executed in parallel.')
	parser.add_argument('--refresh_time', default=10, type=float, help='Refresh time in seconds of the process screen.')
	args = parser.parse_args()

	main(**vars(args))
	


	
	
	
	
    