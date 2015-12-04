# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys
from start_console import runMultiCMD
import fnmatch

def getTrackCommands(video_dir_root, mask_dir_root, tmp_dir_root, json_file = '', video_ext = '.mjpg',
	script_abs_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/compressSingleLocal.py'):
	
	video_dir_root = os.path.abspath(video_dir_root)
	
	cmd_list_compress = []
	for dpath, dnames, fnames in os.walk(video_dir_root):
		for fname in fnames:
			if fnmatch.fnmatch(fname, video_ext):
				video_file = os.path.abspath(os.path.join(dpath, fname))
				assert(os.path.exists(video_file))

				subdir_path = dpath.replace(video_dir_root, '')
				if subdir_path and subdir_path[0] == os.sep: 
					subdir_path = subdir_path[1:] if len(subdir_path) >1 else ''

				mask_dir = os.path.abspath(os.path.join(mask_dir_root, 'MaskedVideos', subdir_path))
				#mask_dir = os.path.abspath(os.path.join(mask_dir_root, subdir_path))
				tmp_masked_dir = os.path.abspath(os.path.join(tmp_dir_root, 'MaskedVideos', subdir_path))
				
				if not os.path.exists(mask_dir): os.makedirs(mask_dir)
				if not os.path.exists(tmp_masked_dir): os.makedirs(tmp_masked_dir)
				
				cmd_list_compress += [['python3',  script_abs_path, video_file, mask_dir, tmp_masked_dir, json_file]]

	return cmd_list_compress

if __name__ == '__main__':
	#name of the scripts used
	scripts_dir = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/'
	script_abs_path = scripts_dir + 'compressSingleLocal.py'

	#create temporary directories. For the moment the user is responsable to clean the directories when the scripts finish
	tmp_dir_root = os.path.join(os.path.expanduser("~"), 'Tmp')

	#input parameters
	max_num_process = 12
	video_ext = '*.mjpg'
	json_file = ''

	video_dir_root = sys.argv[1]
	assert os.path.exists(video_dir_root)
	if video_dir_root[-1] != os.sep: video_dir_root += os.sep #add the path separator at the end the main directory 

	if len(sys.argv) > 2:
		mask_dir_root = sys.argv[2]
	else:
		mask_dir_root = '/Volumes/behavgenom$/GeckoVideo'
	if mask_dir_root[-1] != os.sep: mask_dir_root += os.sep

	if len(sys.argv) > 3:
		json_file = sys.argv[3]
	else:
		json_file = ''

	if len(sys.argv) > 4:
		video_ext = sys.argv[4]

	cmd_list_compress = getTrackCommands(video_dir_root, mask_dir_root, tmp_dir_root, json_file, video_ext, script_abs_path)
	

	cmd_list_compress = cmd_list_compress
	#display commands to be executed
	if cmd_list_compress: 
		
		for cmd in cmd_list_compress: 
			for ii, dd in enumerate(cmd):
				
				if ii >= 2:
					dd = '"' + dd + '"'

				if ii == 0:
					cmd_str = dd
				else:
					cmd_str += ' ' +  dd
			print(cmd_str)

	print(len(cmd_list_compress))
	#run all the commands
	runMultiCMD(cmd_list_compress, max_num_process = max_num_process, refresh_time = 10)
    