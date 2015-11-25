# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys
from start_console import runMultiCMD


def getTrackCommands(mask_dir_root, results_dir_root, tmp_dir_root, json_file = '', 
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

				subdir_path = dpath.replace(video_dir_root, '')
				if subdir_path and subdir_path[0] == os.sep: 
					subdir_path =  subdir_path[1:] if len(subdir_path[0]) >1 else ;;
						
					else:
						subdir_path = ''

				results_dir = os.path.abspath(os.path.join(results_dir_root, subdir_path))
				tmp_masked_dir = os.path.abspath(os.path.join(tmp_dir_root, 'MaskedVideos', subdir_path))
				tmp_results_dir = os.path.abspath(os.path.join(tmp_dir_root, 'Results', subdir_path))

				if not os.path.exists(results_dir): os.makedirs(results_dir)
				if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)
				if not os.path.exists(tmp_masked_dir): os.makedirs(tmp_masked_dir)
				cmd_list_track += [['python3', script_abs_path, masked_image_file, results_dir, tmp_masked_dir, tmp_results_dir, json_file]]

	return cmd_list_track
	


if __name__ == '__main__':
	#name of the scripts used
	scripts_dir = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/'
	script_abs_path = scripts_dir +  'trackSingleLocal.py'

	#create temporary directories. For the moment the user is responsable to clean the directories when the scripts finish
	tmp_dir_root = os.path.join(os.path.expanduser("~"), 'Tmp')

	#input parameters
	max_num_process = 6

	mask_dir_root = sys.argv[1]
	assert os.path.exists(mask_dir_root)

	if len(sys.argv) > 2:
		json_file = sys.argv[2]
	else:
		json_file = ''

	if json_file: assert(os.path.exists(json_file))

	results_dir_root = mask_dir_root.replace('MaskedVideos', 'Results')
	

	cmd_list_track = getTrackCommands(mask_dir_root, results_dir_root, tmp_dir_root, json_file, script_abs_path)
	
	#display commands to be executed
	if cmd_list_track: 
		for dd in cmd_list_track: print(' '.join(dd))

	#run all the commands
	runMultiCMD(cmd_list_track, max_num_process = max_num_process, refresh_time = 10)
    