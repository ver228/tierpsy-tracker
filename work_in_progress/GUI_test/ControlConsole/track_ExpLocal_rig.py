# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys
from start_console import runMultiCMD, print_cmd_list
import argparse
import h5py
import tables

def isBadFile(masked_image_file):
	try:
		with tables.File(masked_image_file, 'r') as mask_fid:
			mask_node = mask_fid.get_node('/mask')
			if mask_node._v_attrs['has_finished'] < 1: 
				raise
			if mask_node.shape[0] == 0:
				raise
			if mask_node.shape[0] != len(mask_fid.get_node('/video_metadata')):
				#print(mask_node.shape[0], len(mask_fid.get_node('/video_metadata')))
				raise
			return 0
	except:
		return 1


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

				subdir_path = dpath.replace(mask_dir_root, '')
				if subdir_path and subdir_path[0] == os.sep: 
					subdir_path = subdir_path[1:] if len(subdir_path) >1 else ''
				
				#get results directory
				results_dir = os.path.abspath(os.path.join(results_dir_root, subdir_path))
				if not os.path.exists(results_dir): os.makedirs(results_dir)

				#get temporary directories
				if tmp_dir_root:
					tmp_mask_dir = os.path.abspath(os.path.join(tmp_dir_root, 'MaskedVideos', subdir_path))
					if not os.path.exists(tmp_mask_dir): os.makedirs(tmp_mask_dir)

					tmp_results_dir = os.path.abspath(os.path.join(tmp_dir_root, 'Results', subdir_path))					
					if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)
					
				else:
					#if tmp_dir_root is empty just use the same directory (no tmp files)
					tmp_mask_dir, tmp_results_dir = '', ''
					
				
				#create a command line with the required arguments
				cmd = ['python3', script_abs_path, masked_image_file, results_dir]
				
				#add the optional arguments if they are present
				for arg in ['tmp_mask_dir', 'tmp_results_dir', 'json_file', 'end_point']:
					if eval(arg): cmd += ['--' + arg, eval(arg)]

				if is_single_worm: cmd.append('--is_single_worm')
				
				cmd_list_track.append(cmd)

	return cmd_list_track
	
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

def main(mask_dir_root, tmp_dir_root, json_file, script_abs_path, max_num_process, refresh_time, end_point, is_single_worm):
	assert os.path.exists(mask_dir_root)
	mask_dir_root = os.path.abspath(mask_dir_root)
	if json_file: assert(os.path.exists(json_file))

	results_dir_root = getResultsDir(mask_dir_root)
	
	cmd_list_track = getTrackCommands(mask_dir_root = mask_dir_root, results_dir_root = results_dir_root, 
		tmp_dir_root = tmp_dir_root, json_file = json_file, 
		script_abs_path = script_abs_path, end_point = end_point, is_single_worm = is_single_worm)
	
	#cmd_list_track = cmd_list_track[0:1]
	
	#display commands to be executed
	print_cmd_list(cmd_list_track)

	#run all the commands
	runMultiCMD(cmd_list_track, max_num_process = max_num_process, refresh_time = refresh_time)


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

	args = parser.parse_args()

	main(**vars(args))