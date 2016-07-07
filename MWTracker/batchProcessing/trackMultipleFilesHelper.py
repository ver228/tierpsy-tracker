# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import sys
import tables
import fnmatch

from .compressMultipleFilesHelper import checkVideoFiles, exploreDirs, getDstDir, isBadMask
from .trackSingleWorker import getStartingPoint, checkpoint, constructNames, \
isBadStageAligment, hasExpCntInfo

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

class checkTrackFiles(checkVideoFiles):
	def __init__(self, mask_dir_root, results_dir_root, tmp_dir_root = '', \
		is_single_worm = False, json_file = '', force_start_point='', end_point = '', \
		script_abs_path = './trackSingleLocal.py', \
		no_skel_filter = False, use_manual_join = False
		):
		
		#checkings before accepting the data
		mask_dir_root = os.path.abspath(mask_dir_root)
		results_dir_root = os.path.abspath(results_dir_root)
		script_abs_path = os.path.abspath(script_abs_path)
		
		if not os.path.exists(results_dir_root):
			os.makedirs(results_dir_root)

		assert os.path.exists(mask_dir_root)
		assert os.path.exists(script_abs_path)

		if json_file: 
			json_file = os.path.abspath(json_file)
			assert os.path.exists(json_file)
		
		if tmp_dir_root: 
			tmp_dir_root = os.path.abspath(tmp_dir_root)
			if not os.path.exists(tmp_dir_root):
				os.makedirs(tmp_dir_root)


		#save inputs as object properties
		self.mask_dir_root = mask_dir_root
		self.results_dir_root = results_dir_root
		self.tmp_dir_root = tmp_dir_root
		self.json_file = json_file
		self.script_abs_path = script_abs_path
		
		self.is_single_worm = is_single_worm
		self.no_skel_filter = no_skel_filter
		self.use_manual_join = use_manual_join
		
		self.force_start_point = force_start_point
		self.end_point_N = checkpoint['END'] if not end_point else checkpoint[end_point]
		self.end_point = end_point

		
		#search extensions that must be invalid to keep the analysis coherent
		self.invalid_ext = ['*_skeletons.hdf5', '*_trajectories.hdf5', '*_features.hdf5', '*_feat_ind.hdf5']
		
		#initialize lists to classify files.
		self.filtered_files_fields = ('SOURCE_GOOD', 'SOURCE_BAD', 'FINISHED_GOOD', 'FINISHED_BAD')
		self.filtered_files = {key : [] for key in self.filtered_files_fields}
	

	def generateIndCMD(self, masked_image_file):
		mask_dir, masked_file_name = os.path.split(masked_image_file)
		results_dir = getDstDir(mask_dir, self.mask_dir_root, self.results_dir_root)		
		if not os.path.exists(results_dir): os.makedirs(results_dir)
		
		if self.tmp_dir_root:
			tmp_mask_dir = getDstDir(mask_dir, self.mask_dir_root, os.path.join(self.tmp_dir_root, 'MaskedVideos'))
			tmp_results_dir = getDstDir(mask_dir, self.mask_dir_root, os.path.join(self.tmp_dir_root, 'Results'))

			if not os.path.exists(tmp_mask_dir): os.makedirs(tmp_mask_dir)
			if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)
		else:
			#if tmp_dir_root is empty just use the same directory (no tmp files)
			tmp_mask_dir, tmp_results_dir = '', ''
					
		#create a command line with the required arguments
		cmd = [sys.executable, self.script_abs_path, masked_image_file, results_dir]
		
		json_file = self.json_file
		end_point = self.end_point
		force_start_point = self.force_start_point
		#add the optional arguments if they are present
		for arg in ['tmp_mask_dir', 'tmp_results_dir', 'json_file', 'end_point', 'force_start_point']:
			tmp_val = eval(arg)
			if tmp_val: cmd += ['--' + arg, tmp_val]
		
		for arg in ['is_single_worm', 'no_skel_filter', 'use_manual_join']:
			if getattr(self, arg): cmd.append('--' + arg)

		return cmd
	
	def checkIndFile(self, masked_image_file):
		mask_dir, masked_file_name = os.path.split(masked_image_file)
		results_dir = getDstDir(mask_dir, self.mask_dir_root, self.results_dir_root)
		start_point = getStartingPoint(masked_image_file, results_dir)

		if self.is_single_worm:
			_, _, skeletons_file, _, _, _ = constructNames(masked_image_file, results_dir)

		#check that the file finished correctly and that there is no force_start_point specified
		if (start_point > self.end_point_N or start_point == checkpoint['END']) and not self.force_start_point:
			return 'FINISHED_GOOD', (masked_image_file, results_dir)
		elif self.is_single_worm and start_point == checkpoint['INT_PROFILE'] and isBadStageAligment(skeletons_file):
			return 'FINISHED_BAD', (masked_image_file, results_dir)
		elif self.is_single_worm and start_point == checkpoint['FEAT_CREATE'] and hasExpCntInfo(skeletons_file):
			return 'FINISHED_BAD', (masked_image_file, results_dir)
		
		elif not isBadMask(masked_image_file):
			return 'SOURCE_GOOD', masked_image_file
		else:
			return 'SOURCE_BAD', masked_image_file

	
	
    