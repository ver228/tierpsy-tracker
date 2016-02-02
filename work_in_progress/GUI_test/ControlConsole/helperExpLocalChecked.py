# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import stat
import glob
import sys
import tables
from start_console import runMultiCMD, print_cmd_list
import fnmatch
import argparse

sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
from MWTracker.helperFunctions.getTrajectoriesWorkerL import getStartingPoint, checkpoint
from MWTracker.helperFunctions.timeCounterStr import timeCounterStr
from MWTracker.compressVideos.getAdditionalData import getAdditionalFiles

def exploreDirs(root_dir, pattern_include = '*', pattern_exclude = ''):
	root_dir = os.path.abspath(root_dir)
	assert os.path.exists(root_dir)
	
	#if there is only a string (only one pattern) let's make it a list to be able to reuse the code
	if not isinstance(pattern_include, (list, tuple)): pattern_include = [pattern_include]
	if not isinstance(pattern_exclude, (list, tuple)): pattern_exclude = [pattern_exclude]

	valid_files = []
	for dpath, dnames, fnames in os.walk(root_dir):
		for fname in fnames:
			good_patterns = any(fnmatch.fnmatch(fname, dd) for dd in pattern_include)
			bad_patterns = any(fnmatch.fnmatch(fname, dd) for dd in pattern_exclude)
			if good_patterns and not bad_patterns:
				fullfilename = os.path.abspath(os.path.join(dpath, fname))
				assert os.path.exists(fullfilename)
				valid_files.append(fullfilename)

	return valid_files

def getDstDir(source_dir, source_root_dir, dst_root_dir):
	'''
	Generate the destination dir path keeping the same structure as the source directory
	'''

	subdir_path = source_dir.replace(source_root_dir, '')
	
	#consider the case the subdirectory is only a directory separation character
	if subdir_path and subdir_path[0] == os.sep: 
		subdir_path = subdir_path[1:] if len(subdir_path) >1 else ''
	dst_dir = os.path.join(dst_root_dir, subdir_path)

	return dst_dir

class checkVideoFiles:
	def __init__(self, video_dir_root, mask_dir_root, \
		tmp_dir_root = '', \
		is_single_worm = False, json_file = '', \
		not_auto_label = False, use_manual_join = False,\
		script_abs_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/compressSingleLocal.py'):
		
		#checkings before accepting the data
		video_dir_root = os.path.abspath(video_dir_root)
		assert os.path.exists(video_dir_root)
		
		if json_file: assert os.path.exists(json_file)
		assert os.path.exists(script_abs_path)

		self.video_dir_root = video_dir_root
		self.json_file = json_file
		self.script_abs_path = script_abs_path
		self.is_single_worm = is_single_worm
		self.not_auto_label = not_auto_label
		self.use_manual_join = use_manual_join
		
		#Let's look look for a MaskedVideos subdirectory, otherwise we add it at the end of the root dir
		self.mask_dir_root = self.checkMaskPrefix(mask_dir_root)
		#If tmp_dir_root is empty let's use the mask_dir_root as our source
		self.tmp_dir_root = self.checkMaskPrefix(tmp_dir_root) if tmp_dir_root else self.mask_dir_root

		self.filtered_files_fields = ('SOURCE_GOOD', 'SOURCE_BAD', 'FINISHED_GOOD', 'FINISHED_BAD')
		self.filtered_files = {key : [] for key in self.filtered_files_fields}
		
	def checkIndFile(self, video_file):
		
		video_dir, video_file_name = os.path.split(video_file)
		mask_dir = getDstDir(video_dir, self.video_dir_root, self.mask_dir_root)
		masked_image_file = self.getMaskName(video_file, mask_dir)
		
		if os.path.exists(masked_image_file):
			if not self.isBadMask(masked_image_file):
				return 'FINISHED_GOOD' , (video_file, masked_image_file)
			else:
				return 'FINISHED_BAD', (video_file, masked_image_file)
		
		else:
			if self.is_single_worm and not self.hasAdditionalFiles(video_file):
				return 'SOURCE_BAD', video_file
		
		return 'SOURCE_GOOD', video_file


	def filterFiles(self, valid_files):
		#intialize filtered files lists
		self.filtered_files = {key : [] for key in self.filtered_files_fields}
		
		progress_timer = timeCounterStr('');
		for ii, video_file in enumerate(valid_files):
			label, proccesed_file = self.checkIndFile(video_file)
			assert label in self.filtered_files_fields

			self.filtered_files[label].append(proccesed_file)

			if (ii % 10) == 0:
				print('Checking file %i of %i. Total time: %s' % (ii+1, len(valid_files), progress_timer.getTimeStr()))
		
		print('Finished to check files. Total time elapsed %s' % progress_timer.getTimeStr())


	@staticmethod
	def checkMaskPrefix(fdir):
		#check if the root dir has a subfolder MaskedVideos otherwise add it to the end
		N = sum('MaskedVideos' == part for part in fdir.split(os.sep))

		if N > 1: 
			print('ERROR: Only one subdirectory is allowed to be named "MaskedVideos"')
			raise

		if N == 0:
			fdir =  os.path.join(fdir, 'MaskedVideos')
		return os.path.abspath(fdir)

	def getCMDlist(self):
		good_video_files = self.filtered_files['SOURCE_GOOD']
		
		cmd_list = []
		for video_file in good_video_files:
			cmd_list.append(self.generateIndCMD(video_file))
		return cmd_list

	def generateIndCMD(self, video_file):
		video_dir, video_file_name = os.path.split(video_file)
		
		mask_dir = getDstDir(video_dir, self.video_dir_root, self.mask_dir_root)
		tmp_mask_dir = getDstDir(video_dir, self.video_dir_root, self.tmp_dir_root)

		if not os.path.exists(mask_dir): os.makedirs(mask_dir)
		if not os.path.exists(tmp_mask_dir): os.makedirs(tmp_mask_dir)
					
		#create a command line with the required arguments
		cmd = ['python3', self.script_abs_path, video_file, mask_dir]
		
		json_file = self.json_file
		#add the optional arguments if they are present
		for arg in ['tmp_mask_dir', 'json_file']:
			tmp_val = eval(arg)
			if tmp_val: cmd += ['--' + arg, tmp_val]
		
		if self.is_single_worm: cmd.append('--is_single_worm')

		return cmd

	def cleanPrevious(self):
		''' Delete any bad finished files.'''
		for video_file, masked_image_file in self.filtered_files['FINISHED_BAD']:
			assert masked_image_file[-5:] == '.hdf5'
			
			#delete wrong file from previous analysis	
			os.chflags(masked_image_file, not stat.UF_IMMUTABLE)
			os.remove(masked_image_file)

			#add the video to files to be processed
			label, proccesed_file = self.checkIndFile(video_file)
			assert label in self.filtered_files_fields
			self.filtered_files[label].append(proccesed_file)
			

		self.filtered_files['FINISHED_BAD'] = []

	@staticmethod
	def getMaskName(video_file, mask_dir):
		base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
		masked_image_file = os.path.join(mask_dir, base_name + '.hdf5')
		return masked_image_file

	@staticmethod
	def hasAdditionalFiles(video_file):
		try:
			#this function will throw and error if the .info.xml or .log.csv are not found
			getAdditionalFiles(video_file)
			return True
		except FileNotFoundError:
			return False
	@staticmethod
	def isBadMask(masked_image_file):
		try:
			with tables.File(masked_image_file, 'r') as mask_fid:
				mask_node = mask_fid.get_node('/mask')
				if mask_node._v_attrs['has_finished'] < 1: 
					raise
				if mask_node.shape[0] == 0:
					raise
				
				#length can vary. I need to use getTimestamp instead, but it is too cumbersome at this moment
				#if '/video_metadata' in mask_fid and \
				#mask_node.shape[0] != len(mask_fid.get_node('/video_metadata')):
				#	raise
				
				return 0
		except:
			return 1

class checkTrackFiles(checkVideoFiles):
	def __init__(self, mask_dir_root, tmp_dir_root = '', \
		is_single_worm = False, json_file = '', end_point = '', \
		script_abs_path = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/trackSingleLocal.py', \
		not_auto_label = False, use_manual_join = False
		):
		
		#checkings before accepting the data
		mask_dir_root = os.path.abspath(mask_dir_root)
		assert os.path.exists(mask_dir_root)
		
		if json_file: assert os.path.exists(json_file)
		assert os.path.exists(script_abs_path)

		self.mask_dir_root = mask_dir_root
		self.tmp_dir_root = tmp_dir_root

		self.json_file = json_file
		self.script_abs_path = script_abs_path
		self.is_single_worm = is_single_worm
		self.not_auto_label = not_auto_label
		self.use_manual_join = use_manual_join
		
		self.end_point_N = checkpoint['END'] if not end_point else checkpoint[end_point]
		self.end_point = end_point

		#get results directory
		self.results_dir_root = self.getResultsDir(mask_dir_root)

		#search extensions that must be invalid to keep the analysis coherent
		self.invalid_ext = ['*_skeletons.hdf5', '*_trajectories.hdf5', '*_features.hdf5', '*_feat_ind.hdf5']
		
		#initialize lists to classify files.
		self.filtered_files_fields = ('SOURCE_GOOD', 'SOURCE_BAD', 'FINISHED_GOOD', 'FINISHED_BAD')
		self.filtered_files = {key : [] for key in self.filtered_files_fields}
	
	@staticmethod
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
		cmd = ['python3', self.script_abs_path, masked_image_file, results_dir]
		
		json_file = self.json_file
		end_point = self.end_point
		#add the optional arguments if they are present
		for arg in ['tmp_mask_dir', 'tmp_results_dir', 'json_file', 'end_point']:
			tmp_val = eval(arg)
			if tmp_val: cmd += ['--' + arg, tmp_val]
		

		for arg in ['is_single_worm', 'not_auto_label', 'use_manual_join']:
			if getattr(self, arg): cmd.append('--' + arg)

		return cmd
	
	def checkIndFile(self, masked_image_file):
		mask_dir, masked_file_name = os.path.split(masked_image_file)
		results_dir = getDstDir(mask_dir, self.mask_dir_root, self.results_dir_root)
		start_point = getStartingPoint(masked_image_file, results_dir)

		if start_point >= self.end_point_N:
			return 'FINISHED_GOOD' , (masked_image_file, results_dir)
		
		elif self.isBadMask(masked_image_file):
			return 'SOURCE_BAD', masked_image_file

		return 'SOURCE_GOOD', masked_image_file
	
	
	
    