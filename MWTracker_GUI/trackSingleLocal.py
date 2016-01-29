# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:36:06 2015

@author: ajaver
"""

import os
import sys
import shutil
import h5py
import pandas as pd
import argparse
curr_script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_script_dir, 'MWTracker_dir.txt'), 'r') as f:
    MWTracker_dir = f.readline()
sys.path.append(MWTracker_dir)


from MWTracker.helperFunctions.getTrajectoriesWorkerL import getTrajectoriesWorkerL, getStartingPoint, checkpoint, checkpoint_label, constructNames
from MWTracker.helperFunctions.miscFun import print_flush

from MWTracker.featuresAnalysis.obtainFeatures_N import getWormFeaturesLab



def copyFilesLocal(files2copy):
	#copy the necessary files (maybe we can create a daemon later)
		for files in files2copy:
			file_name, destination = files
			
			#if not os.path.exists(file_name): continue
			
			assert(os.path.exists(destination))

			if os.path.abspath(os.path.dirname(file_name)) != os.path.abspath(destination):
				print('Copying %s to %s' % (file_name, destination))
				sys.stdout.flush()
				shutil.copy(file_name, destination)

class trackLocal:
	def __init__(self, masked_image_file, results_dir, tmp_mask_dir='', tmp_results_dir='', 
		json_file ='', end_point = 'END', is_single_worm='', 
		not_auto_label = False, use_manual_join = False):
		
		self.masked_image_file = masked_image_file
		self.results_dir = results_dir
		
		self.assign_tmp_dir(tmp_mask_dir, tmp_results_dir)
		self.end_point = checkpoint[end_point]
		
		self.json_file = json_file 
		self.is_single_worm = is_single_worm
		self.use_auto_label = not not_auto_label
		self.use_manual_join = use_manual_join

		assert(os.path.exists(masked_image_file))

		self.getFileNames()
		self.getStartPoints()
		self.copyFilesFromFinal()

		#start the analysis
		getTrajectoriesWorkerL(self.tmp_mask_file, self.tmp_results_dir, param_file = self.json_file, 
			start_point = self.analysis_start_point, end_point = self.end_point, is_single_worm = self.is_single_worm,
			use_auto_label = self.use_auto_label, use_manual_join = self.use_manual_join)

		self.copyFilesToFinal()
		self.cleanTmpFiles()

		print_flush(self.base_name + " Finished ")#,  features_tmp, features_file)

	def assign_tmp_dir(self, tmp_mask_dir, tmp_results_dir):
		if tmp_mask_dir:
			self.tmp_mask_dir = tmp_mask_dir
			if not tmp_results_dir:
				#deduce tmp_results_dir from tmp_mask_dir
				self.tmp_results_dir = self.tmp_mask_dir.replacereplace('MaskedVideos', 'Results')

		if tmp_results_dir:
			self.tmp_results_dir = tmp_results_dir
			if not tmp_mask_dir:
				#deduce tmp_mask_dir from tmp_results_dir
				self.tmp_mask_dir = self.tmp_results_dir.replacereplace('Results', 'MaskedVideos')

		if not tmp_results_dir and not tmp_mask_dir:
			#use final destination and tmp_dir
			self.tmp_results_dir = self.results_dir
			self.tmp_mask_dir = os.path.split(self.masked_image_file)[0]

	def getFileNames(self):
		self.base_name, self.trajectories_file, self.skeletons_file, self.features_file, self.feat_ind_file = constructNames(self.masked_image_file, self.results_dir)
		self.tmp_mask_file = self.tmp_mask_dir + os.sep + self.base_name + '.hdf5'
		_, self.trajectories_tmp, self.skeletons_tmp, self.features_tmp, self.feat_ind_tmp = constructNames(self.tmp_mask_file, self.tmp_results_dir)

		#create temporary directories if they do not exists	
		if not os.path.exists(self.tmp_mask_dir): os.makedirs(self.tmp_mask_dir)
		if not os.path.exists(self.tmp_results_dir): os.makedirs(self.tmp_results_dir)


		print(self.trajectories_tmp, self.skeletons_tmp, self.features_tmp, self.feat_ind_tmp)

	def getStartPoints(self):
		#get starting directories
		self.final_start_point = getStartingPoint(self.masked_image_file, self.results_dir) #starting point calculated from the files in the final destination
		self.tmp_start_point = getStartingPoint(self.tmp_mask_file, self.tmp_results_dir) #starting point calculated from the files in the temporal directory
		self.analysis_start_point = max(self.final_start_point, self.tmp_start_point) #starting point for the analysis
		
		if self.analysis_start_point <= self.end_point:
			print_flush(self.base_name + ' Starting checkpoint: ' + checkpoint_label[self.analysis_start_point])
	

	def copyFilesFromFinal(self):
		if self.final_start_point == checkpoint['END']:
			#If the program has finished there is nothing to do here.
			print_flush('%s The files from completed results analysis were found in %s. Remove them if you want to recalculated them again.' % (self.base_name, self.results_dir))
			sys.exit(0)
		
		#find what files we need to copy from the final destination if the analysis is going to resume from a later point
		files2copy = []
		
		if self.analysis_start_point <= checkpoint['SKE_CREATE']: 
			#we do not need the mask to calculate the features
			try:
				#check if there is already a finished/readable temporary mask file in current directory otherwise copy the 
				with h5py.File(self.tmp_mask_file, "r") as mask_fid:
					if mask_fid['/mask'].attrs['has_finished'] < 1:
						#go to the exception if the mask has any other flag
						raise
			except:
				with h5py.File(self.masked_image_file, "r") as mask_fid:
					#check if the video to mask conversion did indeed finished correctly
					assert mask_fid['/mask'].attrs['has_finished'] >= 1

				files2copy += [(self.masked_image_file, self.tmp_mask_dir)]

		if self.tmp_start_point < self.final_start_point:
			#copy files from an incomplete analysis files.
			if self.final_start_point > checkpoint['TRAJ_CREATE']: #and final_start_point <= checkpoint['SKE_CREATE']:
				files2copy += [(self.trajectories_file, self.tmp_results_dir)]
			if self.final_start_point > checkpoint['SKE_CREATE']:
				files2copy += [(self.skeletons_file, self.tmp_results_dir)]
		
		copyFilesLocal(files2copy)
	
	def copyFilesToFinal(self):
		
		def isFile2Copy(str_point):
		    valid_range = sorted([checkpoint[x] for x in checkpoint if str_point in x])
		    range_intersect = set(valid_range) & set(range(self.final_start_point, self.end_point+1))
		    return len(set(range_intersect))>0

		files2copy = []
		#get files to copy
		print(self.base_name + " Copying result files into the final directory.")
		
		if isFile2Copy('TRAJ'):
			files2copy += [(self.trajectories_tmp, self.results_dir)]
		
		if isFile2Copy('SKE'):
			files2copy += [(self.skeletons_tmp, self.results_dir)]

		if isFile2Copy('FEAT_CREATE'):
			files2copy += [(self.features_tmp, self.results_dir)]
		
		if isFile2Copy('FEAT_MANUAL_CREATE') and self.use_manual_join:
			files2copy += [(self.feat_ind_tmp, self.results_dir)]

		copyFilesLocal(files2copy)

	def cleanTmpFiles(self):
		print_flush(self.base_name + " Deleting temporary files")
		#use the os.path.abspath really compare between paths
		if os.path.abspath(self.tmp_mask_file) != os.path.abspath(self.masked_image_file):
			if os.path.exists(self.tmp_mask_file): os.remove(self.tmp_mask_file)
		
		#this files must exists at this point in the program. Let's check it before deleting anything.
		
		assert os.path.exists(self.trajectories_file)
		
		if self.end_point >= checkpoint['SKE_CREATE']:
			assert os.path.exists(self.skeletons_file)

		if self.end_point >= checkpoint['FEAT_CREATE']:
			assert os.path.exists(self.features_file)

		if self.end_point >= checkpoint['FEAT_MANUAL_CREATE'] and self.use_manual_join:
			assert os.path.exists(self.feat_ind_file)


		#delete the results temporary files
		if os.path.abspath(self.tmp_results_dir) != os.path.abspath(self.results_dir):
			if os.path.exists(self.trajectories_tmp): os.remove(self.trajectories_tmp)
			if os.path.exists(self.skeletons_tmp): os.remove(self.skeletons_tmp)
			if os.path.exists(self.features_tmp): os.remove(self.features_tmp)
			if os.path.exists(self.feat_ind_tmp): os.remove(self.feat_ind_tmp)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Track the worm's hdf5 files in the local drive.")
	parser.add_argument('masked_image_file', help='Fullpath of the .hdf5 with the masked worm videos')
	parser.add_argument('results_dir', help='Final directory where the tracking results are going to be stored')
	parser.add_argument('--tmp_mask_dir', default='', help='Temporary directory where the masked file is stored')
	parser.add_argument('--tmp_results_dir', default='', help='temporary directory where the results are stored')
	parser.add_argument('--json_file', default='', help='File (.json) containing the tracking parameters.')
	parser.add_argument('--end_point', default='END', choices = checkpoint, help='End point of the analysis.')
	parser.add_argument('--is_single_worm', action='store_true', help = 'This flag indicates if the video corresponds to the single worm case.')
	parser.add_argument('--use_manual_join', action='store_true', help = '.')
	parser.add_argument('--not_auto_label', action='store_true', help = '.')
	
	args = parser.parse_args()
	
	trackLocal(**vars(args))

	



