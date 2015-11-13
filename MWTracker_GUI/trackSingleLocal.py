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
curr_script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_script_dir, 'MWTracker_dir.txt'), 'r') as f:
    MWTracker_dir = f.readline()
sys.path.append(MWTracker_dir)


from MWTracker.helperFunctions.getTrajectoriesWorkerL import getTrajectoriesWorkerL, getStartingPoint, checkpoint, checkpoint_label, constructNames


from MWTracker.featuresAnalysis.obtainFeatures_N import getWormFeaturesLab


def copyFilesLocal(files2copy):
	for files in files2copy:
		file_name, destination = files
		
		if not os.path.exists(file_name): continue
		
		assert(os.path.exists(destination))

		if os.path.abspath(os.path.dirname(file_name)) != os.path.abspath(destination):
			print('Copying %s to %s' % (file_name, destination))
			sys.stdout.flush()
			shutil.copy(file_name, destination)

if __name__ == '__main__':
#	try:
	masked_image_file = sys.argv[1]
	results_dir = sys.argv[2]
	tmp_masked_dir = sys.argv[3]
	tmp_results_dir = sys.argv[4]

	json_file = ''
	if len(sys.argv) > 5:
		json_file = sys.argv[5]

	#create temporary directories if they do not exists	
	if not os.path.exists(tmp_masked_dir): os.makedirs(tmp_masked_dir)
	if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)

	#get file names
	base_name, trajectories_file, skeletons_file, features_file, feat_ind_file = constructNames(masked_image_file, results_dir)
	tmp_mask_file = tmp_masked_dir + os.sep + base_name + '.hdf5'
	_, trajectories_tmp, skeletons_tmp, features_tmp, feat_ind_tmp = constructNames(tmp_mask_file, tmp_results_dir)
	
	#get starting directories
	final_start_point = getStartingPoint(masked_image_file, results_dir) #starting point calculated from the files in the final destination
	tmp_start_point = getStartingPoint(tmp_masked_dir, tmp_results_dir) #starting point calculated from the files in the temporal directory
	analysis_start_point = max(final_start_point, tmp_start_point) #starting point for the analysis
	
	print(tmp_start_point)
	if final_start_point == checkpoint['END']:
		#If the program has finished there is nothing to do here.
		print('The files from completed results analysis were found in %s. Remove them if you want to recalculated them again.' % results_dir)
		sys.stdout.flush()
		sys.exit(0)
	
	#find what files we need to copy from the final destination if the analysis is going to resume from a later point
	files2copy = []
	if tmp_start_point < final_start_point:
		#copy files from an incomplete analysis files.
		if final_start_point > checkpoint['TRAJ_CREATE']: #and final_start_point <= checkpoint['SKE_CREATE']:
			files2copy += [(trajectories_file, tmp_results_dir)]
		if final_start_point > checkpoint['SKE_CREATE']:
			files2copy += [(skeletons_file, tmp_results_dir)]
	
	if analysis_start_point < checkpoint['FEAT_CREATE']: 
		#we do not need the mask to calculate the features
		try:
			#check if there is already a finished/readable temporary mask file in current directory otherwise copy the 
			with h5py.File(tmp_mask_file, "r") as mask_fid:
				if mask_fid['/mask'].attrs['has_finished'] != 1:
					#go to the exception if the mask has any other flag
					raise
		except:
			with h5py.File(masked_image_file, "r") as mask_fid:
				#check if the video to mask conversion did indeed finished correctly
				assert mask_fid['/mask'].attrs['has_finished'] == 1

			files2copy += [(masked_image_file, tmp_results_dir)]

	
	print(base_name + ' Starting checkpoint: ' + checkpoint_label[analysis_start_point])
	sys.stdout.flush()

	#copy the necessary files (maybe we can create a daemon later)
	copyFilesLocal(files2copy)

	#start the analysis
	getTrajectoriesWorkerL(tmp_mask_file, tmp_results_dir, param_file = json_file, overwrite = False, start_point = analysis_start_point)

	
	files2copy = []
	#get files to copy
	print(base_name + " Copying result files into the final directory.")
	if final_start_point <= checkpoint['TRAJ_JOIN']:
		files2copy += [(trajectories_tmp, results_dir)]
	elif final_start_point <= checkpoint['SKE_ORIENT']:
		files2copy += [(skeletons_tmp, results_dir)]
	elif final_start_point <= checkpoint['FEAT_CREATE']:
		files2copy += [(features_tmp, results_dir)]
	elif final_start_point <= checkpoint['FEAT_IND']:
		files2copy += [(feat_ind_tmp, results_dir)]

	#print(files2copy)
	#copy files into the final directory
	copyFilesLocal(files2copy)
	
	print(base_name + " Deleting temporary files")
	#use the os.path.abspath really compare between paths
	if os.path.abspath(tmp_mask_file) != os.path.abspath(masked_image_file):
		if os.path.exists(tmp_mask_file): os.remove(tmp_mask_file)
	
	if os.path.abspath(tmp_results_dir) != os.path.abspath(results_dir):
		if os.path.exists(trajectories_tmp): os.remove(trajectories_tmp)
		if os.path.exists(skeletons_tmp): os.remove(skeletons_tmp)
		if os.path.exists(features_tmp): os.remove(features_tmp)
		if os.path.exists(feat_ind_tmp): os.remove(feat_ind_tmp)
	
	
	print(base_name + " Finished")
#except:
	#	print(base_name + " Error")
	#	raise


