# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:36:06 2015

@author: ajaver
"""

import os
import sys
import shutil
import h5py
sys.path.append('..')

from MWTracker.helperFunctions.getTrajectoriesWorkerL import getTrajectoriesWorkerL, getStartingPoint, checkpoint

if __name__ == '__main__':
	masked_image_file = sys.argv[1]
	results_dir = sys.argv[2]
	tmp_masked_dir = sys.argv[3]
	tmp_results_dir = sys.argv[4]

	base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
	trajectories_file = results_dir + base_name + '_trajectories.hdf5'
	skeletons_file = results_dir + base_name + '_skeletons.hdf5'
	features_file = results_dir + base_name + '_features.hdf5'


	start_point = getStartingPoint(trajectories_file, skeletons_file, features_file)
	if start_point == checkpoint['END']:
		print('Results files were completed. Remove %s, %s %s if you want to recalculated them again.' % (trajectories_file, skeletons_file, features_file))
	else:

		with h5py.File(masked_image_file, "r") as mask_fid:
			assert mask_fid['/mask'].attrs['has_finished'] == 1
		
		tmp_mask_file = tmp_masked_dir + os.sep + base_name + '.hdf5'


		trajectories_tmp = tmp_results_dir + base_name + '_trajectories.hdf5'
		skeletons_tmp = tmp_results_dir + base_name + '_skeletons.hdf5'
		features_tmp = tmp_results_dir + base_name + '_features.hdf5'

		if not os.path.exists(tmp_mask_file):
			if not os.path.exists(tmp_masked_dir):
				os.makedirs(tmp_masked_dir)

			print("Copying masked file %s into the temporary directory %s" % (masked_image_file, tmp_masked_dir))
			shutil.copy(masked_image_file, tmp_masked_dir)

		if not os.path.exists(tmp_results_dir):
				os.makedirs(tmp_results_dir)

		if start_point > checkpoint['TRAJ_CREATE'] and not os.path.exists(trajectories_tmp):
			shutil.copy(trajectories_file, tmp_results_dir)

		if start_point > checkpoint['SKE_CREATE'] and not os.path.exists(features_tmp):
			shutil.copy(skeletons_file, tmp_results_dir)

		if start_point > checkpoint['FEAT_CREATE'] and not os.path.exists(features_tmp):
			shutil.copy(features_file, tmp_results_dir)


		getTrajectoriesWorkerL(tmp_mask_file, tmp_results_dir, overwrite = False)

		print("Copying result files into the final directory %s" % results_dir)
		shutil.copy(trajectories_tmp, results_dir)
		shutil.copy(skeletons_tmp, results_dir)
		shutil.copy(features_tmp, results_dir)

		print("Removing temporary files.")
		os.remove(tmp_mask_file)
		os.remove(trajectories_tmp)
		os.remove(skeletons_tmp)
		os.remove(features_tmp)

		print("Finished")



