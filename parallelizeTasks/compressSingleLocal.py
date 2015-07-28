# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:38:05 2015

@author: ajaver
"""
import os
import sys
import h5py
import shutil
sys.path.append('..')

from MWTracker.helperFunctions.compressVideoWorkerL import compressVideoWorkerL


if __name__ == "__main__":
	video_file = sys.argv[1]
	mask_dir = sys.argv[2]
	tmp_mask_dir = sys.argv[3]

	base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
	masked_image_file = mask_dir + base_name + '.hdf5'
	masked_image_file_tmp = tmp_mask_dir + base_name + '.hdf5'
    
	try:
		with h5py.File(masked_image_file, "r") as mask_fid:
			if mask_fid['/mask'].attrs['has_finished'] == 1:
				has_finished = 1
	except:
		has_finished = 0  

	if not has_finished:	
		print("Creating temporal masked file.")
		compressVideoWorkerL(video_file, tmp_mask_dir)

		print("Copying temporal masked file into the final directory.")
		shutil.copy(masked_image_file_tmp, mask_dir)

		print("Finished to create masked file")
	else:
		print('File alread exists: %s' % masked_image_file)
		print('If you want to calculate the mask again delete the existing file.')