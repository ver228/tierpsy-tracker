# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:38:05 2015

@author: ajaver
"""
import os, stat
import sys
import h5py
import shutil
import argparse

curr_script_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(curr_script_dir, 'MWTracker_dir.txt'), 'r') as f:
    MWTracker_dir = f.readline()
sys.path.append(MWTracker_dir)

from MWTracker.helperFunctions.compressVideoWorkerL import compressVideoWorkerL
import argparse

def print_flush(fstr):
    print(fstr)
    sys.stdout.flush()

def main(video_file, mask_dir, tmp_mask_dir='', json_file='', is_single_worm = False): 
	try:
		if not tmp_mask_dir: tmp_mask_dir = mask_dir
		if mask_dir[-1] != os.sep: mask_dir += os.sep 
		if tmp_mask_dir[-1] != os.sep: tmp_mask_dir += os.sep 

		base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
		masked_image_file = mask_dir + base_name + '.hdf5'
		tmp_mask_file = tmp_mask_dir + base_name + '.hdf5'

		try:
			with h5py.File(masked_image_file, "r") as mask_fid:
				has_finished = mask_fid['/mask'].attrs['has_finished']
		except:
			has_finished = 0

			
		if has_finished == 0:
			#check if a finished temporal mask exists. The processes was interrupted during copying.
			try:
				with h5py.File(tmp_mask_file, "r") as mask_fid:
					MAX_CONTROL_FLAG = 2
					is_calculate_mask =  (mask_fid['/mask'].attrs['has_finished'] < MAX_CONTROL_FLAG)
			except (OSError, KeyError):
					is_calculate_mask = True
				

			if is_calculate_mask:
				#start to calculate the mask from raw video
				print_flush(base_name + " Creating temporal masked file.")
				compressVideoWorkerL(video_file, tmp_mask_dir, json_file, is_single_worm)
				
			if os.path.abspath(tmp_mask_file) != os.path.abspath(masked_image_file):
				#it is very important to use os.path.abspath() otherwise there could be some 
				#confunsion in the same file name
				print_flush(base_name + " Copying temporal masked file into the final directory.")
				shutil.copy(tmp_mask_file, masked_image_file)
			
				print_flush(base_name + " Removing temporary files.")
				os.remove(tmp_mask_file)

			#Change the permissions so everybody can read/write. 
			#Otherwise only the owner would be able to change the ummutable flag.
			os.chmod(masked_image_file, stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH|stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH) 
			
			#Protect file from deletion.
			os.chflags(masked_image_file, stat.UF_IMMUTABLE)
			print_flush(base_name + " Finished to create masked file")
		
		if has_finished == 1:
			os.chflags(masked_image_file, not stat.UF_IMMUTABLE)
			
			#The file finished to be processed but the additional data was not stored. We can do this remotely. 
			compressVideoWorkerL(video_file, mask_dir, json_file, is_single_worm)

			os.chflags(masked_image_file, stat.UF_IMMUTABLE)

		if has_finished == 2:
			print_flush('File alread exists: %s. If you want to calculate the mask again delete the existing file.' % masked_image_file)
	except:
		raise
		print_flush(base_name + ' Error')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Compress worm videos into masked hdf5 files processing data first into the local drive.")
	parser.add_argument('video_file', help='Original video.')
	parser.add_argument('mask_dir', help='Final directory where the compressed files are going to be stored')
	parser.add_argument('--tmp_mask_dir', default='', help='Temporary directory where the masked file is stored')
	parser.add_argument('--json_file', default='', help='File (.json) containing the compressed parameters.')
	parser.add_argument('--is_single_worm', action='store_true', help = 'This flag indicates if the video corresponds to the single worm case.')
	args = parser.parse_args()

	main(**vars(args))

