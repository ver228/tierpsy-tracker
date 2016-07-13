# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:38:05 2015

@author: ajaver
"""
import os, stat, sys
import h5py
import shutil
import argparse
import subprocess
import time, datetime

from .compressSingleWorker import compressSingleWorker
import argparse

from MWTracker.helperFunctions.miscFun import print_flush

try:
	#use this directory if it is a one-file produced by pyinstaller
	SCRIPT_COMPRESS_WORKER = [os.path.join(sys._MEIPASS, 'compressSingleWorker')]
except Exception:
	SCRIPT_COMPRESS_WORKER = [sys.executable, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'compressSingleWorker.py')]

class compressLocal:
	def __init__(self, video_file, mask_dir, tmp_mask_dir='', json_file='', \
		is_copy_video = False, is_single_worm = False, cmd_original=''):

		assert os.path.exists(video_file)
		if not tmp_mask_dir: 
			self.tmp_mask_dir = mask_dir
		if json_file:
			assert os.path.exists(json_file)

		self.video_file = video_file
		self.mask_dir = mask_dir
		self.tmp_mask_dir = tmp_mask_dir
		self.json_file = json_file
		self.is_single_worm = is_single_worm
		self.cmd_original = cmd_original
		self.is_copy_video = is_copy_video

	def exec_all(self):
		try:
			self.start()
			self.main_code()
			self.clean()
		except:
			raise
			print_flush(base_name + ' Error')
	
	def start(self):
		self.start_time = time.time()
		
		if not os.path.exists(self.tmp_mask_dir): os.makedirs(self.tmp_mask_dir)
		
		self.get_file_names()
		self.get_start_point()

		#copy the video file locally if requiered. See getFileNames
		if self.final_has_finished == 0 and self.tmp_video_file != self.video_file:
			print_flush(self.base_name + ' Copying video file to local temporary directory.')
			shutil.copy(self.video_file, self.tmp_video_file)
			if self.is_single_worm:
				try:
					dd = self.video_file.rpartition('.')[0]
					shutil.copy(dd + '.log.csv', self.tmp_mask_dir)
					shutil.copy(dd + '.info.xml', self.tmp_mask_dir)
				except FileNotFoundError:
					try:
						dd = self.video_file.rpartition('.')[0]
						dd, base_name = os.path.split(dd)
						dd = os.path.join(dd, '.data', base_name)
						shutil.copy(dd + '.log.csv', self.tmp_mask_dir)
						shutil.copy(dd + '.info.xml', self.tmp_mask_dir)
					except FileNotFoundError:
						pass

			
		#this might be more logically group in main_code, but this operation can and should be do remotely if required
		if self.final_has_finished  == 1:
			#The file finished to be processed but the additional data was not stored. We can do this remotely. 
			if os.name != 'nt': os.chflags(self.masked_image_file, not stat.UF_IMMUTABLE)
			compressSingleWorker(self.video_file, self.mask_dir, self.json_file, self.is_single_worm, self.cmd_original)
			if os.name != 'nt': os.chflags(self.masked_image_file, stat.UF_IMMUTABLE)

		if self.final_has_finished  == 2:
			print_flush('File alread exists: %s. If you want to calculate the mask again delete the existing file.' % self.masked_image_file)
	
		#parameters to calculate the mask, this will return a list with necessary commands in case I want to use it as a subprocess
		if self.is_calculate_mask:
			self.main_input_params = [self.tmp_video_file, self.tmp_mask_dir, self.json_file, self.is_single_worm, self.cmd_original]
		else:
			self.main_input_params = []
		
		return self.create_script()
		

	def create_script(self):
		
		cmd = SCRIPT_COMPRESS_WORKER + self.main_input_params
		#replace bool values by a letter, otherwise one cannot parse them to the command line
		cmd = [x if not isinstance(x, bool) else 'T' if x else '' for x in cmd]
		return cmd

	def get_file_names(self):
		self.video_file = os.path.abspath(self.video_file)
		self.base_name = self.video_file.rpartition('.')[0].rpartition(os.sep)[-1]
		
		self.masked_image_file = os.path.join(self.mask_dir, self.base_name + '.hdf5')
		self.masked_image_file = os.path.abspath(self.masked_image_file)
		assert self.masked_image_file != self.video_file
		
		self.tmp_mask_file = os.path.join(self.tmp_mask_dir, self.base_name + '.hdf5')
		self.tmp_mask_file = os.path.abspath(self.tmp_mask_file)
		
		if self.is_copy_video:
			self.tmp_video_file = os.path.join(self.tmp_mask_dir, os.path.split(self.video_file)[1])
			self.tmp_video_file = os.path.abspath(self.tmp_video_file)
		else:
			self.tmp_video_file = self.video_file

	def get_start_point(self):
		self.final_has_finished = 0
		self.is_calculate_mask = False
		#check if the 
		try:
			with h5py.File(self.masked_image_file, "r") as mask_fid:
				self.final_has_finished = mask_fid['/mask'].attrs['has_finished']
		except (OSError, KeyError):
			self.final_has_finished = 0


		if self.final_has_finished == 0:
			#check if a finished temporal mask exists. The processes was interrupted during copying.
			try:
				with h5py.File(self.tmp_mask_file, "r") as mask_fid:
					MAX_CONTROL_FLAG = 2
					self.is_calculate_mask =  (mask_fid['/mask'].attrs['has_finished'] < MAX_CONTROL_FLAG)
			except (OSError, KeyError):
					self.is_calculate_mask = True
	

	def main_code(self):
		if self.is_calculate_mask:
			#start to calculate the mask from raw video
			print_flush(self.base_name + " Creating temporal masked file.")
			compressSingleWorker(*self.main_input_params)

	def clean(self):
		if self.final_has_finished == 0:
			print_flush(self.base_name + ' Moving files to final destination and removing temporary files.')
			if os.path.abspath(self.tmp_video_file) != os.path.abspath(self.video_file):
				#print_flush(self.base_name + ' Removing video file from local temporary directory.')
				assert os.path.abspath(self.tmp_video_file) != os.path.abspath(self.video_file)
				os.remove(self.tmp_video_file)
				assert os.path.exists(self.video_file)

				if self.is_single_worm:
					dd = self.tmp_video_file.rpartition('.')[0]
					os.remove(dd + '.log.csv')
					os.remove(dd + '.info.xml')


			#assert the local file did finished
			with h5py.File(self.tmp_mask_file, "r") as mask_fid:
				assert mask_fid['/mask'].attrs['has_finished'] > 0

			if os.path.abspath(self.tmp_mask_file) != os.path.abspath(self.masked_image_file):
				#it is very important to use os.path.abspath() otherwise there could be some 
				#confunsion in the same file name
				#print_flush(self.base_name + " Copying temporal masked file into the final directory.")
				shutil.copy(self.tmp_mask_file, self.masked_image_file)
			
				#print_flush(self.base_name + " Removing temporary files.")
				assert os.path.exists(self.masked_image_file)
				os.remove(self.tmp_mask_file)

			if os.name != 'nt':
				#Change the permissions so everybody can read/write. 
				#Otherwise only the owner would be able to change the ummutable flag.
				os.chmod(self.masked_image_file, stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH|stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH) 
			
				#Protect file from deletion.
				os.chflags(self.masked_image_file, stat.UF_IMMUTABLE)
			#print_flush(self.base_name + " Finished to create masked file")


		time_str = str(datetime.timedelta(seconds=round(time.time()-self.start_time)))
		print_flush('%s Finished. Total time = %s' % (self.base_name, time_str))
		
class compressLocal_parser(compressLocal):
	def __init__(self, sys_argv=''):
		if not sys_argv:
			sys_argv = sys.argv

		self.parser = argparse.ArgumentParser(description="Compress worm videos into masked hdf5 files processing data first into the local drive.")
		self.parser.add_argument('video_file', help='Original video.')
		self.parser.add_argument('mask_dir', help='Final directory where the compressed files are going to be stored')
		self.parser.add_argument('--tmp_mask_dir', default='', help='Temporary directory where the masked file is stored')
		self.parser.add_argument('--json_file', default='', help='File (.json) containing the compressed parameters.')
		self.parser.add_argument('--is_single_worm', action='store_true', help = 'Indicates if the video corresponds to the single worm case.')
		self.parser.add_argument('--is_copy_video', action='store_true', help = 'The video file would be copied to the temporary directory.')
		
		args = self.parser.parse_args(sys_argv[1:])

		super(compressLocal_parser, self).__init__(**vars(args), cmd_original = subprocess.list2cmdline(sys_argv))

if __name__ == "__main__":
	d = compressLocal_parser(sys.argv)
	d.exec_all()
