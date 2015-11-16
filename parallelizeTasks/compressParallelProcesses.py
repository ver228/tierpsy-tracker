# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys


sys.path.append('..')
from MWTracker.helperFunctions.parallelProcHelper import runMultiSubproc


masked_movies_root =  '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/'
max_num_process = 6

dir_main = sys.argv[1]
#dir_main = '/Volumes/Mrc-pc/20150522_1940/'

if dir_main[-1] == os.sep:
    dir_main = dir_main[:-1]
    
subdir_base = os.path.split(dir_main)[-1]    

movie_files = glob.glob(dir_main + os.sep + '*.mjpg') 
masked_movies_dir = masked_movies_root + subdir_base + os.sep

if not os.path.exists(masked_movies_dir):
    os.mkdir(masked_movies_dir)


cmd_list_compress = []
for video_file in movie_files:
    cmd_list_compress += [' '.join(['python3 compressSingleFile.py', video_file, masked_movies_dir])]
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    
runMultiSubproc(cmd_list_compress, max_num_process = max_num_process)
