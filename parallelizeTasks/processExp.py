# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys


sys.path.append('../helperFunctions/')
from parallelProcHelper import runMultiSubproc



masked_movies_root =  '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/'
results_root = '/Volumes/behavgenom$/GeckoVideo/Results/'
max_num_process = 6

dir_main = sys.argv[1]
#dir_main = '/Volumes/Mrc-pc/20150522_1940/'

if dir_main[-1] == os.sep:
    dir_main = dir_main[:-1]
    
subdir_base = os.path.split(dir_main)[-1]    

movie_files = glob.glob(dir_main + os.sep + '*.mjpg') 
masked_movies_dir = masked_movies_root + subdir_base + os.sep
results_dir = results_root + subdir_base + os.sep


cmd_list_compress = []
cmd_list_track = []
for video_file in movie_files:
    cmd_list_compress += [' '.join(['python3 compressSingleFile.py', video_file, masked_movies_dir])]
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    
    cmd_list_track += [' '.join(['python3 trackSingleFile.py', masked_image_file, results_dir])]


runMultiSubproc(cmd_list_compress, max_num_process = max_num_process)
print('%'*500)
runMultiSubproc(cmd_list_track, max_num_process = max_num_process)
    