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
results_root = '/Volumes/behavgenom$/GeckoVideo/Results/'


max_num_process = 6

subdir_base = sys.argv[1]

masked_movies_dir = masked_movies_root + subdir_base + os.sep
results_dir = results_root + subdir_base + os.sep

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

movie_files = glob.glob(masked_movies_dir + os.sep + '*.hdf5') 

cmd_list_compress = []
cmd_list_track = []
for video_file in movie_files:
    masked_image_file = masked_movies_dir + video_file
    assert os.path.exists(masked_image_file)
    
    cmd_list_track += [' '.join(['python3 trackSingleFile.py', masked_image_file, results_dir])]


runMultiSubproc(cmd_list_compress, max_num_process = max_num_process)
print('%'*500)
runMultiSubproc(cmd_list_track, max_num_process = max_num_process)
    