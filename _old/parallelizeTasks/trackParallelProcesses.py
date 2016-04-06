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
#masked_movies_root =  '/Users/ajaver/Desktop/Gecko_compressed/MaskedVideos/'
#results_root = '/Users/ajaver/Desktop/Gecko_compressed/Results/'

home = os.path.expanduser("~")
tmp_masked_root = home + os.sep + 'Tmp' + os.sep +  'MaskedVideos' + os.sep
tmp_results_root = home + os.sep + 'Tmp' + os.sep + 'Results' + os.sep


max_num_process = 12

subdir_base = sys.argv[1]

masked_movies_dir = masked_movies_root + subdir_base + os.sep
results_dir = results_root + subdir_base + os.sep

tmp_masked_dir = tmp_masked_root + subdir_base + os.sep
tmp_results_dir = tmp_results_root + subdir_base + os.sep

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(masked_movies_dir):
    os.makedirs(tmp_results_dir)


movie_files = glob.glob(masked_movies_dir + os.sep + '*.hdf5') 

cmd_list_track = []
for masked_image_file in movie_files:
    assert os.path.exists(masked_image_file)
    
    cmd_list_track += [' '.join(['python3 trackSingleLocal.py', masked_image_file, results_dir, tmp_masked_dir, tmp_results_dir])]


runMultiSubproc(cmd_list_track, max_num_process = max_num_process)
    