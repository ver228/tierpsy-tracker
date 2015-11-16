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



home = os.path.expanduser("~")
tmp_masked_root = home + os.sep + 'Tmp' + os.sep +  'MaskedVideos' + os.sep
tmp_results_root = home + os.sep + 'Tmp' + os.sep + 'Results' + os.sep

max_num_process = 12

#dir_main = 
#dir_main = '/Volumes/Mrc-pc/20150522_1940/'

masked_movies_dir =  sys.argv[1]
if masked_movies_dir[-1] == os.sep:
    masked_movies_dir = masked_movies_dir[:-1]

results_dir = masked_movies_dir.replace('MaskedVideos', 'Results')

subdir_base = os.path.split(masked_movies_dir)[-1]    
tmp_results_dir = tmp_results_root + subdir_base + os.sep
#I'll try to do everything having the masked files in the server otherwise the local harddrive gets full
tmp_masked_dir = masked_movies_dir #tmp_masked_root + subdir_base + os.sep


if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(tmp_masked_dir):
    os.makedirs(tmp_masked_dir)

if not os.path.exists(tmp_results_dir):
    os.makedirs(tmp_results_dir)

mask_files = glob.glob(masked_movies_dir + os.sep + '*.hdf5') 

cmd_list_track = []
for masked_file in mask_files:
    if not 'Ch6' in masked_file:
        continue

    cmd_list_track += [' '.join(['python3 trackSingleLocal.py', "'" + masked_file + "'",
                                 "'" + results_dir + "'", "'" + tmp_masked_dir + "'", "'" + tmp_results_dir + "'"])]

print(cmd_list_track)

#runMultiSubproc(cmd_list_track, max_num_process = max_num_process)
    