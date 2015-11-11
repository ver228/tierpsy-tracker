# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

import os
import glob
import sys
from start_console import runMultiCMD

#name of the scripts used
scripts_dir = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/MWTracker_GUI/'
script_track = scripts_dir +  'trackSingleLocal.py'

#input parameters
max_num_process = 6

if len(sys.argv) > 2:
	json_file = sys.argv[2]
else:
	json_file = ''


masked_movies_dir = sys.argv[1]
assert os.path.exists(masked_movies_dir)
if masked_movies_dir[-1] != os.sep: masked_movies_dir += os.sep #add the path separator at the end the main directory 


results_dir = masked_movies_dir.replace('MaskedVideos', 'Results')

#create temporary directories. For the moment the user is responsable to clean the directories when the scripts finish
tmp_dir_root = os.path.join(os.path.expanduser("~"), 'Tmp')
subdir_base = os.path.split(masked_movies_dir[:-1])[-1]
tmp_masked_dir = os.path.join(tmp_dir_root, 'MaskedVideos', subdir_base) + os.sep
tmp_results_dir = os.path.join(tmp_dir_root, 'Results', subdir_base) + os.sep


if not os.path.exists(results_dir): os.makedirs(results_dir)
if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)
if not os.path.exists(tmp_masked_dir): os.makedirs(tmp_masked_dir)

#create the list of commands for the analsys
mask_files = glob.glob(masked_movies_dir + os.sep + '*.hdf5') 

cmd_list_track = []
for masked_image_file in mask_files:
    cmd_list_track += [['python3', script_track, masked_image_file, results_dir, tmp_masked_dir, tmp_results_dir, json_file]]


print(cmd_list_track)
runMultiCMD(cmd_list_track, max_num_process = max_num_process, refresh_time = 0.1)

    