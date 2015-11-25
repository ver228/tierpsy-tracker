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

masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/Alex_Anderson/MaskedVideos/Locomotion_videos_for_analysis_2015-2/'
json_file = '/Users/ajaver/Desktop/Gecko_compressed/Alex_Anderson/param.json'

assert os.path.exists(masked_movies_dir)
if masked_movies_dir[-1] != os.sep: masked_movies_dir += os.sep #add the path separator at the end the main directory 

#construct saving directories from the root directory
results_dir = masked_movies_dir.replace('MaskedVideos', 'Results')
if not os.path.exists(results_dir): os.makedirs(results_dir)


#create temporary directories. For the moment the user is responsable to clean the directories when
home = os.path.expanduser("~")

dum = masked_movies_dir if masked_movies_dir[-1] != os.sep else masked_movies_dir[:-1]
subdir_base = os.path.split(dum)[-1]

tmp_masked_dir = os.path.join(home, 'Tmp', 'MaskedVideos', subdir_base) + os.sep
tmp_results_dir = os.path.join(home, 'Tmp', 'Results', subdir_base) + os.sep

if not os.path.exists(tmp_masked_dir): os.makedirs(tmp_masked_dir)
if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)


#create the list of commands for the analsys
masked_image_files = glob.glob(masked_movies_dir + os.sep + '*.hdf5') 

cmd_list_track = []
for masked_image_file in masked_image_files:
    cmd_list_track += [['python3', script_track, masked_image_file, results_dir, tmp_masked_dir, tmp_results_dir, json_file]]

cmd_list_track = cmd_list_track[-2:-1]
#print(cmd_list_track)
runMultiCMD(cmd_list_track, max_num_process = max_num_process, refresh_time = 10)
#print('%'*500)
#runMultiSubproc(cmd_list_track, max_num_process = max_num_process)
#if tmp_masked_dir != masked_movies_dir: os.remove(tmp_masked_dir)
    