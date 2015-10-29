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
script_compress = scripts_dir + 'compressSingleLocal.py'
script_track = scripts_dir +  'trackSingleLocal.py'

#input parameters
max_num_process = 6
video_ext = '*.mjpg'
json_file = ''

dir_main = sys.argv[1]
assert os.path.exists(dir_main)
if dir_main[-1] != os.sep: dir_main += os.sep #add the path separator at the end the main directory 

if len(sys.argv) > 2:
	main_subfolder = sys.argv[2]
else:
	main_subfolder = '/Volumes/behavgenom$/GeckoVideo'
if main_subfolder[-1] != os.sep: main_subfolder += os.sep

#construct saving directories from the root directory
masked_movies_root =  main_subfolder + 'MaskedVideos/'
results_root = main_subfolder + 'Results/'
tmp_dir_root = os.path.join(os.path.expanduser("~"), 'Tmp')



#create temporary directories. For the moment the user is responsable to clean the directories when

subdir_base = os.path.split(dir_main[:-1])[-1]

masked_movies_dir = masked_movies_root + subdir_base + os.sep
results_dir = results_root + subdir_base + os.sep

tmp_masked_dir = os.path.join(tmp_dir_root, 'MaskedVideos', subdir_base) + os.sep
tmp_results_dir = os.path.join(tmp_dir_root, 'Results', subdir_base) + os.sep


if not os.path.exists(results_dir): os.makedirs(results_dir)
if not os.path.exists(masked_movies_dir): os.makedirs(masked_movies_dir)
if not os.path.exists(tmp_masked_dir): os.makedirs(tmp_masked_dir)
if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)

#create the list of commands for the analsys
movie_files = glob.glob(dir_main + os.sep + video_ext) 

cmd_list_compress = []
cmd_list_track = []
for video_file in movie_files:
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    
    cmd_list_compress += [['python3',  script_compress, video_file, masked_movies_dir, tmp_masked_dir, json_file]]
    cmd_list_track += [['python3', script_track, masked_image_file, results_dir, tmp_masked_dir, tmp_results_dir, json_file]]


cmd_list_compress = cmd_list_compress
#print(cmd_list_compress)
runMultiCMD(cmd_list_compress, max_num_process = max_num_process, refresh_time=10)
#print('%'*500)
#runMultiSubproc(cmd_list_track, max_num_process = max_num_process)
#if tmp_masked_dir != masked_movies_dir: os.remove(tmp_masked_dir)
    