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

dir_main = '/Volumes/Mrc-pc/20150522_1940/'

if dir_main[-1] == os.sep:
    dir_main = dir_main[:-1]
    
subdir_base = os.path.split(dir_main)[-1]    

movie_files = glob.glob(dir_main + os.sep + '*.mjpg') 
masked_movies_dir = masked_movies_root + subdir_base + os.sep
results_dir = results_root + subdir_base + os.sep


cmd_list = []
for movief in movie_files:
    cmd_list += [' '.join(['python3 compNTrackSingleFile.py', movief, masked_movies_dir, results_dir])]


runMultiSubproc(cmd_list, max_num_process = 6)
    