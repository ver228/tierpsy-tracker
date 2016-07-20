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


masked_movies_root = '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/'
results_root = '/Volumes/behavgenom$/GeckoVideo/Results/'
directories_done = ['20150519', '20150521_1115', '20150522_1940']


if results_root[-1] != os.sep:
    results_root += os.sep

masks_files = glob.glob(masked_movies_root + os.sep + '/*/*.hdf5')

cmd_list = []
for masked_image_file in masks_files:
    subdir = masked_image_file.rpartition(os.sep)[0].rpartition(os.sep)[-1]
    if subdir not in directories_done:
        results_dir = results_root + subdir
        cmd_list += [' '.join(['python3 trackSingleFile.py',
                               masked_image_file, results_dir])]


runMultiSubproc(cmd_list, max_num_process=24)
