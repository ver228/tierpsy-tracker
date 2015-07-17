# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:30:56 2015

@author: ajaver
"""
import os
import sys
sys.path.append('..') #path to the MWTracker

from MWTracker.helperFunctions.compressVideoWorker import compressVideoWorker
from MWTracker.helperFunctions.getTrajectoriesWorker import getTrajectoriesWorker



example_dir = '/Users/ajaver/Google Drive/MWTracker_Example/'

video_file = example_dir + 'Capture_Ch1_18062015_140908.mjpg'
mask_files_dir = os.path.join(example_dir, 'MaskedVideos') + os.sep
results_dir = os.path.join(example_dir, 'Results') + os.sep

masked_image_file = compressVideoWorker(video_file, mask_files_dir)
getTrajectoriesWorker(masked_image_file, results_dir, overwrite = False)

