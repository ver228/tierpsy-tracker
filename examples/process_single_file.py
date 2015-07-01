# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:30:56 2015

@author: ajaver
"""
import os
import sys
sys.path.append('..')

from MWTracker.helperFunctions.compressSingleFile import getCompressVidWorker
from MWTracker.helperFunctions.trackSingleFile import getTrajectoriesWorker


example_dir = '/Users/ajaver/Desktop/Multiworm/Data/'

video_file = example_dir + 'Capture_Ch1_18062015_140908.mjpg'
mask_files_dir = os.path.join(example_dir, 'MaskedVideos') + os.sep
results_dir = os.path.join(example_dir, 'Results') + os.sep

masked_image_file = getCompressVidWorker(video_file, mask_files_dir)
getTrajectoriesWorker(masked_image_file, results_dir, overwrite = False)
