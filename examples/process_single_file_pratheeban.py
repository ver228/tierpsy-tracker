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



video_file = '/Users/ajaver/Desktop/Pratheeban_videos/RawData/15_07_03_2hrL1_Ch1_03072015_162628.mjpg'

json_file = video_file.rpartition('.')[0] + '.json'

mask_files_dir = '/Users/ajaver/Desktop/Pratheeban_videos/MaskedVideos/'
results_dir = '/Users/ajaver/Desktop/Pratheeban_videos/Results/'

masked_image_file = compressVideoWorker(video_file, mask_files_dir, param_file = json_file)
getTrajectoriesWorker(masked_image_file, results_dir, param_file = json_file)