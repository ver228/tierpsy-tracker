# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:26:14 2015

@author: ajaver
"""

import sys

from compressSingleFile import getCompressVidWorker
from trackSingleFile import getTrajectoriesWorker


def getCompNTrackWorker(video_file, mask_files_dir, results_dir):
    
    masked_image_file = getCompressVidWorker(video_file, mask_files_dir, \
                    has_timestamp = True, FPS = 25, expected_frames = 1000000)
                    
    #getTrajectoriesWorker(masked_image_file, results_dir)
    
if __name__ == "__main__":
    video_file = sys.argv[1]
    mask_files_dir = sys.argv[2]
    results_dir = sys.argv[3]
    getCompNTrackWorker(video_file, mask_files_dir, results_dir)