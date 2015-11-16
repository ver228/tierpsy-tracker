# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:38:05 2015

@author: ajaver
"""

import sys
sys.path.append('../..')

from MWTracker.helperFunctions.compressVideoWorker import compressVideoWorker

if __name__ == "__main__":
    video_file = sys.argv[1]
    mask_files_dir = sys.argv[2]

    compressVideoWorker(video_file, mask_files_dir)