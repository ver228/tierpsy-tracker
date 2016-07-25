# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:36:06 2015

@author: ajaver
"""

import sys
sys.path.append('..')

from MWTracker.helperFunctions.getTrajectoriesWorker import getTrajectoriesWorker

if __name__ == '__main__':
    #masked_image_file = '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/Bertie_20150618/Capture_Ch3_18062015_141051.hdf5'
    #results_dir = '/Volumes/behavgenom$/GeckoVideo/Results/Bertie_20150618/'
    masked_image_file = '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/Anne_Strains_20150611/Capture_Ch2_11062015_153552.hdf5'
    results_dir = '/Volumes/behavgenom$/GeckoVideo/Results/Anne_Strains_20150611/'

    getTrajectoriesWorker(masked_image_file, results_dir, overwrite=False)
