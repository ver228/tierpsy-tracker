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
    
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/MaskedVideos/Anne_Strains_20150619/Capture_Ch2_19062015_170506.hdf5'
    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/Anne_Strains_20150619'
    
    getTrajectoriesWorker(masked_image_file, results_dir, overwrite = False)

