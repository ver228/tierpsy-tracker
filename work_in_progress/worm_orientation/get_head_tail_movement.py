# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""
import h5py
import numpy as np

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.trackWorms.checkHeadOrientation import isWormHTSwitched

intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 swimming_2011_03_04__13_16_37__8_intensities.hdf5'    

skeletons_file = intensities_file.replace('intensities', 'skeletons')
masked_image_file = intensities_file.replace('Results', 'MaskedVideos').replace('_intensities', '')
#%%
with h5py.File(skeletons_file, 'r') as fid:
    skeletons = fid['/skeleton'][:]


aa,bb = isWormHTSwitched(skeletons, max_gap_allowed=1)