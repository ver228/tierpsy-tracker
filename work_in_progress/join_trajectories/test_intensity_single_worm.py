# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:07:35 2016

@author: ajaver
"""

import pandas as pd
import h5py
import numpy as np
import matplotlib.pylab as plt

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.featuresAnalysis.obtainFeaturesHelper import getValidIndexes, WLAB


intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 swimming_2011_03_04__13_16_37__8_intensities.hdf5'    
#intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 on food Rz_2011_03_04__12_55_53__7_intensities.hdf5'    

#%%
with pd.HDFStore(intensities_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']
    
    worm_maps= fid.get_node('/straighten_worm_intensity')[:]
    worm_avg = fid.get_node('/straighten_worm_intensity_median')[:]

#%%
plt.figure()
plt.imshow(worm_avg[1000:3000].T, interpolation='none', cmap='gray')
plt.grid('off')




