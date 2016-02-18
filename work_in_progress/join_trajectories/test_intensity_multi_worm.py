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

#masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
#masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
#masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
       
skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
intensities_file = skeletons_file.replace('_skeletons', '_intensities')

#intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 swimming_2011_03_04__13_16_37__8_intensities.hdf5'    
#intensities_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_intensities.hdf5'

#%%
with pd.HDFStore(intensities_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

#%%
#joined_tracks = {}
#for worm_N, worm_group in trajectories_data[['worm_index_N', 'worm_index_joined']].groupby('worm_index_N'):
#    joined_tracks[worm_N] = worm_group['worm_index_joined'].unique()
#%%
worm_maps = {}
worm_avg = {}
worm_std = {}
for ind, trajectories_worm in trajectories_data.groupby('worm_index_joined'):
    int_map_id = trajectories_worm['int_map_id']
    
    with pd.HDFStore(intensities_file, 'r') as fid:
        #worm_maps[ind] = fid.get_node('/straighten_worm_intensity')[int_map_id]
        worm_avg[ind] = fid.get_node('/straighten_worm_intensity_median')[int_map_id]
        worm_std[ind] = fid.get_node('/worm_intensity_std')[int_map_id]
#%%


#%%
#for ind in worm_avg:
#    plt.figure()
#    plt.imshow(worm_avg[ind].T, interpolation='none', cmap='gray')
#    plt.grid('off')

#%%
for ind in [1]:#[117, 15, 13, 192]:
    plt.figure()
    plt.imshow(worm_avg[ind].T, interpolation='none', cmap='gray')
    plt.grid('off')
    
#%%
#plt.figure()
#plt.imshow(worm_maps[ind][0], interpolation='none', cmap='gray')
#plt.grid('off')
#%%
for ind in worm_avg:
    med_int = np.median(worm_avg[ind], axis=0).astype(np.float)
    
    plt.figure()
    diff_ori = np.sum(np.abs(med_int-worm_avg[ind]), axis = 1)
    diff_inv = np.sum(np.abs(med_int[::-1]-worm_avg[ind]), axis = 1)
    plt.plot(diff_ori)
    plt.plot(diff_inv)
    plt.title(ind)

#%%
plt.figure()
for ii in range(10):
    plt.plot(worm_avg[ind][ii])

