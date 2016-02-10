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

masked_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
skel_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_skeletons.hdf5'

intensities_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_intensities.hdf5'



#good_traj_index, good_skel_row = getValidIndexes(skel_file, min_num_skel = 100, bad_seg_thresh = 0.5, min_dist = 0)

#%%
with pd.HDFStore(intensities_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']


#%%
some_maps = {}

for ind in [2, 3, 116, 117]:
    good = trajectories_data['worm_index_joined'] == ind
    int_map_id = trajectories_data.loc[good, 'int_map_id']
    
    with pd.HDFStore(intensities_file, 'r') as fid:
        some_maps[ind] = fid.get_node('/straighten_worm_intensity')[int_map_id]
        
#%%
int_maps = {}
for ind in some_maps:
    int_maps[ind] = np.median(some_maps[ind], axis=0)
    plt.figure()
    plt.imshow(int_maps[ind], interpolation='none', cmap='gray')
    plt.grid('off')
#%%
plt.figure()
plt.imshow(int_maps[2]-int_maps[116], interpolation='none', cmap='gray')
plt.grid('off')
plt.figure()
plt.imshow(int_maps[2]-int_maps[117], interpolation='none', cmap='gray')
plt.grid('off')

plt.figure()
plt.imshow(int_maps[3]-int_maps[116], interpolation='none', cmap='gray')
plt.grid('off')
plt.figure()
plt.imshow(int_maps[3]-int_maps[117], interpolation='none', cmap='gray')
plt.grid('off')

#%%
DD = np.zeros((2,2))
for i1, ind1 in enumerate([2,3]):
    for i2, ind2 in enumerate([116, 117]):
        DD[i1,i2] = np.sum(np.abs(int_maps[ind1]-int_maps[ind2]))
