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


#intensities_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_intensities.hdf5'
intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 swimming_2011_03_04__13_16_37__8_intensities.hdf5'    
    


#good_traj_index, good_skel_row = getValidIndexes(skel_file, min_num_skel = 100, bad_seg_thresh = 0.5, min_dist = 0)

#%%
with pd.HDFStore(intensities_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

#%%
joined_tracks = {}
for worm_N, worm_group in trajectories_data[['worm_index_N', 'worm_index_joined']].groupby('worm_index_N'):
    joined_tracks[worm_N] = worm_group['worm_index_joined'].unique()

#joined_tracks = joined_tracks.groupby('worm_index_N')

#%%
some_maps = {}

for ind, trajectories_worm in trajectories_data.groupby('worm_index_joined'):
    int_map_id = trajectories_worm['int_map_id']
    
    with pd.HDFStore(intensities_file, 'r') as fid:
        some_maps[ind] = fid.get_node('/straighten_worm_intensity')[int_map_id]
#%%

#%%
int_maps = {}
for ind in some_maps:
    int_maps[ind] = np.median(some_maps[ind], axis=0).astype(np.float)
    img2 =int_maps[ind]
    #img = np.round(int_maps[ind]).astype(np.uint8)
    #img2 = cv2.resize(img, tuple([x*16 for x in img.shape[::-1]]))
    plt.figure()
    plt.imshow(img2, interpolation='none', cmap='gray')
    plt.grid('off')
    
#%%
R = {}
for ii, ind in enumerate(int_maps):
    img = int_maps[ind]
    #img = img-np.median(img)
    xx = np.median(img[:,5:12], axis=1)
    xx = xx - np.median(xx)
    R[ind] = xx
#%%
plt.figure()

ff = 'rbg'
for mm,ii in enumerate([2,3, 9]):#joined_tracks:
    #plt.figure()
    for iw in joined_tracks[ii]:
        plt.plot(R[iw], ff[mm])


#%%
N = len(R)
DD = np.full((N,N), np.inf)

traj_ind = list(R.keys())
for i1, ind1 in enumerate(traj_ind):
    for i2, ind2 in enumerate(traj_ind):
        if i1 == i2: continue
        x = R[ind1]
        y = R[ind2]
        DD[i1,i2] = np.sum(np.abs(x-y))



for ii, x in enumerate(np.argmin(DD,axis=0)): 
    print(traj_ind[ii],traj_ind[x], DD[x,ii])

#%%
num_min_skel = 1
all_RR = {}
for ii, ind in enumerate(joined_tracks[2]):
    dd = some_maps[ind]
    
    RR = np.zeros((np.round(dd.shape[0]/num_min_skel)+1, dd.shape[1]))    
    #plt.figure()
    for iim, mm in enumerate(range(0,dd.shape[0], num_min_skel)):
        
        xx = np.median(dd[mm:mm+num_min_skel, :, 5:11], axis=(0,2))
        xx = xx - np.median(xx)
        RR[iim] = xx
        #plt.plot(xx)
    all_RR[ind] = RR
    
    plt.figure()
    plt.imshow(all_RR[ind].T, interpolation='none', cmap='gray')
    plt.grid('off')
#%%
        #plt.plot(xx)
    #plt.ylim([-15, 25])




