# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:21:02 2015

@author: ajaver
"""

import pandas as pd
import tables
import h5py
import numpy as np
import cv2
from min_avg_difference import min_avg_difference, avg_difference_mat, min_avg_difference2
#from getWormAngles import calWormAngles
#from image_difference import image_difference
import os
import time


import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_lmap.hdf5';
 
cmap_fid = pd.HDFStore(contrastmap_file, 'r');
block_index = cmap_fid['/block_index'];
cmap_fid.close()

contrastmap_fid = tables.File(contrastmap_file, 'r');
segworm_fid = tables.File(segworm_file, 'r');

lmaps_data = contrastmap_fid.get_node('/block_lmap')
#%%
tic_ini = time.time()
worm_ids = [8]#block_index['worm_index_joined'].unique();
for ii_worm, worm_id in enumerate(worm_ids):
    tic = time.time()
    worm_block = block_index[block_index['worm_index_joined']==worm_id]
    assert np.all(np.diff(worm_block['lmap_id']) == 1)
    block_range = (worm_block['lmap_id'].iloc[0], worm_block['lmap_id'].iloc[-1])
    block_maps = lmaps_data[block_range[0]:block_range[1]+1,:,:]
    
#    all_skeL = segworm_fid['/segworm_results/skeleton_length'][:]
#    segworm_id = worm_block['segworm_id'].values
#    
#    skeL = all_skeL[segworm_id];
#    med = np.median(skeL);
#    mad = np.median(np.abs(skeL-med))
#    good_length = (skeL>med-5*mad) & (skeL<med+5*mad);
    
#    block_count = worm_block.loc[good_length, 'block_id'].value_counts(); 
    #if the segmentation with the wrong length are to be removed, remember to update in the main table later on
#    if len(block_count) <= 1:
#        continue
#    start_block = block_count.index[0]
#    last_block  = block_count.index.max()
#    fist_block = block_count.index.min()
    #%%
    
            


#contrastmap_fid.close()

#%%
#block_ids = np.unique(cluster_dat[key]['ind'])
#bb = 30;
#N = ('D', 'V')
#i_key = 0;
#
#key_a = 'worm_' + N[i_key]
#key_b = 'worm_' + N[not i_key]
#
#good = cluster_dat[key_a]['ind']==bb;
#current_CM = cluster_dat[key_a]['CM'][good,:]
#rest_CM_a = cluster_dat[key_a]['CM'][~good,:]
#rest_id_a = cluster_dat[key_a]['ind'][~good]
#
#rest_CM_b = cluster_dat[key_b]['CM'][~good,:]
#rest_id_b = cluster_dat[key_b]['ind'][~good]
#
#
#tot_dist = cdist(current_CM, rest_CM_a)
#min_ind = np.argmin(tot_dist, axis=1)
#min_dist = tot_dist[range(tot_dist.shape[0]), min_ind];
#min_block = rest_id[min_ind]
#plt.plot(min_dist)
#print min_block
#
#tot_dist = cdist(current_CM, rest_CM_a[:,::-1])
#min_ind = np.argmin(tot_dist, axis=1)
#min_dist = tot_dist[range(tot_dist.shape[0]), min_ind];
#min_block = rest_id[min_ind]
#plt.plot(min_dist)
#print min_block
#
#tot_dist = cdist(current_CM, rest_CM_b)
#min_ind = np.argmin(tot_dist, axis=1)
#min_dist = tot_dist[range(tot_dist.shape[0]), min_ind];
#min_block = rest_id[min_ind]
#plt.plot(min_dist)
#print min_block
#
#tot_dist = cdist(current_CM, rest_CM_b[:,::-1])
#min_ind = np.argmin(tot_dist, axis=1)
#min_dist = tot_dist[range(tot_dist.shape[0]), min_ind];
#min_block = rest_id[min_ind]
#plt.plot(min_dist)
#print min_block
