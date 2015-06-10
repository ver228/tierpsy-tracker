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


def calWormAngles(x,y):
    assert(len(x.shape)==1)
    assert(len(y.shape)==1)
    
    dx = np.diff(x);
    dy = np.diff(y);
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
    
    #    % need to deal with cases where angle changes discontinuously from -pi
    #    % to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
    #    % respectively to all remaining points.  This effectively extends the
    #    % range outside the -pi to pi range.  Everything is re-centred later
    #    % when we subtract off the mean.
    #    
    #    % find discontinuities larger than pi (skeleton cannot change direction
    #    % more than pi from one segment to the next)
    positiveJumps = np.where(dAngles > np.pi)[0] + 1; #%+1 to cancel shift of diff
    negativeJumps = np.where(dAngles <-np.pi)[0] + 1;
    
    #% subtract 2pi from remainging data after positive jumps
    for jump in positiveJumps:
        angles[jump:] = angles[jump:] - 2*np.pi;
    
    #% add 2pi to remaining data after negative jumps
    for jump in negativeJumps:
        angles[jump:] = angles[jump:] + 2*np.pi;
    
    #% rotate skeleton angles so that mean orientation is zero
    meanAngle = np.mean(angles);
    angles = angles - meanAngle;
    
    return (angles, meanAngle)

def calWormAnglesAll(skeleton):
    angles_all = np.zeros((skeleton.shape[0], skeleton.shape[2]-1))
    meanAngles_all = np.zeros(skeleton.shape[0])
    for ss in range(skeleton.shape[0]):
        angles_all[ss,:],meanAngles_all[ss] = calWormAngles(skeleton[ss,0,:],skeleton[ss,1,:])
    return angles_all, meanAngles_all


def recalculate_clusters(buff_main, main_range, max_n_clusters = 20., delta_cluster = 100.):
    cluster_dat = {};
    for i_key, key in enumerate(['V', 'D']):
        buff = buff_main[key][main_range[0]:main_range[1]+1,:]#.astype(np.float32)
        nclusters = int(min(np.ceil(buff.shape[0]/delta_cluster), max_n_clusters));
        
        est = KMeans(n_clusters=nclusters);
        est.fit(buff);
        cluster_dat[key] = est.cluster_centers_
    return cluster_dat
#%%
def calcBestMatchProb(contrastmap_fid, cluster_dat, buffN):
    
    d1 = np.zeros((buffN['D'].shape[0], 4))
    d1[:,0] = np.min(cdist(cluster_dat['V'], buffN['V']), axis=0)
    d1[:,1] = np.min(cdist(cluster_dat['V'], buffN['V'][:,::-1]), axis=0)
    d1[:,2] = np.min(cdist(cluster_dat['V'], buffN['D']), axis=0)
    d1[:,3] = np.min(cdist(cluster_dat['V'], buffN['D'][:,::-1]), axis=0)
    
    d2 = np.zeros((buffN['D'].shape[0], 4))
    d2[:,0] = np.min(cdist(cluster_dat['D'], buffN['D']), axis=0)
    d2[:,1] = np.min(cdist(cluster_dat['D'], buffN['D'][:,::-1]), axis=0)
    d2[:,2] = np.min(cdist(cluster_dat['D'], buffN['V']), axis=0)
    d2[:,3] = np.min(cdist(cluster_dat['D'], buffN['V'][:,::-1]), axis=0)
    
    
    winningV = np.zeros(4)
    for d in np.argmin(d1, axis=1):
        winningV[d] +=1
    winningD = np.zeros(4)
    for d in np.argmin(d2, axis=1):
        winningD[d] +=1
    
    probW = winningV + winningD
    probW = probW/np.sum(probW)

    return (next_pos, buffN['V'].shape[0],  np.argmax(probW), np.max(probW))

#%%
#contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap-short.hdf5';
#contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_lmap.hdf5';
#segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_lmap.hdf5';
 
#contrastmap_fid = h5py.File(contrastmap_file, 'r');

segworm_file_fix = segworm_file[:-5] + '_fix2' + segworm_file[-5:];
os.system('cp "%s" "%s"' % (segworm_file, segworm_file_fix))

cmap_fid = pd.HDFStore(contrastmap_file, 'r');
block_index = cmap_fid['/block_index'];
cmap_fid.close()

contrastmap_fid = tables.File(contrastmap_file, 'r');
segworm_fid = h5py.File(segworm_file_fix, 'r+');




#%%
tic_ini = time.time()
worm_ids = block_index['worm_index_joined'].unique();
for ii_worm, worm_id in enumerate(worm_ids):
    tic = time.time()
    worm_block = block_index[block_index['worm_index_joined']==worm_id]
    
    all_skeL = segworm_fid['/segworm_results/skeleton_length'][:]
    segworm_id = worm_block['segworm_id'].values
    
    skeL = all_skeL[segworm_id];
    med = np.median(skeL);
    mad = np.median(np.abs(skeL-med))
    good_length = (skeL>med-5*mad) & (skeL<med+5*mad);
    
#    plt.figure()
#    plt.plot(skeL)
#    plt.plot(plt.xlim(), np.ones(2)*(med+5*mad), 'r--')
#    plt.plot(plt.xlim(), np.ones(2)*(med-5*mad), 'r--')
#    
#    
    block_count = worm_block.loc[good_length, 'block_id'].value_counts(); 
    #if the segmentation with the wrong length are to be removed, remember to update in the main table later on
    if len(block_count) <= 1:
        continue
    start_block = block_count.index[0]
    last_block  = block_count.index.max()
    fist_block = block_count.index.min()
    #%%
    buff_main = {};
    buff_main['V'] = contrastmap_fid.get_node('/block_lmap/worm_V')[:];
    buff_main['D'] = contrastmap_fid.get_node('/block_lmap/worm_D')[:];
    
    good = (worm_block['block_id'] == start_block); 
    cmaps_ids = worm_block.loc[good, 'lmap_id'].values
    main_range = [cmaps_ids[0], cmaps_ids[-1]]
    main_tot_prev = main_range[1]-main_range[0]+1
    
    allProb = [];
    cluster_dat = recalculate_clusters(buff_main, main_range)
    
    deltaT = max(last_block-start_block, start_block-fist_block);
    for ii_add in range(1,deltaT+1):
        for ii_sign in [-1, 1]:
            next_pos = start_block + ii_add*ii_sign
            
            if next_pos<fist_block or next_pos>last_block:
                continue
            
            good = (worm_block['block_id'] == next_pos); 
            cmaps_ids = worm_block.loc[good, 'lmap_id'].values
            if cmaps_ids.size == 0:
                continue
            
            block_range = (cmaps_ids[0], cmaps_ids[-1])
            
            buffN = {}
            buffN['V'] = buff_main['V'][block_range[0]:block_range[1]+1, :].copy();
            buffN['D'] = buff_main['D'][block_range[0]:block_range[1]+1, :].copy();
        
            probTuple = calcBestMatchProb(contrastmap_fid, cluster_dat, buffN)
            if probTuple[2]%2 == 1:
                buff_main['V'][block_range[0]:block_range[1]+1, :] = buffN['V'][:,::-1];
                buff_main['D'][block_range[0]:block_range[1]+1, :] = buffN['D'][:,::-1];
                
            if probTuple[2] >= 2:
                buff_main['V'][block_range[0]:block_range[1]+1, :] = buffN['D'];
                buff_main['D'][block_range[0]:block_range[1]+1, :] = buffN['V'];
            
            #check if the number of blocks added is large enough to recalculate the clusters
            main_range[0] = min(main_range[0], block_range[0])
            main_range[1] = max(main_range[1], block_range[1])
            allProb += [probTuple]
        
        main_tot = main_range[1]-main_range[0]+1;
        if main_tot - main_tot_prev > 200:
            cluster_dat = recalculate_clusters(buff_main, main_range)
            main_tot_prev = main_tot
            
    #print allProb
    
    #%%
    for block_id, tot_rows, switch_type, dd in allProb:
        kernel = '(worm_index_joined==%d) & (block_id==%d)' % (worm_id, block_id)
        block_segworm_id = block_index.query(kernel)['segworm_id']
        
        if switch_type % 2 == 1: 
            for cc in ['skeleton', 'contour_dorsal', 'contour_ventral']:
                assert len(block_segworm_id) == tot_rows
                for nn in block_segworm_id:
                    aa = segworm_fid['/segworm_results/' + cc][nn,:,:]
                    segworm_fid['/segworm_results/' + cc][nn,:,:] = aa[:,::-1]
                    
        if switch_type >= 2: 
            for nn in block_segworm_id:
                vv = segworm_fid['/segworm_results/contour_ventral'][nn,:,:]
                dd = segworm_fid['/segworm_results/contour_dorsal'][nn,:,:]
                
                segworm_fid['/segworm_results/contour_ventral'][nn,:,:] = dd
                segworm_fid['/segworm_results/contour_dorsal'][nn,:,:] = vv
    
    print ii_worm, len(worm_ids), time.time()-tic, time.time() - tic_ini

segworm_fid.close()
contrastmap_fid.close()

#%%
contrastmap_fid = tables.File(contrastmap_file, 'r');
nsegments = contrastmap_fid.get_node('/block_lmap/worm_V').shape[-1];
block_order = block_count.index
cluster_dat = {};
for key in ['worm_V', 'worm_D']:
    cluster_dat[key] = {'CM':np.zeros((0,nsegments)), 'ind':np.zeros(0)}
plt.figure()
for iblock in range(len(block_order)):
    good = (worm_block['block_id'] == block_order[iblock]) & good_length;
    cmaps_ids = worm_block.loc[good, 'lmap_id'].values
    if len(cmaps_ids)==0:
        continue
    block_range = (cmaps_ids[0], cmaps_ids[-1])
    
    for i_key, key in enumerate(['worm_V', 'worm_D']):
        buff = contrastmap_fid.get_node('/block_lmap/' + key)[block_range[0]:block_range[1]+1, :];
        
        nclusters = min(int(np.ceil(buff.shape[0]/100.0)), 10);
        est = KMeans(n_clusters=nclusters);
        est.fit(buff);
        
        cluster_dat[key]['CM'] = np.vstack((cluster_dat[key]['CM'], est.cluster_centers_))
        cluster_dat[key]['ind']  = np.hstack((cluster_dat[key]['ind'], np.ones(nclusters)*block_order[iblock]))
        plt.subplot(6,6, iblock*2 + i_key+1)
        plt.plot(est.cluster_centers_.T);
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
