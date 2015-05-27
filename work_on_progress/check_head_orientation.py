# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:35:04 2015

@author: ajaver
"""

import pandas as pd
import h5py
import numpy as np
import matplotlib.pylab as plt
import tables

#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_trajectories.hdf5';
#segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_segworm.hdf5';
#contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_lmap.hdf5';

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303.hdf5';
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303_lmap.hdf5';


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
#df =  df[df['worm_index_joined'] == 2608] , 8, 538, 3433
worm_index = 773
df =  df[df['worm_index_joined'] == worm_index]
df = df[df['segworm_id']>=0]; #select only rows with a valid segworm skeleton
table_fid.close()

segworm_fid = h5py.File(segworm_file, 'r');



segworm_id = df['segworm_id'].values

#%%
aa = segworm_fid['/segworm_results/skeleton'].shape
skeletons = np.zeros((segworm_id.shape[0], aa[1], aa[2]),segworm_fid['/segworm_results/skeleton'].dtype) 
for ii, seg_id in enumerate(segworm_id):
    skeletons[ii,:,:] = segworm_fid['/segworm_results/skeleton'][seg_id,:,:]
    if ii %1000 == 0:
        print ii, skeletons.shape[0]
#%%
def getAnglesDelta(dx,dy):
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
        
    
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
#%%
MAX_DELT = 10;
delTs = np.diff(df['frame_number']); 
block_ind = np.zeros(len(delTs)+1, dtype = np.int);    
block_ind[0] = 1;
for ii, delT in enumerate(delTs):
    if delT <= MAX_DELT:
        block_ind[ii+1] = block_ind[ii];
    else:
        block_ind[ii+1] = block_ind[ii]+1;

delta_ind = [5]#range(1,20)

window_std = 25
for ii in delta_ind:
    ts = pd.DataFrame({'head':np.nan, 'tail':np.nan, 'block_id':np.nan}, \
    np.arange(df['frame_number'].min(), df['frame_number'].max()+1))
    
    dx = skeletons[:,0,ii] - skeletons[:,0,0]
    dy = skeletons[:,1,ii] - skeletons[:,1,0]

    angles_head,_ = getAnglesDelta(dx,dy)
    
    dx = skeletons[:,0,-ii-1] - skeletons[:,0,-1]
    dy = skeletons[:,1,-ii-1] - skeletons[:,1,-1]

    angles_tail,_ = getAnglesDelta(dx,dy)
    
    ts.loc[df['frame_number'],'head'] = angles_head
    ts.loc[df['frame_number'],'tail'] = angles_tail
    ts.loc[df['frame_number'],'block_id'] = block_ind
    
    roll_std = pd.rolling_std(ts, window_std, window_std-MAX_DELT);
    roll_std[['head', 'tail']].plot()
    
    HT_diff = np.abs(roll_std['head']-roll_std['tail'])
    med = np.median(HT_diff);
    #mad = np.median(np.abs(HT_diff-med))
    
    prob_head = (roll_std['head']>roll_std['tail'])#.astype(np.float);
    prob_head[(HT_diff<med) | np.isnan(HT_diff)] = np.nan
    
    prob_head = pd.rolling_mean(prob_head, window_std, window_std/2)
    prob_head = np.roll(prob_head, window_std/2)
    ts['prob_head'] = prob_head;
    #plt.plot(prob_head.index, prob_head)
     
head_prob = ts[['block_id', 'prob_head']].groupby('block_id').aggregate(np.mean).dropna();

    
#%%
plt.figure()
dx = skeletons[:,0,0] - skeletons[:,0,-1]
dy = skeletons[:,1,0] - skeletons[:,1,-1]
R = np.sqrt(dx*dx + dy*dy)
plt.plot(R)
#%%
#
#cmaps_fid = tables.File(contrastmap_file, 'r');
#lmap_D = cmaps_fid.get_node('/block_lmap/worm_D');
#lmap_V = cmaps_fid.get_node('/block_lmap/worm_V');
#block_index = cmaps_fid.get_node('/block_index');
#worm_lmap_id = block_index.read_where("worm_index_joined==%i" % worm_index)['lmap_id']
#assert np.all(np.diff(worm_lmap_id)==1)
#
#worm_lmap_D = lmap_D[worm_lmap_id[0]:worm_lmap_id[-1],:]
#worm_lmap_V = lmap_V[worm_lmap_id[0]:worm_lmap_id[-1],:]
#
##%%
#plt.figure()
#plt.plot(np.mean(worm_lmap_D, axis=0)[0:25])
#plt.plot(np.mean(worm_lmap_D, axis=0)[:25:-1])
#plt.figure()
#plt.plot(np.mean(worm_lmap_V, axis=0)[0:25])
#plt.plot(np.mean(worm_lmap_V, axis=0)[:25:-1])
##%%
##
##plt.figure()
##plt.plot(np.mean(worm_lmap_V[:, 4:7] + worm_lmap_D[:, 4:7], axis=1))
##plt.plot(np.mean(worm_lmap_V[:, -6:-3] + worm_lmap_D[:, -6:-3], axis=1))
#
##%%
#fix_ind = 3
#delta_ii = 2
#
#neck_V = np.mean(worm_lmap_V[:, (fix_ind-delta_ii):(fix_ind+delta_ii+1)], axis=1)
#waist_V = np.mean(worm_lmap_V[:, (-fix_ind-delta_ii-1):(-fix_ind+delta_ii)], axis=1)
#neck_D = np.mean(worm_lmap_D[:, (fix_ind-delta_ii):(fix_ind+delta_ii+1)], axis=1)
#waist_D = np.mean(worm_lmap_D[:, (-fix_ind-delta_ii-1):(-fix_ind+delta_ii)], axis=1)
#
#waist_ind =waist_V + waist_D
#neck_ind =neck_V + neck_D
#
##ints = pd.DataFrame({'neck_V':neck_V, 'waist_V':waist_V}, index = df['frame_number'].values[:-1])
##pd.rolling_mean(ints, 100).plot()
##ints = pd.DataFrame({'waist_D':waist_D, 'neck_D':neck_D}, index = df['frame_number'].values[:-1])
##pd.rolling_mean(ints, 100).plot()
#ints = pd.DataFrame({'neck':neck_ind, 'waist':waist_ind}, index = df['frame_number'].values[:-1])
#pd.rolling_mean(ints, 100).plot()

