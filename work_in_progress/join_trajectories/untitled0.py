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



#good_traj_index, good_skel_row = getValidIndexes(skel_file, min_num_skel = 100, bad_seg_thresh = 0.5, min_dist = 0)

#%%
with pd.HDFStore(skel_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']
    print(fid)


good = trajectories_data['auto_label'] == WLAB['GOOD_SKE'];
trajectories_data = trajectories_data[good]
#%%
min_num_skel = 100
N = trajectories_data.groupby('worm_index_joined').agg({'has_skeleton':np.nansum})
N = N[N>min_num_skel].dropna()

good_traj_index = N.index
#%%

good = trajectories_data['worm_index_joined'].isin(good_traj_index)
worm_index = trajectories_data.loc[good, 'worm_index_joined']
tt = trajectories_data.loc[good,'frame_number']
good_skel_row = worm_index.index
worm_index = worm_index.values
#%%

with h5py.File(skel_file, 'r') as fid:
    area = fid['/contour_area'][good_skel_row]
    length = fid['/skeleton_length'][good_skel_row]
    width = fid['/width_midbody'][good_skel_row]
#%%
N_worms = np.bincount(worm_index)[good_traj_index]

#%%
x = length
iqr = np.subtract(*np.percentile(x, [75, 25]))


#freedman-diaconis rule https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
N = np.median(N_worms)#x.size
bin_size = 2*iqr/(N**(1/3))
edges = np.arange(np.min(x) - bin_size/2, np.max(x) + bin_size/2, bin_size)

all_hist = np.zeros((good_traj_index.size, edges.size-1))

for ii, wid in enumerate(good_traj_index):
   good =  worm_index==wid
   xx = x[good]
   counts, _ = np.histogram(xx, edges)
   counts = counts/xx.size
   all_hist[ii, :] = counts

#%%
Dpq = np.zeros((good_traj_index.size, good_traj_index.size))

for ii in range(good_traj_index.size):
    for jj in range(good_traj_index.size):
        #Jensen-Shannon divergence
        p = all_hist[ii]
        q = all_hist[jj]
        m = (p+q)/2
        
        Eq = p*np.log(p/m)
        Ep = q*np.log(q/m)
        Dpq[ii, jj] =  np.sum(Eq[~np.isnan(Eq)]) + np.sum(Ep[~np.isnan(Ep)])
#%%
#freedman-diaconis rule https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
N = min_num_skel
bin_size = 2*iqr/(N**(1/3))
edges = np.arange(np.min(x) - bin_size/2, np.max(x) + bin_size/2, bin_size)

hist_first = np.zeros((good_traj_index.size, edges.size-1))
hist_last = np.zeros((good_traj_index.size, edges.size-1))

for ii, wid in enumerate(good_traj_index):
   good =  worm_index==wid
   xx = x[good]
   
   xx_first = xx[:min_num_skel]
   xx_last = xx[-min_num_skel:]
   
   counts, _ = np.histogram(xx_first, edges)
   counts = counts/xx_first.size
   hist_first[ii, :] = counts

   counts, _ = np.histogram(xx_last, edges)
   counts = counts/xx_last.size
   hist_last[ii, :] = counts



#%%
plt.figure()

plt.plot(all_hist[0,:], 'b')
plt.plot(all_hist[1,:], 'r')

plt.plot(all_hist[17,:], 'b')
plt.plot(all_hist[18,:], 'r')

#%%
plt.figure()

plt.plot(hist_last[0,:], 'b')
plt.plot(hist_last[1,:], 'r')

plt.plot(hist_first[17,:], 'b')
plt.plot(hist_first[18,:], 'r')
#%%
plt.figure()
x = length
good = worm_index==2
plt.plot(tt[good], x[good], '.b')
good = worm_index==3
plt.plot(tt[good], x[good], '.r')

good = worm_index==116
plt.plot(tt[good], x[good], '.b')
good = worm_index==117
plt.plot(tt[good], x[good], '.r')

#%%

#plt.imshow(Dpq, interpolation='none', cmap ='jet')