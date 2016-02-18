# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:36:43 2016

@author: ajaver
"""

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

#skel_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_skeletons.hdf5'
intensities_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_intensities.hdf5'
#intensities_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch5_17112015_205616_intensities.hdf5'

#good_traj_index, good_skel_row = getValidIndexes(skel_file, min_num_skel = 100, bad_seg_thresh = 0.5, min_dist = 0)


def getBlocksIDs(invalid, max_gap_allowed = 10):
    #ONE CAN CALL THIS FROM checkHeadOrientation
    '''The skeleton array is divided in blocks of contingous skeletons with 
    a gap between unskeletonized frames less than max_gap_allowed'''
    
    good_ind = np.where(~invalid)[0];            
    delTs = np.diff(good_ind)
    
    block_ind = np.zeros_like(good_ind)
    block_ind[0] = 1;
    for ii, delT in enumerate(delTs):
        if delT <= max_gap_allowed:
            block_ind[ii+1] = block_ind[ii];
        else:
            block_ind[ii+1] = block_ind[ii]+1;
    block_ids = np.zeros(invalid.size, dtype=np.int)
    
    tot_blocks = block_ind[-1]
    block_ids[good_ind] = block_ind
    
    return block_ids, tot_blocks

#%%
with pd.HDFStore(intensities_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

#%%

worms_per_frame = trajectories_data[['frame_number', 'int_map_id', 'worm_index_joined']].groupby('frame_number')

N_worms = worms_per_frame.agg({'int_map_id':'count'})['int_map_id']
N_worms_max = N_worms.max()

block_ids, tot_blocks = getBlocksIDs((N_worms != N_worms_max).values, max_gap_allowed = 1)




#%%
#good_frames = N_worms.index[N_worms == max_N_worms];

#jumps = np.where(np.diff(good_frames)!=1)[0]

#%%
#valid_groups = []
#current_group = []
#for n_frame, frame_group in worms_per_frame:
#    if n_frame in  good_frames:
#        if len(current_group) == 0:
#            current_group = np.sort(frame_group['worm_index_joined'].values)
#            valid_groups.append([])            
#            valid_groups[-1] = {wid:[] for wid in current_group}
#        
#        frame_group = frame_group.sort(['worm_index_joined'])
#        if np.all(current_group == frame_group['worm_index_joined'].values):
#            for _, row in current_group.itterrows:
#                valid_groups[-1][wid].append(row['int_map_id'])
            
#%%
#separate_traj = trajectories_data[trajectories_data['frame_number'].isin(good_frames)]
#%%

