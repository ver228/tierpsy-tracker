# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:35:04 2015

@author: ajaver
"""

import pandas as pd

import numpy as np
import os

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr


import matplotlib.pylab as plt

from WormClass import WormClass

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

    return angles, meanAngle

def calculateHeadTailAng(skeletons, segment4angle, good):
    angles_head = np.empty(skeletons.shape[0])
    angles_head.fill(np.nan)
    angles_tail = angles_head.copy()
    
    dx = skeletons[good,segment4angle,0] - skeletons[good,0,0]
    dy = skeletons[good,segment4angle,1] - skeletons[good,0,1]
    
    angles_head[good], _ = getAnglesDelta(dx,dy)
    
    dx = skeletons[good,-segment4angle-1,0] - skeletons[good,-1,0]
    dy = skeletons[good,-segment4angle-1,1] - skeletons[good,-1,1]
    
    angles_tail[good], _ = getAnglesDelta(dx,dy)
    return angles_head, angles_tail

def getBlocksIDs(invalid, max_gap_allowed = 10):
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

def isWormHTSwitched(skeletons, segment4angle = 5, max_gap_allowed = 10, \
                     window_std = 25, min_block_size=250):
    invalid = np.isnan(skeletons[:,0,0])
    block_ids, tot_blocks = getBlocksIDs(invalid, max_gap_allowed)
    angles_head, angles_tail = calculateHeadTailAng(skeletons, segment4angle, block_ids!=0)
        
    ts = pd.DataFrame({'head_angle':angles_head, 'tail_angle':angles_tail})
    
    roll_std = pd.rolling_std(ts, window = window_std, min_periods = window_std-max_gap_allowed);
    
    roll_std["is_head"] = (roll_std['head_angle']>roll_std['tail_angle'])
    roll_std["block_id"] = block_ids
    
    #this function will return nan if the number of elements in the group is less than min_block_size
    mean_relevant = lambda x: x.mean() if x.count() > min_block_size else np.nan
    head_prob = roll_std.groupby('block_id').agg({'is_head': mean_relevant})
    
    head_prob.loc[0] = np.nan
    #fill nan, forward with the last valid observation, and then first backward with the next valid observation
    head_prob = head_prob.fillna(method = 'ffill').fillna(method = 'bfill')
    
    is_switch_block = np.squeeze(head_prob.values)<0.5
    is_switch_skel = is_switch_block[block_ids]
    return is_switch_skel, roll_std

def correctHeadTail(skeletons_file, max_gap_allowed = 10, window_std = 25, \
    segment4angle = 5, min_block_size = 250):
    '''
    max_gap_allowed = 10 #maximimun number of consecutive skeletons lost before consider it a new block
    window_std = 25 #frame windows to calculate the standard deviation
    segment4angle = 5 #separation between skeleton segments to calculate the angles
    min_block_size = 250 #consider only around 10s intervals to determine if it is head or tail... 
    '''
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    #%%
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        indexes_data = ske_file_id['/trajectories_data'][['worm_index_joined', 'skeleton_id']]
        #get the first and last frame of each worm_index
        rows_indexes = indexes_data.groupby('worm_index_joined').agg([min, max])['skeleton_id']
        del indexes_data
    #%%
    progress_timer = timeCounterStr('');
    for ii, dat in enumerate(rows_indexes.iterrows()):
        if ii % 10 == 0:
            dd = " Correcting Head-Tail worm %i of %i." % (ii+1, len(rows_indexes))
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
        
        worm_index, row_range = dat        
        
        worm_data = WormClass(skeletons_file, worm_index, \
                    rows_range = (row_range['min'],row_range['max']))
        
        if np.any(~np.isnan(worm_data.skeleton_length)):
            is_switched_skel, roll_std = isWormHTSwitched(worm_data.skeleton, \
            segment4angle = segment4angle, max_gap_allowed = max_gap_allowed, \
            window_std = window_std, min_block_size=min_block_size)
            #plt.figure()
            #roll_std[['head_angle','tail_angle']].plot()
            
            worm_data.switchHeadTail(is_switched_skel)
        
        worm_data.writeData()
        #%%
    print('Finished:' + progress_timer.getTimeStr())
    
if __name__ == "__main__":
    #root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
    #base_name = 'Capture_Ch1_11052015_195105'
    #root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150512/'    
    #base_name = 'Capture_Ch3_12052015_194303'
    
    skeletons_file = root_dir + '/Trajectories/' + base_name + '_skeletons.hdf5'
    

    #correctHeadTail(skeletons_file, max_gap_allowed = 10, \
    #window_std = 25, segment4angle = 5, min_block_size = 250)
