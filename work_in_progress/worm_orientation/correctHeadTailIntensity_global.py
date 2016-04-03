# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:46:36 2016

@author: ajaver
"""
import pandas as pd
import os
import tables
import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.helperFunctions.timeCounterStr import timeCounterStr


if __name__ == '__main__':
    #base directory
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch3_17112015_205616.hdf5'
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
    
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results1')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')
    
    min_block_size = 1    
    
    #get the trajectories table
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        #at this point the int_map_id with the intensity maps indexes must exist in the table
        assert 'int_map_id' in trajectories_data

    trajectories_data = trajectories_data[trajectories_data['int_map_id']>0]
            
    
    grouped_trajectories = trajectories_data.groupby('worm_index_joined')

    tot_worms = len(grouped_trajectories)
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    progress_timer = timeCounterStr('');
    #%%
    with tables.File(intensities_file, 'r') as fid:
        resampling_length = fid.get_node('/straighten_worm_intensity_median').shape[1]
    #%%
    all_worm_profiles = np.zeros((tot_worms, resampling_length));
    all_worm_profiles_N = np.zeros(tot_worms);
    worm_index_ranges = OrderedDict()
    
    
    for index_n, (worm_index, trajectories_worm) in enumerate(grouped_trajectories):
        if index_n % 10 == 0:
            dd = " Getting median intensity profiles. Worm %i of %i." % (index_n+1, tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
        
        int_map_id = trajectories_worm['int_map_id'].values
        int_skeleton_id = trajectories_worm['skeleton_id'].values
        
        
        #read the worm intensity profiles
        with tables.File(intensities_file, 'r') as fid:
            worm_int_profile = fid.get_node('/straighten_worm_intensity_median')[int_map_id,:]
        
        #%%
        #normalize intensities of each individual profile 
        frame_med_int = np.median(worm_int_profile, axis=1);
        worm_int_profile = worm_int_profile - frame_med_int[:, np.newaxis]
        
        #worm median intensity
        median_profile = np.median(worm_int_profile, axis=0).astype(np.float)
        all_worm_profiles[index_n, :] = median_profile
        all_worm_profiles_N[index_n] = len(int_map_id)
        
        worm_index_ranges[worm_index] = {'skel_group' : (np.min(int_skeleton_id),np.max(int_skeleton_id)),
         'int_group' : (np.min(int_map_id),np.max(int_map_id))}
    
#%%
    average_profile = np.sum(all_worm_profiles*all_worm_profiles_N[:,np.newaxis], axis=0)/np.sum(all_worm_profiles_N)

    #diff_ori = np.sum(np.abs(all_worm_profiles-average_profile), axis=1)
    #diff_inv = np.sum(np.abs(all_worm_profiles-average_profile[::-1]), axis=1)