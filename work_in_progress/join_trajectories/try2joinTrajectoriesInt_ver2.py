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

from scipy.signal import savgol_filter
from scipy.signal import medfilt

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
    
    #this is small window that reduce the values on the head a tail, where a segmentation error or noise can have a very big effect
    rr = (np.arange(20)/19)*0.9 + 0.1
    damp_factor = np.ones(131);
    damp_factor[:20] = rr
    damp_factor[-20:] = rr[::-1]
    
    
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
    
    
    valid_data = OrderedDict()
    for index_n, (worm_index, trajectories_worm) in enumerate(grouped_trajectories):
        if index_n % 10 == 0:
            dd = " Correcting Head-Tail using intensity profiles. Worm %i of %i." % (index_n+1, tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
        
        
        int_map_id = trajectories_worm['int_map_id'].values
        first_frame = trajectories_worm['frame_number'].min()
        last_frame = trajectories_worm['frame_number'].max()
        
        
        #only analyze data that contains at least  min_block_size intensity profiles     
        if int_map_id.size < min_block_size:
            continue
        
        
        #get initial and final skeletons mid points
        segworm_id_ini = trajectories_worm['frame_number'].argmin()        
        segworm_id_fin = trajectories_worm['frame_number'].argmax()   
        
        with tables.File(skeletons_file, 'r') as fid:
            mid_point = 25
            first_coord = fid.get_node('/skeleton')[segworm_id_ini, mid_point, :]
            last_coord = fid.get_node('/skeleton')[segworm_id_fin, mid_point, :]
        
        
        #read the worm intensity profiles
        with tables.File(intensities_file, 'r') as fid:
            worm_int_profile = fid.get_node('/straighten_worm_intensity_median')[int_map_id,:]
        
        #reduce the importance of the head and tail
        worm_int_profile *= damp_factor        
        
        #normalize intensities of each individual profile 
        frame_med_int = np.median(worm_int_profile, axis=1);
        worm_int_profile = worm_int_profile - frame_med_int[:, np.newaxis]
        #worm median intensity
        median_profile = np.median(worm_int_profile, axis=0).astype(np.float)
        
        #save data into a dictionary
        valid_data[worm_index] = {'median_profile' : median_profile, 
        'int_map_id' : int_map_id, 'frame_med_int' : frame_med_int, 
        'frame_range' : (first_frame,last_frame), 'coord_limits':(first_coord, last_coord)}

    #get cost matrixes
    valid_worm_index = list(valid_data.keys())    
    valid_worm_order = {x:n for n,x in enumerate(valid_worm_index)}
    tot_valid_worms = len(valid_worm_index)
    
    cost_int_match = np.full((tot_valid_worms, tot_valid_worms), np.nan)    
    cost_dist_before = np.full((tot_valid_worms, tot_valid_worms), np.nan) 
    cost_dist_after = np.full((tot_valid_worms, tot_valid_worms), np.nan) 
    
    cost_time_before = np.full((tot_valid_worms, tot_valid_worms), np.nan) 
    cost_time_after = np.full((tot_valid_worms, tot_valid_worms), np.nan) 
    
    for worm_index in valid_data:
        
        #first and last valid frame of the trajectory
        first_frame = valid_data[worm_index]['frame_range'][0]
        last_frame = valid_data[worm_index]['frame_range'][1]
        
        #filter pausible indexes
        other_worms_ind = valid_worm_index[:]
        other_worms_ind.remove(worm_index)
        
        #trajectories that do not overlap with the current one
        before_worm_ind =   [x for x in other_worms_ind if (valid_data[x]['frame_range'][1] < first_frame)]
        after_worm_ind =   [x for x in other_worms_ind if (valid_data[x]['frame_range'][0] > last_frame)]
        
        other_worms_ind = before_worm_ind + after_worm_ind
        
        if len(other_worms_ind) == 0:
            continue
       
        
        #get the different between pausible averaged maps
        median_profile = valid_data[worm_index]['median_profile'].copy()
    
        other_worm_profile = np.zeros((median_profile.size, len(other_worms_ind)))
        for w_ii, w_ind in enumerate(other_worms_ind):
            other_worm_profile[:,w_ii] = valid_data[w_ind]['median_profile']
    
        
        DD_ori = other_worm_profile - median_profile[:, np.newaxis]
        DD_ori = np.mean(np.abs(DD_ori), axis=0)
        
        DD_inv = other_worm_profile - median_profile[::-1, np.newaxis]
        DD_inv = np.mean(np.abs(DD_inv), axis=0)
        DD_best = np.min((DD_ori, DD_inv), axis=0)
        
        
        wi1 = valid_worm_order[worm_index]
        wi2 = [valid_worm_order[x] for x in other_worms_ind]
        cost_int_match[wi1,wi2] = DD_best
        
        #get distance from the final point of other to the start point of current
        if len(before_worm_ind)>0:
            first_coord = valid_data[worm_index]['coord_limits'][0];
            dist_r = [valid_data[wi]['coord_limits'][1]-first_coord for wi in before_worm_ind]
            dist_r = [np.sqrt(np.sum(x*x)) for x in dist_r]
            
            
            del_t = [first_frame-valid_data[x]['frame_range'][1] for x in before_worm_ind]
            
            wi1 = valid_worm_order[worm_index]
            wi2 = [valid_worm_order[x] for x in before_worm_ind]
            cost_dist_before[wi1,wi2] = dist_r
            cost_time_before[wi1,wi2] = del_t
            
        #get distance from the final point of current to the start point of other
        if len(after_worm_ind)>0:
            last_coord = valid_data[worm_index]['coord_limits'][1];
            dist_r = [valid_data[wi]['coord_limits'][0] - first_coord for wi in after_worm_ind]
            dist_r = [np.sqrt(np.sum(x*x)) for x in dist_r]
            
            del_t = [valid_data[x]['frame_range'][0]-last_frame for x in after_worm_ind]
                        
            
            wi1 = valid_worm_order[worm_index]
            wi2 = [valid_worm_order[x] for x in after_worm_ind]
            cost_dist_after[wi1,wi2] = dist_r
            cost_time_after[wi1,wi2] = del_t
            
        #%%
    plt.figure()
    plt.imshow(np.isnan(cost_dist_after), interpolation='none')       
    
    plt.figure()
    plt.imshow(np.isnan(cost_dist_before), interpolation='none')       
        
    #%% get valid indexes from manually joined trajectories
    traj_groups = {}
    joined_indexes = {}
    dd = trajectories_data[trajectories_data['worm_label']==1]
    grouped_trajectories_N = dd.groupby('worm_index_N')
    for index_n, (worm_index, trajectories_worm) in enumerate(grouped_trajectories_N):
        joined_indexes[worm_index] = trajectories_worm['worm_index_joined'].unique()
        for wi in joined_indexes[worm_index]:
            traj_groups[wi] = worm_index

#%%
