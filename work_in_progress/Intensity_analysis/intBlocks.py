# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:49:25 2015

@author: ajaver
"""
import pandas as pd

import sys
sys.path.append('../../') # #MWTracker path

from MWTracker.trackWorms.checkHeadOrientation import getBlocksIDs

#getBlocksIDs(invalid, max_gap_allowed = 10)


if __name__ == '__main__':
    base_name = 'Capture_Ch1_18062015_140908'
    mask_dir = '/Users/ajaver/Google Drive/MWTracker_Example/MaskedVideos/'    
    results_dir = '/Users/ajaver/Google Drive/MWTracker_Example/Results/'    
    
    #directory with the masked and results files
    #mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
    #results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    
    
    masked_image_file = mask_dir + base_name + '.hdf5'
    trajectories_file = results_dir + base_name + '_trajectories.hdf5'
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    intensities_file = results_dir + base_name + '_intensities.hdf5'

    bad_seg_thresh = 0.5
    
    size_avg = 50
    #max_size_avg = 100
    
    
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        #data to extract the ROI
        trajectories_df = ske_file_id['/trajectories_data']
        
        #filter low skeletonized trajectories (typically problematic or bad)
        skeleton_fracc = trajectories_df[['worm_index_joined', 'has_skeleton']].groupby('worm_index_joined').agg('mean')
        skeleton_fracc = skeleton_fracc['has_skeleton']
        valid_worm_index = skeleton_fracc[skeleton_fracc>=bad_seg_thresh].index
        trajectories_df = trajectories_df[trajectories_df['worm_index_joined'].isin(valid_worm_index)]
        
        for worm_index, dat in trajectories_df.groupby('worm_index_joined'):
            block_ids, tot_blocks = getBlocksIDs(~dat['has_skeleton'])
            
            for bb in range(tot_blocks):
                skeleton_id = dat['skeleton_id'][block_ids==bb+1]
                print(worm_index, bb+1, skeleton_id.size)
                
                if skeleton_id.size >= size_avg:
                    bot_ids = skeleton_id[:size_avg]
                    aa = (worm_index, bb+1, 0, bot_ids.size, bot_ids)
                
                if skeleton_id.size >= 2*size_avg:
                    top_ids = skeleton_id[-size_avg:]
                    aa = (worm_index, bb+1, 2, top_ids.size, top_ids)
                
                if skeleton_id.size >= 3*size_avg:
                    m2 = round(skeleton_id.size/2)
                    delta_size = round(size_avg/2)
                    mid_ids = skeleton_id[m2-delta_size:m2+delta_size]
                    
                    aa = (worm_index, bb+1, 1, mid_ids.size, mid_ids)
                
                
                    
                
        #get the first and last frame of each worm_index
        #indexes_data = trajectories_df[['worm_index_joined', 'skeleton_id']]
        #rows_indexes = indexes_data.groupby('worm_index_joined').agg([min, max])['skeleton_id']
        #del indexes_data