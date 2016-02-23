# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:46:36 2016

@author: ajaver
"""



if __name__ == '__main__':
    #base directory
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch3_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
    
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')
    
    #get the trajectories table
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        #at this point the int_map_id with the intensity maps indexes must exist in the table
        assert 'int_map_id' in trajectories_data
    
    grouped_trajectories = trajectories_data.groupby('worm_index_joined')

    tot_worms = len(grouped_trajectories)
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    progress_timer = timeCounterStr('');
    
    
    bad_worms = [] #worms with not enough difference between the normal and inverted median intensity profile
    switched_blocks = [] #data from the blocks that were switched
    
    for index_n, (worm_index, trajectories_worm) in enumerate(grouped_trajectories):
        if index_n % 10 == 0:
            dd = " Correcting Head-Tail using intensity profiles. Worm %i of %i." % (index_n+1, tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print_flush(dd)
        
        good = trajectories_worm['int_map_id']>0;
        int_map_id = trajectories_worm.loc[good, 'int_map_id'].values
        int_skeleton_id = trajectories_worm.loc[good, 'skeleton_id'].values
        int_frame_number = trajectories_worm.loc[good, 'frame_number'].values
        #only analyze data that contains at least  min_block_size intensity profiles     
        if int_map_id.size < min_block_size:
            continue
        
        
        #read the worm intensity profiles
        with tables.File(intensities_file, 'r') as fid:
            worm_int_profile = fid.get_node('/straighten_worm_intensity_median')[int_map_id,:]