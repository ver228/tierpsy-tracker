#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:31:24 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables

from tierpsy_features import get_timeseries_features, timeseries_columns, durations_columns

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name, TABLE_FILTERS
from tierpsy.helper.params import read_fps

def _h_get_timeseries_feats_table(features_file):
    timeseries_features = []
    fps = read_fps(features_file)
    
    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    #only use data that was skeletonized
    trajectories_data = trajectories_data[trajectories_data['skeleton_id']>=0]
    
    
    trajectories_data_g = trajectories_data.groupby('worm_index_joined')
    progress_timer = TimeCounter('')
    base_name = get_base_name(features_file)
    tot_worms = len(trajectories_data_g)
    def _display_progress(n):
            # display progress
        dd = " Calculating tierpsy features. Worm %i of %i done." % (n+1, tot_worms)
        print_flush(
            base_name +
            dd +
            ' Total time:' +
            progress_timer.get_time_str())
    
    _display_progress(0)
    
    
    with tables.File(features_file, 'r+') as fid:
        
        for gg in ['/timeseries_data', '/event_durations', '/timeseries_features']:
            if gg in fid:
                fid.remove_node(gg)
                
            
        feat_dtypes = [(x, np.float32) for x in timeseries_columns]
        feat_dtypes = [('worm_index', np.int32), ('timestamp', np.int32)] + feat_dtypes
        timeseries_features = fid.create_table(
                '/',
                'timeseries_data',
                obj = np.recarray(0, feat_dtypes),
                filters = TABLE_FILTERS)
        
        #deal with event_type that is string (pandas save df strings as 'O', but pytables does not like that)
        e_durations_dtypes = [(x, np.float32) if x != 'event_type' else (x, np.dtype('S16')) for x in durations_columns]
        e_durations_dtypes = [('worm_index', np.int32)] + e_durations_dtypes
        
        event_durations = fid.create_table(
                '/',
                'event_durations',
                obj = np.recarray(0, e_durations_dtypes),
                filters = TABLE_FILTERS)
        


        if '/food_cnt_coord' in fid:
            food_cnt = fid.get_node('/food_cnt_coord')[:]
        else:
            food_cnt = None
    
        for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
            with tables.File(features_file, 'r') as fid:
                skel_id = worm_data['skeleton_id'].values
                args = []
                for p in ('skeletons', 'widths', 'dorsal_contours', 'ventral_contours'):
                     dd = fid.get_node('/coordinates/' + p)
                     if len(dd.shape) == 3:
                         args.append(dd[skel_id, :, :])
                     else:
                         args.append(dd[skel_id, :])
                    
            feats, e_dur = get_timeseries_features(*args, 
                                                   timestamp = worm_data['timestamp_raw'].values,
                                                   food_cnt = food_cnt,
                                                   fps = fps
                                                   )
            
            #save timeseries features data
            feats = feats.astype(np.float32)
            feats['worm_index'] = np.int32(worm_index)
            #move the last fields to the first columns
            cols = feats.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            feats = feats[cols]
            timeseries_features.append(feats.to_records(index=False))
            
            #save the event duration data
            e_dur['worm_index'] = np.int32(worm_index)
            #move the last fields to the first columns
            cols = e_dur.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            
            dd = e_dur[cols].to_records(index=False).astype(e_durations_dtypes)
            event_durations.append(dd)

            _display_progress(ind_n+1)
#%%            
def get_tierpsy_features(features_file):
    #I am adding this so if I add the parameters to calculate the features i can pass it to this function
    _h_get_timeseries_feats_table(features_file)
#%%    
if __name__ == '__main__':
    #base_file = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/gpa-10(pk362)V@NL1147/food_OP50/XX/30m_wait/clockwise/gpa-10 (pk362)V on food L_2009_07_16__12_55__4'
    #base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/WT2/Results/WT2'
    #base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/AVI_VIDEOS/Results/AVI_VIDEOS_4'
    #base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/GECKO_VIDEOS/Results/GECKO_VIDEOS'
    #base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/RIG_HDF5_VIDEOS/Results/RIG_HDF5_VIDEOS'
    #base_file = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm/MY16_worms5_food1-10_Set5_Pos4_Ch1_02062017_131004'
    #base_file = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm/170817_matdeve_exp7co1_12_Set0_Pos0_Ch1_17082017_140001'
    #features_file = base_file + '_featuresN.hdf5'
    #is_WT2 = False
    
    import glob
    import os
    #save_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/'
    save_dir = '/Users/ajaver/OneDrive - Imperial College London/swiss_strains'
    
    features_files = glob.glob(os.path.join(save_dir, '**', '*_featuresN.hdf5'), recursive=True)
    for ii, features_file in enumerate(features_files):
        print(ii+1, len(features_files))
        #with tables.File(features_file, 'r+') as fid:
        #    for gg in ['/provenance_tracking/FEAT_TIERPSY', '/timeseries_features']:
        #        if gg in fid:
        #            fid.remove_node(gg)
            
        get_tierpsy_features(features_file)
        break