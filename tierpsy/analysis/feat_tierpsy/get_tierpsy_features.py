#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:31:24 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables

from tierpsy.features.tierpsy_features import get_timeseries_features, timeseries_all_columns
from tierpsy.features.tierpsy_features.summary_stats import get_summary_stats

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name, TABLE_FILTERS
from tierpsy.helper.params import read_fps, read_ventral_side

def save_timeseries_feats_table(features_file, derivate_delta_time):
    timeseries_features = []
    fps = read_fps(features_file)
    
    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    #only use data that was skeletonized
    #trajectories_data = trajectories_data[trajectories_data['skeleton_id']>=0]
    
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
                
        
        feat_dtypes = [(x, np.float32) for x in timeseries_all_columns]
        

        feat_dtypes = [('worm_index', np.int32), ('timestamp', np.int32)] + feat_dtypes
        timeseries_features = fid.create_table(
                '/',
                'timeseries_data',
                obj = np.recarray(0, feat_dtypes),
                filters = TABLE_FILTERS)
        
        if '/food_cnt_coord' in fid:
            food_cnt = fid.get_node('/food_cnt_coord')[:]
        else:
            food_cnt = None
    
        #If i find the ventral side in the multiworm case this has to change
        ventral_side = read_ventral_side(features_file)
            
        for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
            with tables.File(features_file, 'r') as fid:
                skel_id = worm_data['skeleton_id'].values
                
                #deal with any nan in the skeletons
                good_id = skel_id>=0
                skel_id_val = skel_id[good_id]
                traj_size = skel_id.size

                args = []
                for p in ('skeletons', 'widths', 'dorsal_contours', 'ventral_contours'):
                    
                    node_str = '/coordinates/' + p
                    if node_str in fid:
                        node = fid.get_node(node_str)
                        dat = np.full((traj_size, *node.shape[1:]), np.nan)
                        if skel_id_val.size > 0:
                            if len(node.shape) == 3:
                                dd = node[skel_id_val, :, :]
                            else:
                                dd = node[skel_id_val, :]
                            dat[good_id] = dd
                    else:
                        dat = None
                    
                    args.append(dat)

                timestamp = worm_data['timestamp_raw'].values.astype(np.int32)
            
            feats = get_timeseries_features(*args, 
                                           timestamp = timestamp,
                                           food_cnt = food_cnt,
                                           fps = fps,
                                           ventral_side = ventral_side,
                                           derivate_delta_time = derivate_delta_time
                                           )
            #save timeseries features data
            feats = feats.astype(np.float32)
            feats['worm_index'] = worm_index
            #move the last fields to the first columns
            cols = feats.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            feats = feats[cols]
            
            feats['worm_index'] = feats['worm_index'].astype(np.int32)
            feats['timestamp'] = feats['timestamp'].astype(np.int32)
            feats = feats.to_records(index=False)
    
            timeseries_features.append(feats)
            _display_progress(ind_n)

def save_feats_stats(features_file, derivate_delta_time):
    with pd.HDFStore(features_file, 'r') as fid:
        fps = fid.get_storer('/trajectories_data').attrs['fps']
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features'] if '/blob_features' in fid else None
    
    
    #Now I want to calculate the stats of the video
    exp_feats = get_summary_stats(timeseries_data, 
                      fps,  
                      blob_features, 
                      derivate_delta_time)
    
    if len(exp_feats)>0:
        tot = max(len(x) for x in exp_feats.index)
        dtypes = [('name', 'S{}'.format(tot)), ('value', np.float32)]
        exp_feats_rec = np.array(list(zip(exp_feats.index, exp_feats)), dtype = dtypes)
        with tables.File(features_file, 'r+') as fid:
            for gg in ['/features_stats']:
                if gg in fid:
                    fid.remove_node(gg)
            fid.create_table(
                    '/',
                    'features_stats',
                    obj = exp_feats_rec,
                    filters = TABLE_FILTERS)    


            
def get_tierpsy_features(features_file, derivate_delta_time = 1/3):
    #I am adding this so if I add the parameters to calculate the features i can pass it to this function
    save_timeseries_feats_table(features_file, derivate_delta_time)
    save_feats_stats(features_file, derivate_delta_time)
    

        