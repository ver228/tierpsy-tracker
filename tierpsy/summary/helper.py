#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:45:29 2018

@author: avelinojaver
"""
from tierpsy.features.tierpsy_features.summary_stats import get_n_worms_estimate
from tierpsy.analysis.split_fov.helper import was_fov_split

import random
import math
import pdb

fold_args_dflt = {'n_folds' : 5, 
                 'frac_worms_to_keep' : 0.8,
                 'time_sample_seconds' : 10*60
                 }

def add_trajectory_info(df_stats, worm_index, timeseries_data, fps):
    df_stats['worm_index'] = worm_index
    df_stats['ini_time'] = timeseries_data['timestamp'].min()/fps
    df_stats['tot_time'] = timeseries_data['timestamp'].size/fps
    df_stats['frac_valid_skels'] = (~timeseries_data['length'].isnull()).mean()
    
    is_fov_tosplit = was_fov_split(timeseries_data)
    if is_fov_tosplit:
        try:
            assert len(set(timeseries_data['well_name']) - set(['n/a'])) == 1, \
        "A single trajectory is spanning more than one well!"
        except:
            pdb.set_trace()
        well_name = list(set(timeseries_data['well_name']) - set(['n/a']))[0]
        df_stats['well_name'] = well_name
    
    cols = df_stats.columns.tolist()   
    if not is_fov_tosplit:
        cols = cols[-4:] + cols[:-4]
    else: # there's one extra column
        cols = cols[-5:] + cols[:-5]


    #import pdb
    #pdb.set_trace()    
    df_stats = df_stats[cols]
    
    
    return df_stats
    

def augment_data(df,
                 n_folds = 10, 
                 frac_worms_to_keep = 0.8,
                 time_sample_seconds = 600,
                 fps = 25
                 ):
    
    timestamp = df['timestamp']
    worm_index = df['worm_index']
    
    #fraction of trajectories to keep. I want to be proportional to the number of worms present.
    n_worms_estimate = get_n_worms_estimate(timestamp)
    frac_worms_to_keep_r = min(frac_worms_to_keep, (1-1/n_worms_estimate))
    if frac_worms_to_keep_r <= 0:
        #if this fraction is really zero, just keep everything
        frac_worms_to_keep_r = 1
    
    time_sample_frames = time_sample_seconds*fps
    
    ini_ts = timestamp.min()
    last_ts = timestamp.max()
    
    ini_sample_last = max(ini_ts, last_ts - time_sample_frames)
    
    fold_masks = []
    for i_fold in range(n_folds):
        ini = random.randint(ini_ts, ini_sample_last)
        fin = ini + time_sample_frames
        
        good = (timestamp >= ini) & (timestamp<= fin) 
        
        #select only a small fraction of the total number of trajectories present
        ts_sampled_worm_idxs = worm_index[good]
        available_worm_idxs = ts_sampled_worm_idxs.unique()
        random.shuffle(available_worm_idxs)
        n2select = math.ceil(len(available_worm_idxs)*frac_worms_to_keep_r)
        idx2select = available_worm_idxs[:n2select]
        
        good = good & ts_sampled_worm_idxs.isin(idx2select)
        
        fold_masks.append(good)
        
    return fold_masks
