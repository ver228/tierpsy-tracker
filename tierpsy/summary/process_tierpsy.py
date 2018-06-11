#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:30:17 2018

@author: avelinojaver
"""
from tierpsy.summary.helper import augment_data, add_trajectory_info
from tierpsy.helper.params import read_fps
from tierpsy_features.summary_stats import get_summary_stats
from tierpsy.helper.misc import WLAB

import pandas as pd


#%%
def read_data(fname, is_manual_index):
    with pd.HDFStore(fname, 'r') as fid:
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features']
        
        if is_manual_index:
            #keep only data labeled as worm or worm clusters
            valid_labels = [WLAB[x] for x in ['WORM', 'WORMS']]
            trajectories_data = fid['/trajectories_data']
            if not 'worm_index_manual' in trajectories_data:
                #no manual index, nothing to do here
                return
            
            good = trajectories_data['worm_label'].isin(valid_labels)
            good = good & (trajectories_data['skeleton_id'] >= 0)
            skel_id = trajectories_data['skeleton_id'][good]
            
            
            timeseries_data = timeseries_data.loc[skel_id]
            timeseries_data['worm_index'] = trajectories_data['worm_index_manual'][good].values
            timeseries_data = timeseries_data.reset_index(drop=True)
            
            blob_features = blob_features.loc[skel_id].reset_index(drop=True)
    
    return timeseries_data, blob_features
#%%    
def tierpsy_plate_summary(fname, is_manual_index = False, delta_time = 1/3):
    
    fps = read_fps(fname)
    data_in = read_data(fname, is_manual_index)
    if data_in is None:
        return
    timeseries_data, blob_features = data_in
    
    plate_feats = get_summary_stats(timeseries_data, fps,  blob_features, delta_time)
    plate_feats = pd.DataFrame(plate_feats).T
    
    return plate_feats

def tierpsy_trajectories_summary(fname, is_manual_index = False, delta_time = 1/3):
    fps = read_fps(fname)
    data_in = read_data(fname, is_manual_index)
    if data_in is None:
        return
    timeseries_data, blob_features = data_in
    
    all_summary = []
    for w_ind, w_ts_data in timeseries_data.groupby('worm_index'):
        w_blobs = blob_features.loc[w_ts_data.index]
    
        w_ts_data = w_ts_data.reset_index(drop=True)
        w_blobs = w_blobs.reset_index(drop=True)
        
        
        
        worm_feats = get_summary_stats(w_ts_data, fps,  w_blobs, delta_time)
        worm_feats = pd.DataFrame(worm_feats).T
        worm_feats = add_trajectory_info(worm_feats, w_ind, w_ts_data, fps)
        
        
        all_summary.append(worm_feats)
    all_summary = pd.concat(all_summary, ignore_index=True)
    return all_summary

#%%
    
def tierpsy_plate_summary_augmented(fname, is_manual_index = False, delta_time = 1/3, **fold_args):
    fps = read_fps(fname)
    data_in = read_data(fname, is_manual_index)
    if data_in is None:
        return
    timeseries_data, blob_features = data_in

    fold_index = augment_data(timeseries_data, fps=fps, **fold_args)
    all_summary = []
    for i_fold, ind_fold in enumerate(fold_index):
        
        
        timeseries_data_r = timeseries_data[ind_fold].reset_index(drop=True)
        blob_features_r = blob_features[ind_fold].reset_index(drop=True)
        
        
        plate_feats = get_summary_stats(timeseries_data_r, fps,  blob_features_r, delta_time)
        plate_feats = pd.DataFrame(plate_feats).T
        plate_feats.insert(0, 'i_fold', i_fold)
        
        all_summary.append(plate_feats)
    
    all_summary = pd.concat(all_summary, ignore_index=True)
   
    return all_summary
