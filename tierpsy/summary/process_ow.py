#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:30:17 2018

@author: avelinojaver
"""
from tierpsy.summary.helper import augment_data, add_trajectory_info
from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormStats
from tierpsy.helper.params import read_fps
import pandas as pd
import tables
import numpy as np
#%%
def read_feat_events(fname, traj2read = None):
    
    def _feat_correct(x):
        x = x.replace('_motion', '')
        dat2rep =  [('_upsilon', '_upsilon_turns'),
                    ('upsilon_turn', 'upsilon_turns'),
                    ('_omega', '_omega_turns'),
                    ('omega_turn', '_omega_turns'),
                    ('coil_', 'coils_'),
                    ]
        for p_old, p_new in dat2rep:
            if not (p_new in x) and (p_old in x):
                x = x.replace(p_old, p_new)
        return x
    
    
    with tables.File(fname, 'r') as fid:
        features_events = {}
        node = fid.get_node('/features_events')
        
        if traj2read is None:
            traj2read = node._v_children.keys()
        
        
        
        for worn_n in traj2read:
            worm_node = fid.get_node('/features_events/' + worn_n)
            
            for feat in worm_node._v_children.keys():
                
                dat = fid.get_node(worm_node._v_pathname, feat)[:]
                
                
                feat_c = _feat_correct(feat)
                
                if not feat_c in features_events:
                    features_events[feat_c] = []
                features_events[feat_c].append(dat)
       
    features_events = {feat:np.concatenate(val) for feat, val in features_events.items()}
    
    return features_events
#%%
def ow_plate_summary(fname):
    all_feats = read_feat_events(fname)
    
    with pd.HDFStore(fname, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
    for cc in features_timeseries:
        all_feats[cc] = features_timeseries[cc].values
    
    wStats = WormStats()
    exp_feats = wStats.getWormStats(all_feats, np.nanmean)
    
    
    exp_feats = pd.DataFrame(exp_feats)
    
    valid_order = [x for x in exp_feats.columns if x not in wStats.extra_fields]
    exp_feats = exp_feats.loc[:, valid_order]
    
    return exp_feats
#%%
def ow_trajectories_summary(fname):
    
    fps = read_fps(fname)
    with pd.HDFStore(fname, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
    
    all_summary = []
    
    valid_order = None
    wStats = WormStats()
    for w_ind, w_ts_data in features_timeseries.groupby('worm_index'):
        
        ll = ['worm_{}'.format(int(w_ind))]
        all_feats = read_feat_events(fname, ll)
        for cc in w_ts_data:
            all_feats[cc] = w_ts_data[cc].values
        
        
        exp_feats = wStats.getWormStats(all_feats, np.nanmean)
        exp_feats = pd.DataFrame(exp_feats)
        
        if valid_order is None:
            #only calculate this the first time...
            valid_order = [x for x in exp_feats.columns if x not in wStats.extra_fields]
        
        #remove uncalculated indexes from wStats
        exp_feats = exp_feats.loc[:, valid_order]
        assert not 'worm_index' in exp_feats
        
        exp_feats = add_trajectory_info(exp_feats, w_ind, w_ts_data, fps)
        
        
        all_summary.append(exp_feats)
    all_summary = pd.concat(all_summary, ignore_index=True)

    return all_summary
#%%
def ow_plate_summary_augmented(fname, **fold_args):
    #NOTE: I will only augment the timeseries features. 
    #It is not trivial to include the event features sampling over time.
    
    fps = read_fps(fname)
    with pd.HDFStore(fname, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
    
    fold_index = augment_data(features_timeseries, fps=fps, **fold_args)
    
    valid_order = None
    wStats = WormStats()
    
    all_summary = []
    for i_fold, ind_fold in enumerate(fold_index):
        timeseries_data_r = features_timeseries[ind_fold].reset_index(drop=True)
        
        
        all_feats = {}
        for cc in timeseries_data_r:
            all_feats[cc] = timeseries_data_r[cc].values
        exp_feats = wStats.getWormStats(all_feats, np.nanmean)
        exp_feats = pd.DataFrame(exp_feats)
        
        if valid_order is None:
            #only calculate this the first time...
            valid_order = [x for x in exp_feats.columns if x not in wStats.extra_fields]
        exp_feats = exp_feats.loc[:, valid_order]
        
        exp_feats.insert(0, 'i_fold', i_fold)
        
        
        all_summary.append(exp_feats)
    
    all_summary = pd.concat(all_summary, ignore_index=True)
   
    return all_summary

#%%


