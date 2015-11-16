# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:07:42 2015

@author: ajaver
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%%

for chN in [1]:#range(1,7):
    features_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150612_1430/Capture_Ch%i_12062015_142858_features.hdf5' % chN
    
    feat_fid = pd.HDFStore(features_file, 'r')
    
    
    
    avg_feats = feat_fid['/features_means']
    
    worm_index = avg_feats['worm_index'].values.astype(np.int)
    
    feats_motion = {}
    for ii in worm_index:
        worm_id = 'worm_%i' % ii
        feats_motion[worm_id] = feat_fid['/features/' + worm_id + '/locomotion']
        
    #%%
    #worm_lengths = pd.DataFrame()
    plt.figure()
    for worm_id in feats_motion:
        feat_name = 'length'
        feat_name2 = 'midbody_speed'#'midbody_width'#'midbody_speed'#'area'#'length'#
        ll = feats_motion[worm_id][['frame_number', feat_name, feat_name2]]
        good = (ll['frame_number']>150000) &  (ll['frame_number']<155000)
        ll = ll.loc[good]        
        
        
        avg_period = 60*90*25
        worm_l = pd.rolling_median(ll[feat_name], avg_period, min_periods = avg_period/2)
        worm_b = pd.rolling_median(ll[feat_name2], avg_period, min_periods = avg_period/2)
        #worm_l = ll[feat_name]
        
        plt.plot(worm_l, worm_b, '.')
        #plt.plot(ll[feat_name], ll[feat_name2], '.')
        #plt.plot(ll['frame_number'], worm_b)
    plt.ylabel(feat_name2)
    plt.xlabel(feat_name)
    
    #%%
    fig = plt.figure()
    fig.set_size_inches([12, 4])
    for worm_id in list(feats_motion.keys())[0:1]:
        feat_name = 'length'
        feat_name2 = 'midbody_speed'#'midbody_width'#'midbody_speed'#'area'#'length'#
        ll = feats_motion[worm_id][['frame_number', feat_name]]
        #good = (ll['frame_number']>150000) &  (ll['frame_number']<155000)
        #ll = ll.loc[good]        
        
        
        avg_period = 5
        worm_l = pd.rolling_median(ll[feat_name], avg_period, min_periods = avg_period/2)
        #worm_b = pd.rolling_median(ll[feat_name2], avg_period, min_periods = avg_period/2)
        #worm_l = ll[feat_name]
        
        #plt.plot(worm_l, worm_b, '.')
        #plt.plot(ll[feat_name], ll[feat_name2], '.')
        plt.plot(ll['frame_number']/25, worm_l*9.5)
    plt.ylabel('Worm Length ($\mu$m)', fontsize=15)
    plt.xlabel('Time (s)', fontsize=15)
    plt.savefig('worm_length.png')
    
    #worm_lengths = worm_lengths.fillna(method = 'ffill')
    #worm_lengths = pd.rolling_median(worm_lengths, 25*60*10)
    #worm_lengths.plot()