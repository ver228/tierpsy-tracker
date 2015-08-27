# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:07:42 2015

@author: ajaver
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%%

for chN in range(1,7):
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
        feat_name2 = 'midbody_bend_mean'#'midbody_width'#'midbody_speed'#'area'#'length'#
        ll = feats_motion[worm_id][['frame_number', feat_name, feat_name2]]
        good = (ll['frame_number']>150000) &  (ll['frame_number']<155000)
        ll = ll.loc[good]        
        ll = ll.dropna()
        worm_l = pd.rolling_median(ll[feat_name], 13)#, min_periods = 25*60)
        worm_b = pd.rolling_median(ll[feat_name2], 13)#, min_periods = 25*60)
        #worm_l = ll[feat_name]
        
        worm_b =         
        #plt.plot(ll[feat_name], ll[feat_name2], '.')
        #plt.plot(ll['frame_number'], worm_l)
    plt.ylabel(feat_name)
    plt.xlabel(feat_name)
    
    #worm_lengths = worm_lengths.fillna(method = 'ffill')
    #worm_lengths = pd.rolling_median(worm_lengths, 25*60*10)
    #worm_lengths.plot()