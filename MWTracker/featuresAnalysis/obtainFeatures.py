# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:53 2015

@author: ajaver
"""
import os, sys
#import sys
import tables
import pandas as pd
import numpy as np
from math import floor, ceil

import warnings
warnings.filterwarnings('ignore', '.*empty slice*',)
tables.parameters.MAX_COLUMNS = 1024 #(http://www.pytables.org/usersguide/parameter_files.html)

from collections import OrderedDict

from ..helperFunctions.timeCounterStr import timeCounterStr
from ..helperFunctions.miscFun import print_flush

from open_worm_analysis_toolbox import NormalizedWorm
from open_worm_analysis_toolbox import WormFeaturesDos, VideoInfo
from open_worm_analysis_toolbox.statistics import specifications

from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass, WormFromTable, getValidIndexes, WLAB

def getWormFeatures(skeletons_file, features_file, good_traj_index, \
    use_auto_label = True, use_manual_join = False):
    
    #we should get fps and pix2num values from the skeleton file (but first I need to store them there)
    pix2mum = 1
    fps = 25

    #get total number of valid worms and break if it is zero
    tot_worms = len(good_traj_index)

    #initialize by getting the specs data subdivision
    wStats = wormStatsClass()
    #list to save trajectories mean features
    all_stats = []
    
    #function to calculate the progress time. Useful to display progress 
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    progress_timer = timeCounterStr('');
    
    #filter used for each fo the tables
    filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
    
    with tables.File(features_file, 'w') as features_fid:

        group_feat_events = features_fid.create_group('/', 'features_events')
        
        #initialize timeseries table. All the features here are a numpy array having the same length as the worm trajectory
        feat_timeseries = wStats.features_info[wStats.features_info['is_time_series']==1].index.values;
        feat_events = wStats.features_info[wStats.features_info['is_time_series']==0].index.values;
        
        header_timeseries = {'worm_index':tables.Int32Col(pos=0),\
        'timestamp':tables.Int32Col(pos=1),\
        'motion_modes':tables.Float32Col(pos=2)}
        for ii, feat in enumerate(feat_timeseries):
            header_timeseries[feat] = tables.Float32Col(pos=ii+2)
        
        table_timeseries = features_fid.create_table('/', 'features_timeseries', header_timeseries, filters=filters_tables)
        
        #start to calculate features for each worm trajectory      
        for ind, worm_index  in enumerate(good_traj_index):
            #if ind <7: continue
            
            #initialize worm object, and extract data from skeletons file
            worm = WormFromTable(skeletons_file, worm_index, \
                use_auto_label = use_auto_label, use_manual_join = use_manual_join, \
                pix2mum = pix2mum, fps = fps, smooth_window = 5)
            
            #import pdb
            #pdb.set_trace()

            #check that the worm data is not only nan's
            if np.all(np.isnan(worm.length)):
                tot_worms = tot_worms - 1
                continue
            
            #save data as a subgroup for each worm
            worm_node = features_fid.create_group(group_feat_events, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_index'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.timestamp[0], worm.timestamp[-1])

            #save skeleton
            features_fid.create_carray(worm_node, 'skeletons', \
                                    obj = worm.skeleton, filters = filters_tables)
            
            #IMPORTANT change axis to an openworm format before calculating features
            worm.changeAxis()
            #OpenWorm feature calculation
            worm_features = WormFeaturesDos(worm)

            #get the average for each worm feature
            worm_stats = wStats.getWormStats(worm_features, np.mean)
            worm_stats['n_frames'] = worm.n_frames
            worm_stats['worm_index'] = worm_index
            worm_stats['n_valid_skel'] = worm.n_valid_skel
            for feat in ['n_valid_skel', 'n_frames', 'worm_index']:
                worm_stats.move_to_end(feat, last=False)
            all_stats.append(worm_stats)
            
            #save event features in each worm node
            for feat in feat_events:
                feat_obj = wStats.features_info.loc[feat, 'feat_name_obj']
                tmp_data = worm_features.features[feat_obj].value
                
                if isinstance(tmp_data, (float, int)): tmp_data = np.array([tmp_data])
                if tmp_data is None or tmp_data.size == 0: tmp_data = np.array([np.nan])
                
                table_tmp = features_fid.create_carray(worm_node, feat, \
                                    obj = tmp_data, filters = filters_tables)

            #save the timeseries data as a general table
            timeseries_data = [[]]*len(header_timeseries)
            timeseries_data[header_timeseries['timestamp']._v_pos] = worm.timestamp
            timeseries_data[header_timeseries['worm_index']._v_pos] = np.full(worm.n_frames, worm.worm_index, dtype=np.int64)
            timeseries_data[header_timeseries['motion_modes']._v_pos] = worm_features._temp_features['locomotion.motion_mode'].value
            
            for feat in feat_timeseries:
                feat_obj = wStats.features_info.loc[feat, 'feat_name_obj']
                tmp_data = worm_features.features[feat_obj].value
                timeseries_data[header_timeseries[feat]._v_pos] = tmp_data
            
            timeseries_data = list(zip(*timeseries_data))
            table_timeseries.append(timeseries_data)
            table_timeseries.flush()
            del timeseries_data
            
            #report progress
            dd = " Extracting features. Worm %i of %i done." % (len(all_stats), tot_worms)
            print_flush(base_name + dd + ' Total time:' + progress_timer.getTimeStr())
        
        #assert data make sense    
        tot_rows = len(all_stats)
        assert tot_worms == tot_rows

        #create and save a table containing the averaged worm feature for each worm
        if tot_rows > 0:
            dtype = [(x, np.float32) for x in wStats.feat_avg_names]
            mean_features_df = np.recarray(tot_rows, dtype = dtype);
            
            for kk, row_dict in enumerate(all_stats):
                for key in row_dict:
                    mean_features_df[key][kk] = row_dict[key]
            feat_mean = features_fid.create_table('/', 'features_means', obj = mean_features_df, filters=filters_tables)
        else:
            #if no valid worms were selected create an empty table with only one column
            feat_mean = features_fid.create_table('/', 'features_means', {'worm_index' : tables.Int32Col(pos=0)}, filters=filters_tables)
        
        #flag and report a success finish
        feat_mean._v_attrs['has_finished'] = 1
        print_flush(base_name + ' Feature extraction finished: ' + progress_timer.getTimeStr())
        
def getWormFeaturesFilt(skeletons_file, features_file, use_auto_label, use_manual_join, feat_filt_param):
    assert (use_auto_label or use_manual_join) or feat_filt_param

    if not (use_manual_join or use_auto_label):
        #filter using the parameters in feat_filt_param
        dd = {x : feat_filt_param[x] for x in ['min_num_skel', 'bad_seg_thresh', 'min_dist']}
        good_traj_index, _ = getValidIndexes(skeletons_file, **dd)
        
    else:
        with pd.HDFStore(skeletons_file, 'r') as table_fid:
            trajectories_data = table_fid['/trajectories_data']
        
        if use_manual_join:
            #select tables that were manually labeled as worms
            good = trajectories_data['worm_label'] == WLAB['WORM']
            trajectories_data = trajectories_data[good]

        if use_auto_label:
            #select data that was labeld in FEAT_FILTER
            good = trajectories_data['auto_label'] == WLAB['GOOD_SKE']
            trajectories_data = trajectories_data[good]
        
        N = trajectories_data.groupby('worm_index_joined').agg({'has_skeleton':np.nansum})
        N = N[N>feat_filt_param['min_num_skel']].dropna()
        
        good_traj_index = N.index
        
    #calculate features
    getWormFeatures(skeletons_file, features_file, good_traj_index, \
            use_auto_label = use_auto_label, use_manual_join = use_manual_join)
