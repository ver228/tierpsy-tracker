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

from collections import OrderedDict

from MWTracker.helperFunctions.timeCounterStr import timeCounterStr
from MWTracker.helperFunctions.miscFun import print_flush

import open_worm_analysis_toolbox as mv

from MWTracker.featuresAnalysis.obtainFeaturesHelper import WormStatsClass, WormFromTable, getValidIndexes
from MWTracker.helperFunctions.miscFun import WLAB


def getWormFeatures(worm, wStats = []):
    
    if not isinstance(wStats, WormStatsClass):
        wStats = WormStatsClass()

    #IMPORTANT change axis to an openworm format before calculating features
    assert worm.skeleton.shape[2] == 2
    worm.changeAxis()
    #OpenWorm feature calculation
    assert worm.skeleton.shape[1] == 2
    worm_features = mv.WormFeatures(worm)

    #convert the timeseries features into a recarray
    tot_frames = worm.timestamp.size
    timeseries_data = np.full(tot_frames, np.nan, wStats.feat_timeseries_dtype)

    timeseries_data['timestamp'] = worm.timestamp
    timeseries_data['worm_index'] = worm.worm_index
    timeseries_data['motion_modes'] = worm_features._features['locomotion.motion_mode'].value

    for feat in wStats.feat_timeseries:
        feat_obj = wStats.features_info.loc[feat, 'feat_name_obj']
        timeseries_data[feat] = worm_features._features[feat_obj].value

    #convert the events features into a dictionary
    events_data = {}
    for feat in wStats.feat_events:
        feat_obj = wStats.features_info.loc[feat, 'feat_name_obj']
        events_data[feat] = worm_features._features[feat_obj].value

    #calculate the mean value of each feature
    worm_stats = wStats.getWormStats(worm_features, np.mean)
    worm_stats['n_frames'] = worm.n_frames
    worm_stats['worm_index'] = worm_index
    worm_stats['n_valid_skel'] = worm.n_valid_skel


    return timeseries_data, events_data, worm_stats

def getFPS(skeletons_file, fps_expected):
        #try to infer the fps from the timestamp
    try:
        with tables.File(skeletons_file, 'r') as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
            fps = 1/np.median(np.diff(timestamp_time))
            if np.isnan(fps): 
                raise ValueError
            is_default_timestamp = 0
    except (tables.exceptions.NoSuchNodeError, IOError, ValueError):
        fps = fps_expected
        is_default_timestamp = 1
    
    return fps, is_default_timestamp

#%% these function are related with the singleworm case it might be necesary to change them in the future
def getMicroPerPixel(skeletons_file):
    try:
        with tables.File(skeletons_file, 'r') as fid:
            return fid.get_node('/stage_movement')._v_sattrs['pixel_per_micron_scale']
    except (tables.exceptions.NoSuchNodeError, IOError, KeyError):
            #i need to change it to something better, but for the momement let's use 1 as default
            return 1

def isValidSingleWorm(skeletons_file, good_traj_index):
    #check if it is sigle worm and if the stage movement was aligned
    try:
        with tables.File(skeletons_file, 'r') as fid:
            if fid.get_node('/stage_movement')._v_attrs['has_finished'] != 1:
                #single worm case with a bad flag termination in the stage movement
                #return
                bad
            else:
                assert len(good_traj_index) == 1
                is_single_worm = True
    except (tables.exceptions.NoSuchNodeError, IOError, KeyError):
        is_single_worm = False
    return is_single_worm


def correctSingleWorm(worm, skeletons_file):
    with tables.File(skeletons_file, 'r') as fid:
        stage_vec_ori = fid.get_node('/stage_movement/stage_vec')[:]
        timestamp_ind = fid.get_node('/timestamp/raw')[:].astype(np.int)
        rotation_matrix = fid.get_node('/stage_movement')._v_attrs['rotation_matrix']

    #adjust the stage_vec to match the timestamps in the skeletons
    timestamp_ind = timestamp_ind - worm.first_frame
    good = (timestamp_ind>=0) & (timestamp_ind<=worm.timestamp[-1])
    
    timestamp_ind = timestamp_ind[good]
    stage_vec_ori = stage_vec_ori[good]
    
    stage_vec = np.full((worm.timestamp.size,2), np.nan)
    stage_vec[timestamp_ind, :] = stage_vec_ori
    
    tot_skel = worm.skeleton.shape[0]
    
    for field in ['skeleton', 'ventral_contour', 'dorsal_contour']:
        tmp_dat = getattr(worm, field)
        for ii in range(tot_skel):
            tmp_dat[ii] = np.dot(tmp_dat[ii], rotation_matrix) - stage_vec[ii]
        setattr(worm, field, tmp_dat)
    return worm


#def getWormFeatures(skeletons_file, features_file, good_traj_index, \
#    use_skel_filter = True, use_manual_join = False):

if __name__ == '__main__':
    skeletons_file = '/Users/ajaver/Desktop/Videos/single_worm/agar_1/Results/431 JU298 on food L_2011_03_17__12_02_58___2___3_skeletons.hdf5'
    features_file = 'dum.hdf5'
    good_traj_index = [1]
    use_skel_filter = True
    use_manual_join = False
    
    fps_expected = 25
    fps, is_default_timestamp = getFPS(skeletons_file, fps_expected)
    micronsPerPixel = pix2mum = getMicroPerPixel(skeletons_file)
    
    is_single_worm = isValidSingleWorm(skeletons_file, good_traj_index)

    #function to calculate the progress time. Useful to display progress 
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    #filter used for each fo the tables
    filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
    #Time series
    progress_timer = timeCounterStr('');
    
    #get total number of valid worms and break if it is zero
    tot_worms = len(good_traj_index)
    #initialize by getting the specs data subdivision
    wStats = WormStatsClass()
    #initialize rec array with the averaged features of each worm
    mean_features_df = np.full(tot_worms, np.nan, dtype=wStats.feat_avg_dtype)
    with tables.File(features_file, 'w') as features_fid:
        #initialize groups for the timeseries and event features
        group_feat_events = features_fid.create_group('/', 'features_events')
        header_timeseries = {feat:tables.Float32Col(pos=ii) for ii, (feat,_) in enumerate(wStats.feat_timeseries_dtype)}
        table_timeseries = features_fid.create_table('/', 'features_timeseries', header_timeseries, filters=filters_tables)
        
        #start to calculate features for each worm trajectory
        for ind_N, worm_index  in enumerate(good_traj_index):
            #initialize worm object, and extract data from skeletons file
            worm = WormFromTable(skeletons_file, worm_index, \
                use_skel_filter = use_skel_filter, use_manual_join = use_manual_join, \
                micronsPerPixel = micronsPerPixel,fps=fps, smooth_window = 5)
            
            if is_single_worm:
                assert worm_index == 1 and ind_N == 0
                worm = correctSingleWorm(worm, skeletons_file)

            #check that the worm data is not only nan's
            if np.all(np.isnan(worm.length)):
                tot_worms = tot_worms - 1
                continue

            #let's make a copy of the skeletons before chaning axis
            skeletons = worm.skeleton.copy()

            #calculate features
            timeseries_data, events_data, worm_stats = getWormFeatures(worm, wStats)

            #%%
            #save timeseries data
            table_timeseries.append(timeseries_data)
            table_timeseries.flush()
            
            #save event data as a subgroup per worm
            worm_node = features_fid.create_group(group_feat_events, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_index'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.timestamp[0], worm.timestamp[-1])
            for feat in events_data:
                tmp_data = events_data[feat]
                #consider the cases where the output is a single number, empty or None
                if isinstance(tmp_data, (float, int)): tmp_data = np.array([tmp_data])
                if tmp_data is None or tmp_data.size == 0: tmp_data = np.array([np.nan])
                features_fid.create_carray(worm_node, feat, \
                                    obj = tmp_data, filters = filters_tables)

            #save the skeletons in the same group
            features_fid.create_carray(worm_node, 'skeletons', \
                                    obj = skeletons, filters = filters_tables)

            #store the average for each worm feature
            mean_features_df[ind_N] = worm_stats
            #%%
            #report progress
            dd = " Extracting features. Worm %i of %i done." % (ind+1, tot_worms)
            print_flush(base_name + dd + ' Total time:' + progress_timer.getTimeStr())

        #create and save a table containing the averaged worm feature for each worm
        feat_mean = features_fid.create_table('/', 'features_means', obj = mean_features_df, filters=filters_tables)
        
        #flag and report a success finish
        feat_mean._v_attrs['has_finished'] = 1
        print_flush(base_name + ' Feature extraction finished: ' + progress_timer.getTimeStr())
