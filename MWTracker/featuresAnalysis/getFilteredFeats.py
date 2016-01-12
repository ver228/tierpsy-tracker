# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 23:39:22 2015

@author: ajaver
"""

import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import warnings

from sklearn.covariance import EllipticEnvelope, MinCovDet
np.seterr(invalid='ignore') 

from .obtainFeatures import getWormFeatures

worm_partitions = {'neck': (8, 16),
                'midbody':  (16, 33),
                'hips':  (33, 41),
                # refinements of ['head']
                'head_tip': (0, 4),
                'head_base': (4, 8),    # ""
                # refinements of ['tail']
                'tail_base': (40, 45),
                'tail_tip': (45, 49)}

wlab = {'U':0, 'WORM':1, 'WORMS':2, 'BAD':3, 'GOOD_SKE':4}

name_width_fun = lambda part: 'width_' + part

def saveLabelData(skel_file, trajectories_data):
    trajectories_recarray = trajectories_data.to_records(index=False)
    with tables.File(skel_file, "r+") as ske_file_id:
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        newT = ske_file_id.create_table('/', 'trajectories_data_d', obj = trajectories_recarray, filters=table_filters)
        ske_file_id.remove_node('/', 'trajectories_data')
        newT.rename('trajectories_data')
    

def getValidIndexes(skel_file, min_num_skel = 100, bad_seg_thresh = 0.8, min_dist = 5):
    #min_num_skel - ignore trajectories that do not have at least this number of skeletons
    
    with pd.HDFStore(skel_file, 'r') as table_fid:
        trajectories_data = table_fid['/trajectories_data']
        trajectories_data =  trajectories_data[trajectories_data['worm_index_joined'] > 0]
        
        if len(trajectories_data['worm_index_joined'].unique()) == 1:
            good_skel_row = trajectories_data['skeleton_id'][trajectories_data.has_skeleton.values.astype(np.bool)].values
            return good_skel_row, trajectories_data
        
        #get the fraction of worms that were skeletonized per trajectory
        how2agg = {'has_skeleton':['mean', 'sum'], 'coord_x':['max', 'min', 'count'],
                   'coord_y':['max', 'min']}
        tracks_data = trajectories_data.groupby('worm_index_joined').agg(how2agg)
        
        delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
        delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
        
        max_avg_dist = np.sqrt(delX*delX + delY*delY)#/tracks_data['coord_x']['count']
        
        skeleton_fracc = tracks_data['has_skeleton']['mean']
        skeleton_tot = tracks_data['has_skeleton']['sum']
        
        good_worm = (skeleton_fracc>=bad_seg_thresh) & (skeleton_tot>=min_num_skel)
        good_worm = good_worm & (max_avg_dist>min_dist)
        
        good_row = (trajectories_data.worm_index_joined.isin(good_worm[good_worm].index)) \
        & (trajectories_data.has_skeleton.values.astype(np.bool))
        
        trajectories_data['auto_label'] = wlab['U']
        trajectories_data.loc[good_row, 'auto_label'] = wlab['WORM']
        
        good_skel_row = trajectories_data.loc[good_row, 'skeleton_id'].values
    
        return good_skel_row, trajectories_data

def read_field(fid, field, valid_index):
    data = fid.get_node(field)[:]
    data = data[valid_index]
    return data

def nodes2Array(skel_file, valid_index = np.zeros(0)):

    nodes4fit = ['/skeleton_length', '/contour_area'] + \
    ['/' + name_width_fun(part) for part in worm_partitions]

    with tables.File(skel_file, 'r') as fid:
        assert all(node in fid for node in nodes4fit)

        if len(valid_index) == 0:
            valid_index = np.arange(fid.get_node(nodes4fit[0]).shape[0])
            
        n_samples = len(valid_index)
        n_features = len(nodes4fit)
        
        X = np.zeros((n_samples, n_features))
        for ii, node in enumerate(nodes4fit):
            X[:,ii] = read_field(fid, node, valid_index)
        
        return X

def calculate_widths(skel_file):
    with tables.File(skel_file, 'r+') as fid:
        if any(not '/' + name_width_fun(part) in fid for part in worm_partitions):
            widths = fid.get_node('/contour_width')[:]
            tot_rows = widths.shape[0]
            
            table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
            
            for part in worm_partitions:
                pp = worm_partitions[part]
                widths_mean = np.mean(widths[:, pp[0]:pp[1]], axis=1)
                
                #fid.remove_node('/', name_width(part))
                fid.create_carray('/', name_width_fun(part), obj = widths_mean, \
                                        atom = tables.Float32Atom(dflt = np.nan), \
                                        filters = table_filters);

def labelValidSkeletons(skel_file, valid_index, trajectories_data, fit_contamination = 0.1):
    #calculate valid widths if they were not used
    calculate_widths(skel_file)
    
    #calculate classifier for the outliers    
    X4fit = nodes2Array(skel_file, valid_index)        
    clf = EllipticEnvelope(contamination = fit_contamination)
    clf.fit(X4fit)
    
    #calculate outliers using the fitted classifier
    X = nodes2Array(skel_file) #use all the indexes
    y_pred = clf.decision_function(X).ravel() #less than zero would be an outlier

    #labeled rows of valid individual skeletons as GOOD_SKE
    trajectories_data['auto_label'] = ((y_pred>0).astype(np.int))*wlab['GOOD_SKE'] #+ wlab['BAD']*np.isnan(y_prev)
    saveLabelData(skel_file, trajectories_data)

def getFrameStats(feat_file):
    
    stats_funs = {'median':np.nanmedian, 'mean':np.nanmean, \
    'std':np.nanstd, 'count':lambda a : np.count_nonzero(~np.isnan(a))}
    
    table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
    warnings.simplefilter("ignore")
            
    with pd.HDFStore(feat_file) as feat_fid:
        features_motion = feat_fid['/features_motion']            
    
    if len(features_motion['worm_index'].unique()) == 1:
        return #nothing to do here
    
    groupbyframe = features_motion.groupby('frame_number')
        
    with warnings.catch_warnings(), tables.File(feat_file, 'r+') as feat_fid:
        if '/frame_stats' in feat_fid:
            feat_fid.remove_node('/', 'frame_stats', recursive = True)
        
        frame_stats_group = feat_fid.create_group('/', 'frame_stats')
        
        for stat in stats_funs:
            #ignore wargings from frames/features with only nan's
            plate_stats = groupbyframe.agg(stats_funs[stat])
            
            feat_fid.create_table(frame_stats_group, stat, \
            obj = plate_stats.to_records(), filters = table_filters)

def getFilteredFeats(skel_file, feat_file, fps = 25, min_num_skel = 100, bad_seg_thresh = 0.8, min_dist = 5):
    #get valid rows using the trajectory displacement and the skeletonization success
    valid_index, trajectories_data = getValidIndexes(skel_file, \
    min_num_skel = 100, bad_seg_thresh = 0.8, min_dist = 5)
    
    labelValidSkeletons(skel_file, valid_index, trajectories_data)
    getWormFeatures(skel_file, feat_file, bad_seg_thresh = bad_seg_thresh*0.6, fps = fps)
    getFrameStats(feat_file)

