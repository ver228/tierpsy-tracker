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

from open_worm_analysis_toolbox import NormalizedWorm
from open_worm_analysis_toolbox import WormFeatures, VideoInfo
from open_worm_analysis_toolbox.statistics import specifications

from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass, WormFromTable

def getWormFeatures(skeletons_file, features_file, bad_seg_thresh = 0.5, fps = 25, min_num_skel = 25):

    #useful to display progress 
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    #read skeletons index data
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        indexes_data = ske_file_id['/trajectories_data']
    
    if 'has_skeleton' in indexes_data.columns:
        indexes_data = indexes_data[['worm_index_joined', 'skeleton_id', 'has_skeleton']]
    else:
        indexes_data = indexes_data[['worm_index_joined', 'skeleton_id']]
        with tables.File(skeletons_file, 'r') as ske_file_id:
            #this is slow but faster than having to recalculate all the skeletons
            indexes_data['has_skeleton'] = ~np.isnan(ske_file_id.get_node('/skeleton_length'))
            
    #%%
    
    #get the fraction of worms that were skeletonized per trajectory
    dum = indexes_data.groupby('worm_index_joined').agg({'has_skeleton':['mean', 'sum']})
    skeleton_fracc = dum['has_skeleton']['mean']
    skeleton_tot = dum['has_skeleton']['sum']
    valid_worm_index = skeleton_fracc[(skeleton_fracc >= bad_seg_thresh) & (skeleton_tot>=min_num_skel)].index
    
    #remove the bad worms, we do not care about them
    indexes_data = indexes_data[indexes_data['worm_index_joined'].isin(valid_worm_index)]
    
    #get the first and last frame of each worm_index
    rows_indexes = indexes_data.groupby('worm_index_joined').agg({'skeleton_id':[min, max]})
    rows_indexes = rows_indexes['skeleton_id']
    
    #remove extra variable to free memory
    del indexes_data

    #get total number of valid worms and break if it is zero
    tot_worms = len(rows_indexes)

    #initialize by getting the specs data subdivision
    wStats = wormStatsClass()
    
    #list to save trajectories mean features
    all_stats = []
    
    progress_timer = timeCounterStr('');
    
    #filter used for each fo the tables
    filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
    with tables.File(features_file, 'w') as features_fid:

        group_events = features_fid.create_group('/', 'features_events')
        
        #initialize motion table. All the features here are a numpy array having the same length as the worm trajectory
        motion_header = {'worm_index':tables.Int32Col(pos=0),\
        'frame_number':tables.Int32Col(pos=1),\
        'motion_modes':tables.Float32Col(pos=2)}
        
        for ii, spec in enumerate(wStats.specs_motion):
            feature = wStats.spec2tableName[spec.name]
            motion_header[feature] = tables.Float32Col(pos=ii+2)
        table_motion = features_fid.create_table('/', 'features_motion', motion_header, filters=filters_tables)
        
        #get the is_signed flag for motion specs and store it as an attribute
        #is_signed flag is used by featureStat in order to subdivide the data if required
        is_signed_motion = np.zeros(len(motion_header), np.uint8);
        for ii, spec in enumerate(wStats.specs_motion):
            feature = wStats.spec2tableName[spec.name]
            is_signed_motion[motion_header[feature]._v_pos] = spec.is_signed

        table_motion._v_attrs['is_signed'] = is_signed_motion
        
        #start to calculate features for each worm trajectory      
              
        for ind, dat  in enumerate(rows_indexes.iterrows()):
            worm_index, row_range = dat
            
            #initialize worm object, and extract data from skeletons file
            worm = WormFromTable()
            worm.fromFile(skeletons_file, worm_index, fps = 25, isOpenWorm = True)
            
            if np.all(np.isnan(worm.length)):
                tot_worms = tot_worms - 1
                continue
            
            # Generate the OpenWorm movement validation repo version of the features
            worm_features = WormFeatures(worm)

            #get the average for each worm feature
            worm_stats = wStats.getWormStats(worm_features, np.mean)
            worm_stats['n_frames'] = worm.n_frames
            worm_stats['worm_index'] = worm_index
            worm_stats['n_valid_skel'] = worm.n_valid_skel
            
            for feat in ['n_valid_skel', 'n_frames', 'worm_index']:
                worm_stats.move_to_end(feat, last=False)
            
            all_stats.append(worm_stats)
            #save the motion data as a general table
            motion_data = [[]]*len(motion_header)
            motion_data[motion_header['frame_number']._v_pos] = worm.frame_number
            motion_data[motion_header['worm_index']._v_pos] = np.full(worm.n_frames, worm.worm_index, dtype=np.int64)
            motion_data[motion_header['motion_modes']._v_pos] = worm_features.locomotion.motion_mode
            for spec in wStats.specs_motion:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                motion_data[motion_header[feature]._v_pos] = tmp_data
            
            motion_data = list(zip(*motion_data))
            table_motion.append(motion_data)
            table_motion.flush()
            del motion_data
            
            #save events data as a subgroup for the worm
            worm_node = features_fid.create_group(group_events, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_index'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.frame_number[0], worm.frame_number[-1])
            worm_node._v_attrs['skeletons_rows_range'] = tuple(row_range.values)
            worm_node._v_attrs['n_valid_skel'] = worm.n_valid_skel
            
            for spec in wStats.specs_events:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                
                if tmp_data is not None and tmp_data.size > 0:
                    table_tmp = features_fid.create_carray(worm_node, feature, \
                                    obj = tmp_data, filters=filters_tables)
                    table_tmp._v_attrs['is_signed'] = int(spec.is_signed)
            
            dd = " Extracting features. Worm %i of %i done." % (len(all_stats), tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
            sys.stdout.flush()
            sys.stderr.flush()
        

        #create and save a table containing the averaged worm feature for each worm
        tot_rows = len(all_stats)
        assert tot_worms == tot_rows
        
        if tot_rows > 0:
            dtype = [(x, np.float32) for x in (all_stats[0])]
            mean_features_df = np.recarray(tot_rows, dtype = dtype);
            for kk, row_dict in enumerate(all_stats):
                for key in row_dict:
                    mean_features_df[key][kk] = row_dict[key]
            feat_mean = features_fid.create_table('/', 'features_means', obj = mean_features_df, filters=filters_tables)
        else:
            #if no valid worms were selected create an empty table with only one column
            feat_mean = features_fid.create_table('/', 'features_means', {'worm_index' : tables.Int32Col(pos=0)}, filters=filters_tables)
            
        feat_mean._v_attrs['has_finished'] = 1
        
        print(base_name + ' Feature extraction finished: ' + progress_timer.getTimeStr())
        sys.stdout.flush()
        
if __name__ == "__main__":
    
#    base_name = 'Capture_Ch3_12052015_194303'
#    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
#    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    

    base_name = 'Capture_Ch5_11052015_195105'
    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/'
    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'
    
    masked_image_file = mask_dir + base_name + '.hdf5'
    trajectories_file = results_dir + base_name + '_trajectories.hdf5'
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    features_file = results_dir + base_name + '_features.hdf5'
    
    assert os.path.exists(masked_image_file)
    assert os.path.exists(trajectories_file)
    assert os.path.exists(skeletons_file)
        
    getWormFeatures(skeletons_file, features_file)
