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

import warnings
warnings.filterwarnings('ignore', '.*empty slice*',)
tables.parameters.MAX_COLUMNS = 1024 #(http://www.pytables.org/usersguide/parameter_files.html)

from collections import OrderedDict

from ..helperFunctions.timeCounterStr import timeCounterStr

from MWTracker import config_param
from open_worm_analysis_toolbox import WormFeatures, FeatureProcessingOptions

from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass, WormFromTable

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:16:41 2015

@author: ajaver
"""

def getWormFeaturesLab(skeletons_file, features_file, worm_indexes, fps = 25, time_range = []):

    #overight processing options
    processing_options = FeatureProcessingOptions()
    #increase the time window (s) for the velocity calculation 
    processing_options.locomotion.velocity_tip_diff = 0.5
    processing_options.locomotion.velocity_body_diff = 1

    #useful to display progress 
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    #initialize by getting the specs data subdivision
    wStats = wormStatsClass()

    #list to save trajectories mean features
    all_stats = []
    
    progress_timer = timeCounterStr('');

    #filter used for each fo the tables
    filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
    
    #create the motion table header
    motion_header = {'frame_number':tables.Int32Col(pos=0),\
    'skeleton_id':tables.Int32Col(pos=1),\
    'motion_modes':tables.Float32Col(pos=2)}

    for ii, spec in enumerate(wStats.specs_motion):
        feature = wStats.spec2tableName[spec.name]
        motion_header[feature] = tables.Float32Col(pos=ii+2)

    #get the is_signed flag for motion specs and store it as an attribute
    #is_signed flag is used by featureStat in order to subdivide the data if required
    is_signed_motion = np.zeros(len(motion_header), np.uint8);
    for ii, spec in enumerate(wStats.specs_motion):
        feature = wStats.spec2tableName[spec.name]
        is_signed_motion[motion_header[feature]._v_pos] = spec.is_signed


    with tables.File(features_file, 'w') as features_fid:

        #features group
        group_features = features_fid.create_group('/', 'features')
        tot_rows = 0
            
        #Calculate features for each worm trajectory      
        tot_worms = len(worm_indexes)        
        for ind, worm_index  in enumerate(worm_indexes):
            #initialize worm object, and extract data from skeletons file
            worm = WormFromTable()
            worm.fromFile(skeletons_file, worm_index, fps = fps, isOpenWorm = False, time_range=time_range)
            assert not np.all(np.isnan(worm.skeleton))

            #save data as a subgroup for each worm
            worm_node = features_fid.create_group(group_features, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_index'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.frame_number[0], worm.frame_number[-1])

            #save skeleton
            features_fid.create_carray(worm_node, 'skeletons', \
                                    obj = worm.skeleton, filters=filters_tables)
            
            #change axis to an openworm format
            worm.changeAxis()

            # Generate the OpenWorm movement validation repo version of the features
            worm_features = WormFeatures(worm, processing_options=processing_options)
            

            #get the average for each worm feature
            worm_stats = wStats.getWormStats(worm_features, np.mean)
            worm_stats['n_frames'] = worm.n_frames
            worm_stats['worm_index'] = worm_index
            worm_stats.move_to_end('n_frames', last=False)
            worm_stats.move_to_end('worm_index', last=False)
            all_stats.append(worm_stats)
            
            
            #save event features
            events_node = features_fid.create_group(worm_node, 'events')
            for spec in wStats.specs_events:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                
                if tmp_data is not None and tmp_data.size > 0:
                    table_tmp = features_fid.create_carray(events_node, feature, \
                                    obj = tmp_data, filters=filters_tables)
                    table_tmp._v_attrs['is_signed'] = int(spec.is_signed)
            
            dd = " Extracting features (labeled). Worm %i of %i done." % (ind+1, tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
            sys.stdout.flush()
            
            #initialize motion table. All the features here are a numpy array having the same length as the worm trajectory
            table_motion = features_fid.create_table(worm_node, 'locomotion', motion_header, filters=filters_tables)
            table_motion._v_attrs['is_signed'] = is_signed_motion
            
            #save the motion data as a general table
            motion_data = [[]]*len(motion_header)
            motion_data[motion_header['frame_number']._v_pos] = worm.frame_number
            #motion_data[motion_header['worm_index']._v_pos] = np.full(worm.n_frames, worm.worm_index)
            motion_data[motion_header['skeleton_id']._v_pos] = worm.skeleton_id
            motion_data[motion_header['motion_modes']._v_pos] = worm_features.locomotion.motion_mode
            for spec in wStats.specs_motion:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                motion_data[motion_header[feature]._v_pos] = tmp_data
            
            motion_data = list(zip(*motion_data))
            table_motion.append(motion_data)
            
            table_motion.flush()
            del motion_data
            tot_rows += 1
            
        assert tot_worms == tot_rows
        
        if tot_rows > 0:
            dtype = [(x, np.float32) for x in (all_stats[0])]
            mean_features_df = np.recarray(tot_rows, dtype = dtype);
            for kk, row_dict in enumerate(all_stats):
                for key in row_dict:
                    mean_features_df[key][kk] = row_dict[key]
            feat_mean = features_fid.create_table('/', 'features_means', obj = mean_features_df, filters=filters_tables)
        else:
            feat_mean = features_fid.create_table('/', 'features_means', {'worm_index' : tables.Int32Col(pos=0)}, filters=filters_tables)
        
        feat_mean._v_attrs['has_finished'] = 1
        
        print(base_name + ' Feature extraction (labeled) finished:' + progress_timer.getTimeStr())
        sys.stdout.flush()

def featFromLabSkel(skel_file, ind_feat_file, fps=25):

    with pd.HDFStore(skel_file, 'r') as ske_file_id:
        trajectories_data = ske_file_id['/trajectories_data']

    if not 'worm_label' in trajectories_data.columns:
        return
    
    trajectories_data = trajectories_data[trajectories_data['worm_label']==1]
    worm_indexes = trajectories_data['worm_index_N'].unique()
    getWormFeaturesLab(skel_file, ind_feat_file, worm_indexes, fps)
