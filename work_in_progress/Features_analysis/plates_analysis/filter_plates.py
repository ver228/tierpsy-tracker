# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:16:24 2015

@author: ajaver
"""


import os
import csv
import pandas as pd
import tables
import numpy as np
#import h5py
from collections import OrderedDict

MWTracker_dir = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking'
import sys
sys.path.append(MWTracker_dir)

from MWTracker import config_param #add the directory path for the validation movement
from MWTracker.FeaturesAnalysis.obtainFeatures_N import wormStatsClass 

from movement_validation import FeatureProcessingOptions
from movement_validation.features.worm_features import WormMorphology


class getWormMorphology(object):
    def __init__(self, nw):
        self.nw = nw
        self.morphology = WormMorphology(self)
#%%
def featureStat(stat_func, data, name, is_signed, is_motion, motion_mode = np.zeros(0), stats={}):
    
    funcEmptyNaN = lambda x: np.nan if x.size == 0 else stat_func(x)
        
    # I prefer to keep this function quite independend and pass the stats and moition_mode argument 
    #rather than save those values in the class
    if data is None:
        data = np.zeros(0)
    
    motion_types = {'all':np.nan};
    if is_motion:
        #if the the feature is motion type we can subdivide in Foward, Paused or Backward motion
        assert motion_mode.size == data.size
        motion_types['Foward'] = 1;
        motion_types['Paused'] = 0;
        motion_types['Backward'] = -1;
    
    
    valid_nan =  ~np.isnan(data)
    for key in motion_types:
        if key == 'all':
            valid = valid_nan
            sub_name = name
        else:
            valid = (motion_mode == motion_types[key]) & valid_nan
            sub_name = name + '_' + key
            
        stats[sub_name] = funcEmptyNaN(data[valid]);
        if is_signed:
            # if the feature is signed we can subdivide in positive, negative and absolute 
            stats[sub_name + '_Abs'] = funcEmptyNaN(np.abs(data[valid]))
            stats[sub_name + '_Neg'] = funcEmptyNaN(data[data<0 & valid])
            stats[sub_name + '_Pos'] = funcEmptyNaN(data[data>0 & valid])



#%%
def getVideoData(header_file):
    with open(header_file) as f:
        data = [row for row in csv.reader(f)]
    header_data = {}
    for col in zip(*data):
        header_data[col[0]] = col[1:]
    video_data = {}
    for ii, file_name in enumerate(header_data['file_name']):
        camera_focus = float(header_data['camera_focus'][ii])
        base_name = file_name.rpartition('.')[0]
        video_data[base_name] = {'pix2mum':-0.1937*camera_focus+13.4377, 
        'strain':header_data['strain'][ii]}
    return video_data
            
def getFeatSpecs():
    with open('FeatureSpecifications.csv') as f:
        data = [row for row in csv.reader(f)]
    featSpecs = {}
    for col in zip(*data):
        featSpecs[col[0]] = col[1:]
    
    feat_specs = {}
    for ii, name in enumerate(featSpecs['name']):
        feature = name.split(' (')[0].replace(' ', '_').replace('.', '').replace('-', '_')
        if '/' in feature:
            feature = feature.replace('/', '_') + '_Ratio'
        feat_specs[feature] = {'feature_type':featSpecs['feature_type'][ii], 
        'units':featSpecs['units'][ii].lower(), 'is_signed':int(featSpecs['is_signed'][ii])}
    return feat_specs

def labelValidTraj(skeletons_file, features_file):
    '''
    Select valid trajectories using Area_Length_Ratio, Width_Length_Ratio and Length
    I used this values because they depend on the morphology and seem to follow a nice gaussian distribution.
    For the moment I hard coded the thresholds, but in the future I can either give a user 
    defined number of do a statistical test to define this values. 
    '''
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        trajectories_data = ske_file_id['/trajectories_data']

    with pd.HDFStore(features_file, 'r') as feat_file_id:
        feat_avg = feat_file_id['/Features_means']
    
    #add this fields if they do not exists
    if not 'worm_index_N' in trajectories_data.columns:
        trajectories_data['worm_index_N'] = trajectories_data['worm_index_joined']
    
    trajectories_data['worm_label'] = 0
    #convert data into a rec array to save into pytables
    good_AL = (feat_avg['Area_Length_Ratio']>4.5) & (feat_avg['Area_Length_Ratio']<10)
    good_WL = (feat_avg['Width_Length_Ratio']>0.06) & (feat_avg['Width_Length_Ratio']<0.11)
    
    worm_length = feat_avg[good_AL]['Length']*pix2mum        
    good_L = (worm_length>750) & (worm_length<1400) #900 and 1400 gives a narrower estimate
    good = good_AL & good_WL & good_L
    valid_worm_index = feat_avg.loc[good, 'worm_index']
    valid_rows = trajectories_data['worm_index_joined'].isin(valid_worm_index)
    trajectories_data.loc[valid_rows, 'worm_label'] = 1
    
    #let's save the labels in trajectories_data to visualize if the selection make sense
    trajectories_recarray = trajectories_data.to_records(index=False)
    with tables.File(skeletons_file, "r+") as ske_file_id:
        #pytables filters.
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        newT = ske_file_id.create_table('/', 'trajectories_data_d', obj = trajectories_recarray, filters=table_filters)
        ske_file_id.remove_node('/', 'trajectories_data')
        newT.rename('trajectories_data')
    
    #this number can be extracted by reading trajectories data
    return valid_worm_index
#%%
if __name__ == "__main__":
    
    #%%
    main_dir = '/Volumes/behavgenom$/GeckoVideo/Ana_Strains/'
    datasetS = ['Ana_Strains_20150611', 'Ana_Strains_20150615', 'Ana_Strains_20150619']
    
    statfunc = np.nanmedian#np.nanmean    
    plates_file = '/Volumes/behavgenom$/GeckoVideo/Ana_Strains/PlateFeatures_MED.hdf5'
    
    #%%
    #get specs for each of the features extracted
    feat_specs = getFeatSpecs()
    
    all_stats = []  
    all_stats_video = []
    for dataset in datasetS:
        #%% Get info of the videos pixels to micrometers conversion and strain name
        #I should include this information in the .hdf5 video file
        header_file = os.path.join(main_dir , 'Log_files' ,dataset + '.csv')
        video_data = getVideoData(header_file)
        
        #%%
        results_dir = os.path.join(main_dir , 'Results' ,dataset) + os.sep
        filelist = os.listdir(results_dir)
        filelist = [base_name.rpartition('_skeletons.')[0] for base_name in filelist if '_skeletons.hdf5' in base_name]
        for base_name in filelist:
            print(base_name)
            pix2mum = video_data[base_name]['pix2mum']
            strain = video_data[base_name]['strain']
            fps = 25
            skeletons_file = results_dir + base_name + '_skeletons.hdf5'
            features_file = results_dir + base_name + '_features.hdf5'
            
            assert os.path.exists(skeletons_file)
            assert os.path.exists(features_file)
            
            #filter the data by morphology. In features it was already filter by the fraction of valid skeletons in a given trajectory.
            valid_worm_index = labelValidTraj(skeletons_file, features_file)
            #%%
            with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
                trajectories_data = ske_file_id['/trajectories_data']
                
                #trajectories_data = trajectories_data[trajectories_data['frame_number']<=90000]
                
            valid_rows = trajectories_data['worm_index_joined'].isin(valid_worm_index)
            
            trajectories_data = trajectories_data[valid_rows]
            
            dd = trajectories_data[['worm_index_joined', 'frame_number']].groupby('worm_index_joined').aggregate([max, min])
            traj_size = dd['frame_number']['max']-dd['frame_number']['min']

            skel_per_frame = trajectories_data[['frame_number', 'has_skeleton']].groupby('frame_number').aggregate(np.sum)
            traj_per_frame = trajectories_data[['frame_number', 'worm_index_joined']].groupby('frame_number').aggregate(len)
            
            video_stats = OrderedDict()
            video_stats['Base_Name'] = base_name
            video_stats['Total_Frames'] = trajectories_data['frame_number'].max();
            video_stats['Total_Skeletons']  = trajectories_data['has_skeleton'].sum()
            video_stats['Total_Trajectories'] = len(traj_size)            
            
            traj_quat = traj_size.quantile([0.25, 0.5, 0.75])
            video_stats['Traj_Size_Q25'] = traj_quat[0.25]
            video_stats['Traj_Size_Q50'] = traj_quat[0.5]
            video_stats['Traj_Size_Q75'] = traj_quat[0.75]             
            
            video_stats['Skel_Per_Frame_Med'] = np.median(skel_per_frame)
            video_stats['Traj_Per_Frame_Med'] = np.median(traj_per_frame)
            
            all_stats_video.append(video_stats)
            #%%            
            with pd.HDFStore(features_file, 'r') as feat_file_id:
                feat_motion = feat_file_id['/Features_motion']
                feat_motion = feat_motion[feat_motion['worm_index'].isin(valid_worm_index)]
            
            
            #%%
            #get the averages by subdividing data in different ways
            plate_stats = OrderedDict()
            plate_stats['Base_Name'] = base_name
            plate_stats['Strain'] = strain
            
            motion_mode = feat_motion['Motion_Modes'].values
            for name in feat_specs.keys():
                spec_data = feat_specs[name]
                is_movement = spec_data['feature_type'] == 'movement'
                is_signed = spec_data['is_signed']
                
                if is_movement:
                    data = feat_motion[name]
                else:
                    with tables.File(features_file, 'r') as feat_file_id:
                        data = np.zeros(0)
                        for worm_index in valid_worm_index:
                            h5path = '/Features_events/worm_%i/%s' % (worm_index, name)
                            if h5path in feat_file_id:
                                dum = feat_file_id.get_node(h5path)[:]
                                data = np.hstack((data, dum))
    
                #convert the features from pixels to microns
                units = spec_data['units']
                if 'microns' in units:
                    if units == 'microns^2':
                        data = data*(pix2mum**2)
                    elif units == '/microns':
                        data = data/pix2mum
                    else:
                        data = data*pix2mum
                        
                featureStat(statfunc, data, name, is_signed, is_movement, 
                            motion_mode=motion_mode, stats=plate_stats)
            #append the stats to the major list
            all_stats.append(plate_stats)
        #%%
        #create and save a table containing the averaged worm feature for each worm
        tot_rows = len(all_stats)
        
        #get data type for each field
        plate_dtype = []
        for x in all_stats[0].keys():
            if x=='Strain' or x == 'Base_Name':
                dd = np.dtype((np.character,50))
            else:
                dd = np.float
            plate_dtype.append((x,dd))
        #pass it to a rec array
        mean_features_df = np.recarray(tot_rows, dtype = plate_dtype);
        for kk, row_dict in enumerate(all_stats):
            for key in row_dict:
                mean_features_df[key][kk] = row_dict[key]
        
        #video stats to rec
        video_dtype = []
        for x in all_stats_video[0].keys():
            if x == 'Base_Name':
                dd = np.dtype((np.character,50))
            else:
                dd = np.float
            video_dtype.append((x,dd))
        
        video_features_df = np.recarray(tot_rows, dtype = video_dtype);
        for kk, row_dict in enumerate(all_stats_video):
            for key in row_dict:
                video_features_df[key][kk] = row_dict[key]
        
        filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
        with tables.File(plates_file, 'w') as plates_fid:
            feat_mean = plates_fid.create_table('/', 'avg_feat_per_plate', obj = mean_features_df, filters=filters_tables)
            feat_mean = plates_fid.create_table('/', 'video_features', obj = video_features_df, filters=filters_tables)
        
        
        #%%
        #fields2filter = ['worm_index_joined', 'has_skeleton']#, 'coord_x', 'coord_y']
        #tracks_data = trajectories_data[fields2filter].groupby('worm_index_joined').aggregate(['max', 'min', 'mean', 'count'])
        
        #delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
        #delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
                
        #good = (tracks_data['has_skeleton']['mean']>=bad_seg_thresh) #& (delX>min_displacement) & (delY>min_displacement)
        #valid_worm_index = (tracks_data[good]).index
        
        
        
        #'Area', 'Length', 'Head_Width', 'Midbody_Width', 'Tail_Width', 'Area_Length_Ratio'
        
#        tot_row = len(valid_worm_index)
#        morphology_df = np.recarray(tot_row, dtype = [('worm_index', np.int32), \
#        ('area', np.float32), ('length', np.float32), ('width_per_length', np.float32), 
#        ('area_per_length', np.float32), ('width_head', np.float32), ('width_midbody', np.float32), 
#        ('width_tail', np.float32)])        
#        
#        for ii, worm_index in enumerate(valid_worm_index):
#            print(ii+1, tot_row)
#            worm = WormFromTable()
#            worm.fromFile(skeletons_file, worm_index, fps = fps, pix2mum = pix2mum, isOpenWorm=True)
#            features = getWormMorphology(worm)
#            
#            morphology_df['worm_index'][ii] = worm_index
#            morphology_df['area'][ii] = np.nanmedian(features.morphology.area)            
#            morphology_df['length'][ii] = np.nanmedian(features.morphology.length) 
#            morphology_df['width_per_length'][ii] = np.nanmedian(features.morphology.width_per_length) 
#            morphology_df['area_per_length'][ii] = np.nanmedian(features.morphology.area_per_length) 
#            morphology_df['width_head'][ii] = np.nanmedian(features.morphology.width.head) 
#            morphology_df['width_midbody'][ii] = np.nanmedian(features.morphology.width.midbody) 
#            morphology_df['width_tail'][ii] = np.nanmedian(features.morphology.width.tail) 
            #assert not np.all(np.isnan(worm.skeleton))
            #print(w_index)
            

