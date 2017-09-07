#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:31:24 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables
from scipy.interpolate import interp1d

from tierpsy_features import SmoothedWorm, get_timeseries_features
from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormFromTable
from tierpsy.analysis.stage_aligment.alignStageMotion import _h_get_stage_inv

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name, TABLE_FILTERS
from tierpsy.helper.params import copy_unit_conversions, read_fps, read_microns_per_pixel

def _fill_dropped_frames(worm_data, dflt_val):
    ini_t = worm_data['timestamp_raw'].min()
    fin_t = worm_data['timestamp_raw'].max()
    n_size = fin_t-ini_t + 1
    
    index_raw = worm_data['timestamp_raw'] - ini_t
    
    dflt_dtypes = list(worm_data.dtypes.items())
    dat = np.array([dflt_val]*n_size, dflt_dtypes)
    
    dat['timestamp_raw'] = np.arange(ini_t, fin_t+1)
    dat['worm_index_joined'] = worm_data['worm_index_joined'].values[0]
    dat['roi_size'] = worm_data['roi_size'].values[0]
    
    for field in ['frame_number', 'was_skeletonized', 'old_trajectory_data_index']:
        dat[field][index_raw] = worm_data[field]
    
    for field in ['timestamp_time', 'coord_x', 'coord_y', 'threshold', 'area']:
        f = interp1d(worm_data['timestamp_raw'], worm_data[field].values)
        dat[field] = f(dat['timestamp_raw'])
    
    worm_df = pd.DataFrame(dat, 
                           columns = worm_data.columns
                           )
    return worm_df

def _r_fill_trajectories_data(skeletons_file):
    valid_fields = ['timestamp_raw', 'timestamp_time', 'worm_index_joined', 
                    'coord_x', 'coord_y', 
                    'threshold', 'roi_size',  'area', 
                    'frame_number', 'is_good_skel'
                    ]
    #%%
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        if 'worm_index_manual' in trajectories_data:
            valid_fields += ['worm_index_manual', 'worm_label']
        
    #%%
    trajectories_data = trajectories_data[valid_fields]
    trajectories_data.rename(columns = {'worm_index_joined' : 'worm_index_joined', 
                                        'is_good_skel' : 'was_skeletonized'},
                                        inplace=True)
    
    trajectories_data['skeleton_id'] = np.int32(-1)
    trajectories_data['old_trajectory_data_index'] = trajectories_data.index.values.astype(np.int32)
    
    #change table to 32bits if necessary
    for col in trajectories_data:
        if trajectories_data[col].dtype == np.int64:
            trajectories_data[col] = trajectories_data[col].astype(np.int32)
        elif trajectories_data[col].dtype == np.float64:
            trajectories_data[col] = trajectories_data[col].astype(np.float32)
        elif trajectories_data[col].dtype == np.bool:
            trajectories_data[col] = trajectories_data[col].astype(np.uint8)
    
    assert set(x for _,x in trajectories_data.dtypes.items()) == \
    {np.dtype('uint8'), np.dtype('int32'), np.dtype('float32')}
    
    if trajectories_data['timestamp_raw'].isnull().all():
        fps = read_fps(skeletons_file)
        trajectories_data['timestamp_raw'] = trajectories_data['frame_number']
        trajectories_data['timestamp_time'] = trajectories_data['frame_number']*fps
        
    else:
        #if it is not nan convert this data into int
        trajectories_data['timestamp_raw'] = trajectories_data['timestamp_raw'].astype(np.int32)
        
        dflt_d = {np.dtype('int32') : -1, 
                  np.dtype(np.float32) : np.nan, 
                  np.dtype('uint8') : 0
                  }
        dflt_val = tuple([dflt_d[x] for _,x in trajectories_data.dtypes.items()])
        
        all_worm_data = []
        for worm_index, worm_data in trajectories_data.groupby('worm_index_joined'):
            worm_data = worm_data.dropna(subset = ['timestamp_raw'])
            worm_data = worm_data.drop_duplicates(subset = ['timestamp_raw'], 
                                      keep = 'first')
            
            if not (worm_data['frame_number'] == worm_data['timestamp_raw']).all():
                worm_data = _fill_dropped_frames(worm_data, dflt_val)
            
            all_worm_data.append(worm_data)
    
        trajectories_data = pd.concat(all_worm_data, ignore_index=True)
    
    return trajectories_data

def _r_fill_blob_features(skeletons_file, trajectories_data_f, is_WT2):
    microns_per_pixel = read_microns_per_pixel(skeletons_file)
    
    with pd.HDFStore(skeletons_file, 'r') as fid:
        if not 'blob_features' in fid:
            return
        
        blob_features = fid['/blob_features']
    
    blob_features['area'] *= microns_per_pixel**2
    
    for feat in ['coord_x', 'coord_y', 'perimeter', 'box_length', 'box_width']:
        blob_features[feat] *= microns_per_pixel
    
    blob_features = blob_features.merge( 
            trajectories_data_f[['old_trajectory_data_index']],
            left_index=True, 
            right_on='old_trajectory_data_index',
            how = 'right'
            )
    
    del blob_features['old_trajectory_data_index']
    if is_WT2:
        stage_vec_inv, _ = _h_get_stage_inv(skeletons_file, 
                                            trajectories_data_f['timestamp_raw'].values)
        is_stage_move = np.isnan(stage_vec_inv[:, 0])
        blob_features[is_stage_move] = np.nan
        
        blob_features['coord_x'] += stage_vec_inv[:, 0]
        blob_features['coord_y'] += stage_vec_inv[:, 1]
        
    blob_features = blob_features.interpolate()
    return blob_features

def _h_smooth_skeletons_table(skeletons_file, 
                            features_file,
                            is_WT2 = False,
                            skel_smooth_window = 5,
                            coords_smooth_window_s = 0.25,
                            gap_to_interp_s = 0.25
                            ):
    #%%
    fps = read_fps(skeletons_file)
    coords_smooth_window = int(np.round(fps*coords_smooth_window_s))
    gap_to_interp = int(np.round(fps*gap_to_interp_s))
    
    if coords_smooth_window <= 3: #do not interpolate
        coords_smooth_window = None
        
    trajectories_data = _r_fill_trajectories_data(skeletons_file)
    #%%
    trajectories_data_g = trajectories_data.groupby('worm_index_joined')
    progress_timer = TimeCounter('')
    base_name = get_base_name(skeletons_file)
    tot_worms = len(trajectories_data_g)
    def _display_progress(n):
            # display progress
        dd = " Smoothing skeletons. Worm %i of %i done." % (n+1, tot_worms)
        print_flush(
            base_name +
            dd +
            ' Total time:' +
            progress_timer.get_time_str())
    
    
    _display_progress(0)
    #%%
    #initialize arrays
    with tables.File(skeletons_file, 'r') as fid:
        n_segments = fid.get_node('/skeleton').shape[1]
    
    with tables.File(features_file, 'w') as fid_features:
        worm_coords_array = {}
        w_node = fid_features.create_group('/', 'coordinates')
        for array_name in ['skeletons', 'dorsal_contours', 'ventral_contours', 'widths']:
            if array_name != 'widths':  
                a_shape = (0, n_segments, 2)
            else:
                a_shape = (0, n_segments)
            
            worm_coords_array[array_name] = fid_features.create_earray(
                w_node,
                array_name,
                shape= a_shape,
                atom=tables.Float32Atom(shape=()),
                filters=TABLE_FILTERS)
            
    
        tot_skeletons = 0
        for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
            if worm_data['was_skeletonized'].sum() < 2:
                continue
            
            worm = WormFromTable(skeletons_file,
                                worm_index,
                                worm_index_type = 'worm_index_joined'
                                )
        
            if is_WT2:
                worm.correct_schafer_worm()
                
            wormN = SmoothedWorm(
                         worm.skeleton, 
                         worm.widths, 
                         worm.ventral_contour, 
                         worm.dorsal_contour,
                         skel_smooth_window = skel_smooth_window,
                         coords_smooth_window = coords_smooth_window,
                         gap_to_interp = gap_to_interp
                        )
            
            dat_index = pd.Series(False, index = worm_data['timestamp_raw'].values)
            dat_index[worm.timestamp] = True
            
            
            #%%
            skeleton_id = np.arange(wormN.skeleton.shape[0]) + tot_skeletons
            tot_skeletons = skeleton_id[-1] + 1
            row_ind = worm_data.index[dat_index.values]
            trajectories_data.loc[row_ind, 'skeleton_id'] = skeleton_id
            
            #add data
            worm_coords_array['skeletons'].append(getattr(wormN, 'skeleton'))
            worm_coords_array['dorsal_contours'].append(getattr(wormN, 'dorsal_contour'))
            worm_coords_array['ventral_contours'].append(getattr(wormN, 'ventral_contour'))
            worm_coords_array['widths'].append(getattr(wormN, 'widths'))
            
            #display progress
            _display_progress(ind_n + 1)
            
        #save trajectories data
        newT = fid_features.create_table(
                '/',
                'trajectories_data',
                obj = trajectories_data.to_records(index=False),
                filters = TABLE_FILTERS)
        copy_unit_conversions(newT, skeletons_file)
        newT._v_attrs['is_WT2'] = is_WT2
        
        
        #save blob features interpolating in dropped frames and stage movement (WT2)
        blob_features = _r_fill_blob_features(skeletons_file, trajectories_data, is_WT2)
        if blob_features is not None:
            fid_features.create_table(
                '/',
                'blob_features',
                obj = blob_features.to_records(index=False),
                filters = TABLE_FILTERS)

def _h_get_timeseries_feats_table(features_file, 
                                  delta_time,
                                  curvature_window):
    timeseries_features = []
    fps = read_fps(features_file)
    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    #only use data that was skeletonized
    trajectories_data = trajectories_data[trajectories_data['skeleton_id']>=0]
    
    
    trajectories_data_g = trajectories_data.groupby('worm_index_joined')
    progress_timer = TimeCounter('')
    base_name = get_base_name(features_file)
    tot_worms = len(trajectories_data_g)
    def _display_progress(n):
            # display progress
        dd = " Smoothing skeletons. Worm %i of %i done." % (n+1, tot_worms)
        print_flush(
            base_name +
            dd +
            ' Total time:' +
            progress_timer.get_time_str())
    
    _display_progress(0)
    
    
    for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
        with tables.File(features_file, 'r') as fid:
            skel_id = worm_data['skeleton_id'].values
            args = []
            for p in ('skeletons', 'widths', 'dorsal_contours', 'ventral_contours'):
                 dd = fid.get_node('/coordinates/' + p)
                 if len(dd.shape) == 3:
                     args.append(dd[skel_id, :, :])
                 else:
                     args.append(dd[skel_id, :])
                
        feats = get_timeseries_features(*args, 
                                        fps = fps,
                                        delta_time = delta_time, #delta time in seconds to calculate the velocity
                                        curvature_window = curvature_window
                                        )
        feats = feats.astype(np.float32)
        feats['worm_index'] = np.int32(worm_index)
        feats['timestamp'] = worm_data['timestamp_raw'].values
        
        #move the last fields to the first columns
        cols = feats.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        timeseries_features.append(feats[cols])
        
        _display_progress(ind_n+1)
        
    timeseries_features = pd.concat(timeseries_features, ignore_index=True)
    
    with tables.File(features_file, 'r+') as fid:
        if '/timeseries_features' in fid:
            fid.remove_node('/timeseries_features')

        fid.create_table(
                '/',
                'timeseries_features',
                obj = timeseries_features.to_records(index=False),
                filters = TABLE_FILTERS)

def get_tierpsy_features(
        skeletons_file, 
        features_file,
        is_WT2 = False,
        skel_smooth_window = 5,
        coords_smooth_window_s = 0.25,
        gap_to_interp_s = 0.25,
        delta_time = 1/3,
        curvature_window = 7
        ):
    
    _h_smooth_skeletons_table(skeletons_file, 
                              features_file, 
                              is_WT2 = is_WT2,
                              skel_smooth_window = skel_smooth_window,
                              coords_smooth_window_s = coords_smooth_window_s,
                              gap_to_interp_s = gap_to_interp_s)
   
    _h_get_timeseries_feats_table(features_file, 
                                delta_time,
                                curvature_window
                                )
    
if __name__ == '__main__':
    
    
    #base_file = '/Volumes/behavgenom_archive$/single_worm/finished/mutants/gpa-10(pk362)V@NL1147/food_OP50/XX/30m_wait/clockwise/gpa-10 (pk362)V on food L_2009_07_16__12_55__4'
    #base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/WT2/Results/WT2'
    #is_WT2 = True
    
    #base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/AVI_VIDEOS/Results/AVI_VIDEOS_4'
    base_file = '/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/GECKO_VIDEOS/Results/GECKO_VIDEOS'
    is_WT2 = False
    
    
    #/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/GECKO_VIDEOS/MaskedVideos/GECKO_VIDEOS.hdf5
    #'/Users/ajaver/Documents/GitHub/tierpsy-tracker/tests/data/AVI_VIDEOS/MaskedVideos/AVI_VIDEOS_4.hdf5'
    skeletons_file = base_file + '_skeletons.hdf5'
    features_file = base_file + '_featuresN.hdf5'
    
    get_tierpsy_features(
        skeletons_file, 
        features_file,
        is_WT2,
        skel_smooth_window = 5,
        coords_smooth_window_s = 0.25,
        gap_to_interp_s = 0.25,
        delta_time = 1/3
        )
