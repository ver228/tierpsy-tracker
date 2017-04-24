#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 19:15:50 2017

@author: ajaver
"""
import copy
import numpy as np
import pandas as pd
import tables
import warnings
import os
from collections import OrderedDict
from scipy.signal import savgol_filter

from tierpsy.analysis.stage_aligment.alignStageMotion import isGoodStageAligment
from tierpsy.analysis.contour_orient.correctVentralDorsal import read_ventral_side
from tierpsy.helper.params import read_fps, read_microns_per_pixel
from tierpsy import AUX_FILES_DIR
import open_worm_analysis_toolbox as mv

# (http://www.pytables.org/usersguide/parameter_files.html)
tables.parameters.MAX_COLUMNS = 1024

def _h_smooth_curve(curve, window=5, pol_degree=3):
    '''smooth curves using the savgol_filter'''

    if curve.shape[0] < window:
        # nothing to do here return an empty array
        return np.full_like(curve, np.nan)

    # consider the case of one (widths) or two dimensions (skeletons, contours)
    if curve.ndim == 1:
        smoothed_curve = savgol_filter(curve, window, pol_degree)
    else:
        smoothed_curve = np.zeros_like(curve)
        for nn in range(curve.ndim):
            smoothed_curve[:, nn] = savgol_filter(
                curve[:, nn], window, pol_degree)

    return smoothed_curve

def _h_smooth_curve_all(curves, window=5, pol_degree=3):
    for ii in range(curves.shape[0]):
        if not np.any(np.isnan(curves[ii])):
            curves[ii] = _h_smooth_curve(
                curves[ii], window=window, pol_degree=pol_degree)
    return curves

class WormFromTable():
    def __init__(self, 
                file_name, 
                worm_index, 
                use_skel_filter=True,
                worm_index_str='worm_index_joined',
                smooth_window=-1, 
                POL_DEGREE_DFLT=3):
        # Populates an empty normalized worm.
        #if it does not exists return 1 as a default, like that we can still calculate the features in pixels and frames, instead of micrometers and seconds.
        self.microns_per_pixel = read_microns_per_pixel(file_name, dflt=1)
        self.fps = read_fps(file_name, dflt=1)
        
        # savitzky-golay filter polynomial order default
        self.POL_DEGREE_DFLT = POL_DEGREE_DFLT
        # save the input parameters
        self.file_name = file_name
        self.worm_index = worm_index
        self.use_skel_filter = use_skel_filter
        self.worm_index_str = worm_index_str
        # set to less than POL_DEGREE_DFLT to eliminate smoothing
        self.smooth_window = smooth_window

        # smooth window must be an odd number larger than the polynomial degree
        # (savitzky-golay filter requirement)
        if self.smooth_window >= self.POL_DEGREE_DFLT and self.smooth_window % 2 == 0:
            self.smooth_window += 1

        self.ventral_side = 'unknown'
        self._h_read_data()

        # smooth data if required
        if self.smooth_window > self.POL_DEGREE_DFLT:
            # print('Smoothing...')
            self.skeleton = _h_smooth_curve_all(
                self.skeleton, window=self.smooth_window)
            self.widths = _h_smooth_curve_all(
                self.widths, window=self.smooth_window)

        
        # assert the dimenssions of the read data are correct
        self._h_assert_data_dim()

    def _h_get_table_indexes(self):
        '''
        Get the relevant info from the trajectory_data table for a single worm. skeleton_id, timestamp.
        '''
        # intialize just to make clear the relevant variables for this function

        with pd.HDFStore(self.file_name, 'r') as ske_file_id:
            trajectories_data_f = ske_file_id['/trajectories_data']

            # get the rows of valid skeletons
            assert self.worm_index_str in trajectories_data_f
            good = trajectories_data_f[self.worm_index_str] == self.worm_index
            trajectories_data = trajectories_data_f.loc[good]

            skel_table_id = trajectories_data['skeleton_id'].values
            try:
                # try to read the time stamps, if there are repeated or not a
                # number use the frame nuber instead
                timestamp_raw = trajectories_data['timestamp_raw'].values
                if np.any(np.isnan(timestamp_raw)) or np.any(np.diff(timestamp_raw) == 0):
                    raise ValueError
                else:
                    timestamp_inds = timestamp_raw.astype(np.int)
            except (ValueError, KeyError):
                # if the time stamp fails use the frame_number value instead
                # (the index of the mask) and return nan as the fps
                timestamp_inds = trajectories_data['frame_number'].values
                
            # we need to use (.values) to be able to use the & operator
            good_skeletons = (trajectories_data['has_skeleton'] == 1).values
            if self.use_skel_filter and 'is_good_skel' in trajectories_data:
                # only keep skeletons that where labeled as good skeletons in
                # the filtering step
                good_skeletons &= (
                    trajectories_data['is_good_skel'] == 1).values

            skel_table_id = skel_table_id[good_skeletons]
            timestamp_inds = timestamp_inds[good_skeletons]
            return skel_table_id, timestamp_inds


    def _h_read_data(self):
        skel_table_id, timestamp_inds = self._h_get_table_indexes()
        
        if not np.array_equal(np.sort(timestamp_inds), timestamp_inds): #the time stamp must be sorted
            warnings.warn('{}: The timestamp is not sorted in worm_index {}'.format(self.file_name, self.worm_index))
        
        # use real frames to define the size of the object arrays
        first_frame = np.min(timestamp_inds)
        last_frame = np.max(timestamp_inds)
        n_frames = last_frame - first_frame + 1

        
        # get the apropiate index in the object array
        ind_ff = timestamp_inds - first_frame

        # get the number of segments from the normalized skeleton
        with tables.File(self.file_name, 'r') as ske_file_id:
            self.n_segments = ske_file_id.get_node('/skeleton').shape[1]
 
        # add the data from the skeleton_id's and timestamps used
        self.timestamp = np.arange(first_frame, last_frame+1)
        
        self.skeleton_id = np.full(n_frames, -1, np.int32)
        self.skeleton_id[ind_ff] = skel_table_id
        
        # initialize the rest of the arrays
        self.skeleton = np.full((n_frames, self.n_segments, 2), np.nan)
        self.ventral_contour = np.full((n_frames, self.n_segments, 2), np.nan)
        self.dorsal_contour = np.full((n_frames, self.n_segments, 2), np.nan)
        self.widths = np.full((n_frames, self.n_segments), np.nan)

        # read data from the skeletons table
        with tables.File(self.file_name, 'r') as ske_file_id:
            self.skeleton[ind_ff] = \
            ske_file_id.get_node('/skeleton')[skel_table_id, :, :] * self.microns_per_pixel
            self.ventral_contour[ind_ff] = \
            ske_file_id.get_node('/contour_side1')[skel_table_id, :, :] * self.microns_per_pixel
            self.dorsal_contour[ind_ff] = \
            ske_file_id.get_node('/contour_side2')[skel_table_id, :, :] * self.microns_per_pixel

            microns_per_pixel_abs = np.mean(np.abs(self.microns_per_pixel))
           
            self.widths[ind_ff] = \
            ske_file_id.get_node('/contour_width')[skel_table_id, :] * microns_per_pixel_abs
            
            
    def _h_assert_data_dim(self):
        # assertions to check the data has the proper dimensions
        fields2check = [
            'skeleton',
            'widths',
            'ventral_contour',
            'dorsal_contour']
        
        for field in fields2check:
            A = getattr(self, field)
            assert A.shape[0] == self.n_frames
            if A.ndim >= 2:
                assert A.shape[1] == self.n_segments
            if A.ndim == 3:
                assert A.shape[2] == 2
           

    def to_open_worm(self):
        def _chage_axis(x):
            A = np.rollaxis(x, 0, x.ndim)
            return np.asfortranarray(A)

        fields = [
            'skeleton',
            'widths',
            'ventral_contour',
            'dorsal_contour']
        args = [_chage_axis(getattr(self, ff)) for ff in fields]

        nw =  mv.NormalizedWorm.from_normalized_array_factory(*args)
        nw.video_info.fps = self.fps
        nw.video_info.set_ventral_mode(self.ventral_side)
        if nw.video_info.ventral_mode != 0:
            #check that the contour orientation and the ventral_mode are the same
            signed_a = nw.signed_area[np.argmax(~np.isnan(nw.signed_area))] #first element not nan
            if signed_a < 0:
                assert nw.video_info.ventral_mode == 2 #anticlockwise
            else:
                assert nw.video_info.ventral_mode == 1


        return nw

    def split(self, split_size):

        #get the indexes to made the splits
        split_ind = np.arange(split_size, self.n_frames, split_size, dtype=np.int)
        n_splits = split_ind.size + 1

        #get the fields that will be splitted, they should be ndarrays with the same number of elements in the fisrt dimension
        fields2split = [field for field, val in self.__dict__.items() if isinstance(val, np.ndarray)]
        # check all the fields have the same number of frames in the first dimension
        assert all(getattr(self, x).shape[0] for x in fields2split)
        
        #copy the main object to initialize the smaller trajectories
        base_worm =copy.copy(self)
        #delete the ndarray fields so we don't copy large amount of data twice
        [setattr(base_worm, x, np.nan) for x in fields2split]

        #split each fields
        splitted_worms = [copy.copy(base_worm) for n in range(n_splits)]
        for field in fields2split:
            splitted_field = np.split(getattr(self, field), split_ind, axis=0)
            
            for worm_s, dat_s in zip(splitted_worms, splitted_field):
                setattr(worm_s, field, dat_s)

        return splitted_worms

    @property
    def n_valid_skel(self):
        # calculate the number of valid skeletons
        return np.sum(~np.isnan(self.skeleton[:, 0, 0]))

    @property
    def n_frames(self):
        return self.timestamp.size
    
    @property
    def last_frame(self):
        return self.timestamp[-1]
    
    @property
    def first_frame(self):
        return self.timestamp[0]


    def correct_schafer_worm(self):
        if hasattr(self, 'stage_vec_inv'):
            print('The worm has been previously corrected. The attribute "stage_vec_inv" exists. ')
            return
        
        self.ventral_side = read_ventral_side(self.file_name)
        
        assert isGoodStageAligment(self.file_name)
        with tables.File(self.file_name, 'r') as fid:
            stage_vec_ori = fid.get_node('/stage_movement/stage_vec')[:]
            timestamp_ind = fid.get_node('/timestamp/raw')[:].astype(np.int)
            rotation_matrix = fid.get_node('/stage_movement')._v_attrs['rotation_matrix']
            microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
            #2D to control for the scale vector directions
            
        # let's rotate the stage movement
        dd = np.sign(microns_per_pixel_scale)
        rotation_matrix_inv = np.dot(
            rotation_matrix * [(1, -1), (-1, 1)], [(dd[0], 0), (0, dd[1])])

        # adjust the stage_vec to match the timestamps in the skeletons
        timestamp_ind = timestamp_ind
        good = (timestamp_ind >= self.first_frame) & (timestamp_ind <= self.last_frame)

        ind_ff = timestamp_ind[good] - self.first_frame
        stage_vec_ori = stage_vec_ori[good]

        stage_vec = np.full((self.timestamp.size, 2), np.nan)
        stage_vec[ind_ff, :] = stage_vec_ori
        # the negative symbole is to add the stage vector directly, instead of
        # substracting it.
        self.stage_vec_inv = -np.dot(rotation_matrix_inv, stage_vec.T).T

        for field in ['skeleton', 'ventral_contour', 'dorsal_contour']:
            if hasattr(self, field):
                tmp_dat = getattr(self, field)
                # rotate the skeletons
                # for ii in range(tot_skel):
            #tmp_dat[ii] = np.dot(rotation_matrix, tmp_dat[ii].T).T
                tmp_dat = tmp_dat + self.stage_vec_inv[:, np.newaxis, :]
                setattr(self, field, tmp_dat)
        

class WormStats():

    def __init__(self):
        '''get the info for each feature chategory'''
        
        feat_names_file = os.path.join(AUX_FILES_DIR, 'features_names.csv')
        #feat_names_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'aux', 'features_names.csv')
        
        self.extra_fields =  ['worm_index', 'n_frames', 'n_valid_skel', 'first_frame']
        self.features_info = pd.read_csv(feat_names_file, index_col=0)
        self.builtFeatAvgNames()  # create self.feat_avg_names

        # get files that would be used in the construction of objects
        self.feat_avg_dtype = [(x, np.float32) for x in self.feat_avg_names]
        self.feat_timeseries = list(
            self.features_info[
                self.features_info['is_time_series'] == 1].index.values)

        extra_fields = ['worm_index', 'timestamp', 'skeleton_id', 'motion_modes']
        timeseries_fields =  extra_fields + self.feat_timeseries
        self.feat_timeseries_dtype = [(x, np.float32) for x in timeseries_fields]

        self.feat_events = list(
            self.features_info[
                self.features_info['is_time_series'] == 0].index.values)

    def builtFeatAvgNames(self):
        feat_avg_names = self.extra_fields[:]
        for feat_name, feat_info in self.features_info.iterrows():

            motion_types = ['']
            if feat_info['is_time_series']:
                motion_types += ['_forward', '_paused', '_backward']

            for mtype in motion_types:
                sub_name = feat_name + mtype
                feat_avg_names.append(sub_name)
                if feat_info['is_signed']:
                    for atype in ['_abs', '_neg', '_pos']:
                        feat_avg_names.append(sub_name + atype)

        self.feat_avg_names = feat_avg_names

    def getFieldData(worm_features, name):
        data = worm_features
        for field in name.split('.'):
            data = getattr(data, field)
        return data

    def getWormStats(self, worm_features, stat_func=np.mean):
        ''' Calculate the statistics of an object worm features, subdividing data
            into Backward/Forward/Paused and/or Positive/Negative/Absolute, when appropiated.
            The default is to calculate the mean value, but this can be changed
            using stat_func.

            Return the feature list as an ordered dictionary.
        '''
        
        if isinstance(worm_features, dict):
            def read_feat(feat_name):
                if feat_name in worm_features:
                    return worm_features[feat_name]
                else:
                    return None
            motion_mode = read_feat('motion_modes')
        else:
            
            def read_feat(feat_name):
                feat_obj = self.features_info.loc[feat_name, 'feat_name_obj']
                if feat_obj in  worm_features._features:
                    return worm_features._features[feat_obj].value
                else:
                    return None
            motion_mode = worm_features._features['locomotion.motion_mode'].value


        # return data as a numpy recarray
        feat_stats = np.full(1, np.nan, dtype=self.feat_avg_dtype)
        
        for feat_name, feat_props in self.features_info.iterrows():
            tmp_data = read_feat(feat_name)
            if tmp_data is None:
                feat_stats[feat_name] = np.nan

            elif isinstance(tmp_data, (int, float)):
                feat_stats[feat_name] = tmp_data

            else:
                feat_avg = self._featureStat(
                    stat_func,
                    tmp_data,
                    feat_name,
                    feat_props['is_signed'],
                    feat_props['is_time_series'],
                    motion_mode)
                for feat_avg_name in feat_avg:
                    feat_stats[feat_avg_name] = feat_avg[feat_avg_name]

        return feat_stats

    @staticmethod
    def _featureStat(
            stat_func,
            data,
            name,
            is_signed,
            is_time_series,
            motion_mode=np.zeros(0)):
        # I prefer to keep this function quite independend and pass the stats and moition_mode argument
        # rather than save those values in the class
        if data is None:
            data = np.zeros(0)

        #filter nan data
        valid = ~np.isnan(data)
        data = data[valid]
        
        motion_types = OrderedDict()
        motion_types['all'] = np.nan
        if is_time_series:
            # if the the feature is motion type we can subdivide in Forward,
            # Paused or Backward motion
            motion_mode = motion_mode[valid]
            assert motion_mode.size == data.size
            
            motion_types['forward'] = motion_mode == 1
            motion_types['paused'] = motion_mode == 0
            motion_types['backward'] = motion_mode == -1

        stats = OrderedDict()
        for key in motion_types:
            
            if key == 'all':
                sub_name = name
                valid_data = data
            else:
                sub_name = name + '_' + key
                #filter by an specific motion type
                valid_data = data[motion_types[key]]

            assert not np.any(np.isnan(valid_data))
            
            stats[sub_name] = stat_func(valid_data)
            if is_signed:
                # if the feature is signed we can subdivide in positive,
                # negative and absolute
                stats[sub_name + '_abs'] = stat_func(np.abs(valid_data))

                neg_valid = (valid_data < 0)
                stats[sub_name + '_neg'] = stat_func(valid_data[neg_valid])

                pos_valid = (valid_data > 0) 
                stats[sub_name + '_pos'] = stat_func(valid_data[pos_valid])
        return stats
                
if __name__ == '__main__':
    
    main_dir = '/Users/ajaver/OneDrive - Imperial College London/Local_Videos/single_worm/global_sample_v3/'
    base_name = 'N2 on food R_2011_09_13__11_59___3___3'
    skel_file = os.path.join(main_dir, base_name + '_skeletons.hdf5')
    
    worm = WormFromTable(skel_file, 1)

