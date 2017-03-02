#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 19:15:50 2017

@author: ajaver
"""
import copy
import os
import math
import numpy as np
import pandas as pd
import tables
import warnings
from scipy.signal import savgol_filter

#import open_worm_analysis_toolbox as mv

def read_fps(skeletons_file, min_allowed_fps=1, dflt_fps=25):
        # try to infer the fps from the timestamp
    try:
        with tables.File(skeletons_file, 'r') as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]

            if np.all(np.isnan(timestamp_time)):
                raise ValueError
            fps = 1 / np.nanmedian(np.diff(timestamp_time))

            if np.isnan(fps) or fps < 1:
                raise ValueError
            is_default_timestamp = 0

    except (tables.exceptions.NoSuchNodeError, IOError, ValueError):
        with tables.File(skeletons_file, 'r') as fid:
            node = fid.get_node('/trajectories_data')
            if 'expected_fps' in node._v_attrs:
                fps = node._v_attrs['expected_fps']
            else:
                fps = dflt_fps #default in old videos
            is_default_timestamp = 1

    return fps, is_default_timestamp

def read_microns_per_pixel(skeletons_file):
    # these function are related with the singleworm case it might be necesary to change them in the future
    try:
        with tables.File(skeletons_file, 'r') as fid:
            microns_per_pixel_scale = fid.get_node('/stage_movement')._v_attrs['microns_per_pixel_scale']
    except (KeyError, tables.exceptions.NoSuchNodeError):
        return 1

    if microns_per_pixel_scale.size == 2:
        assert np.abs(
            microns_per_pixel_scale[0]) == np.abs(
            microns_per_pixel_scale[1])
        microns_per_pixel_scale = np.abs(microns_per_pixel_scale[0])
        return microns_per_pixel_scale
    else:
        return 1


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
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.
    In the future it might become a method of NormalizedWorm,
    but for the moment let's test it as a separate identity
    """

    def __init__(self, file_name, worm_index, use_skel_filter=True,
                 worm_index_str='worm_index_joined',
                 smooth_window=-1, POL_DEGREE_DFLT=3):
        # Populates an empty normalized worm.
        self.microns_per_pixel = read_microns_per_pixel(file_name)
        self.fps, self.is_default_timestamp = read_fps(file_name)
        
        #fields requiered by NormalizedWorm (will be filled in readSkeletonsData)
        self.timestamp = None
        self.skeleton_id = None
        self.skeleton = None
        self.ventral_contour = None
        self.dorsal_contour = None
        self.widths = None

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

        
        self.read_data(self.microns_per_pixel)

        # smooth data if required
        if self.smooth_window > self.POL_DEGREE_DFLT:
            # print('Smoothing...')
            self.skeleton = _h_smooth_curve_all(
                self.skeleton, window=self.smooth_window)
            self.widths = _h_smooth_curve_all(
                self.widths, window=self.smooth_window)

        
        # assert the dimenssions of the read data are correct
        self.assert_data_dim()

    def _h_get_table_indexes(self):
        '''
        Get the relevant info from the trajectory_data table for a single worm. skeleton_id, timestamp.
        '''
        # intialize just to make clear the relevant variables for this function

        with pd.HDFStore(self.file_name, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']

            # get the rows of valid skeletons
            assert self.worm_index_str in trajectories_data
            good = trajectories_data[self.worm_index_str] == self.worm_index

            trajectories_data = trajectories_data.loc[good]

            skel_table_id = trajectories_data['skeleton_id'].values

            try:
                # try to read the time stamps, if there are repeated or not a
                # number use the frame nuber instead
                timestamp_raw = trajectories_data['timestamp_raw'].values
                if np.any(np.isnan(timestamp_raw)) or np.any(np.diff(timestamp_raw) == 0):
                    raise ValueError
                else:
                    timestamp_raw = timestamp_raw.astype(np.int)
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


    def read_data(self, microns_per_pixel):
        skel_table_id, timestamp_inds = self.get_table_indexes()
        
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
        self.timestamp = np.full(n_frames, -1, np.int32)
        self.skeleton_id = np.full(n_frames, -1, np.int32)
        self.timestamp[ind_ff] = timestamp_inds
        self.skeleton_id[ind_ff] = skel_table_id

        # initialize the rest of the arrays
        self.skeleton = np.full((n_frames, self.n_segments, 2), np.nan)
        self.ventral_contour = np.full(
            (n_frames, self.n_segments, 2), np.nan)
        self.dorsal_contour = np.full((n_frames, self.n_segments, 2), np.nan)
        self.length = np.full(n_frames, np.nan)

        # read data from the skeletons table
        with tables.File(self.file_name, 'r') as ske_file_id:
            #print('reading skeletons...')
            self.skeleton[ind_ff] = ske_file_id.get_node(
                '/skeleton')[skel_table_id, :, :] * microns_per_pixel

            microns_per_pixel_abs = np.mean(np.abs(microns_per_pixel))
            self.widths[ind_ff] = ske_file_id.get_node(
                '/contour_width')[skel_table_id, :] * microns_per_pixel_abs
            
            #print('reading ventral contours...')
            self.ventral_contour[ind_ff] = ske_file_id.get_node('/contour_side1')[skel_table_id, :, :] * microns_per_pixel

            #print('reading dorsal contours...')
            self.dorsal_contour[ind_ff] = ske_file_id.get_node('/contour_side2')[skel_table_id, :, :] * microns_per_pixel


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
    



    def assert_data_dim(self):
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
           

    def changeAxis(self):
        fields2change = [
            'skeleton',
            'widths',
            'angles',
            'ventral_contour',
            'dorsal_contour']

        for field in fields2change:
            A = getattr(self, field)
            # roll axis to have it as 49x2xn
            A = np.rollaxis(A, 0, A.ndim)
            # change the memory order so the last dimension changes slowly
            #A = np.asfortranarray(A)
            # finally copy it back to the field
            setattr(self, field, A)

            #assert getattr(self, field).shape[0] == self.n_segments

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