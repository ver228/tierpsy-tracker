#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:56:17 2016

@author: ajaver
"""
import json

import numpy as np
import pandas as pd
import tables
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.signal import savgol_filter

from tierpsy.analysis.compress.extractMetaData import read_and_save_timestamp
from tierpsy.helper.params import copy_unit_conversions, ske_init_defaults
from tierpsy.analysis.ske_init.filterTrajectModel import filterModelWorms
from tierpsy.helper.misc import TABLE_FILTERS

def getSmoothedTraj(trajectories_file,
                    min_track_size=100,
                    displacement_smooth_win=101,
                    threshold_smooth_win=501,
                    roi_size = -1):
    '''
    Organize the data produced in the trajectories analysis to filter the main trajctories data.
    It will sort the table by index, determine a sensible roi_size, interpolate missing points,
    and smooth the data.
    '''

    def _read_plate_worms(trajectories_file):
        # read that frame an select trajectories that were considered valid by
        # join_trajectories
        #fields that must be included in plate_worms
        plate_worms_fields=  ['worm_index_joined',
                    'frame_number',
                    'coord_x',
                    'coord_y',
                    'threshold',
                    'bounding_box_xmax',
                    'bounding_box_xmin',
                    'bounding_box_ymax',
                    'bounding_box_ymin',
                    'area']

        with pd.HDFStore(trajectories_file, 'r') as table_fid:
            df = table_fid['/plate_worms'][plate_worms_fields]
            df = df[df['worm_index_joined'] > 0]


        with tables.File(trajectories_file, 'r') as fid:
            timestamp_raw = fid.get_node('/timestamp/raw')[:]
            timestamp_time = fid.get_node('/timestamp/time')[:]

        #exit if the timestamp does not make sense
        if len(timestamp_raw) < df['frame_number'].max():
            raise Exception(
                'bad %i, %i. \nFile: %s' %
                (len(timestamp_raw),
                 df['frame_number'].max(),
                    trajectories_file))
        return df, timestamp_raw, timestamp_time

    def _get_roi_size(df):
        # calculate the ROI size as the maximum bounding box size for a given trajectory
        bb_x = df['bounding_box_xmax'] - df['bounding_box_xmin'] + 1
        bb_y = df['bounding_box_ymax'] - df['bounding_box_ymin'] + 1
        worm_lim = pd.concat([bb_x, bb_y], axis=1).max(axis=1)

        df_bb = pd.DataFrame(
            {'worm_index_joined': df['worm_index_joined'], 'roi_range': worm_lim})
        roi_range = df_bb.groupby('worm_index_joined').agg(max) + 10
        roi_range = dict(roi_range['roi_range'])

        return roi_range

    def _get_total_number_rows(df, min_track_size):
        # caluculate the total number of rows that will be used by the table.
        # we need this number to reserve space in the recarray
        # get the total length of each track, this is more accurate than using
        # count since parts of the track could have got lost for a few frames

        if df.size == 0:
            return 0


        tracks_data = df.groupby('worm_index_joined').aggregate(['max', 'min'])
        track_lenghts = (
            tracks_data['frame_number']['max'] -
            tracks_data['frame_number']['min'] +
            1)
        tot_num_rows = track_lenghts[track_lenghts > min_track_size].sum()


        return tot_num_rows

    # a track size less than 2 will break the interp_1 function
    if min_track_size < 2:
        min_track_size = 2

    # the filter window must be odd
    if displacement_smooth_win % 2 == 0:
        displacement_smooth_win += 1

    df, timestamp_raw, timestamp_time = _read_plate_worms(trajectories_file)
    roi_range = _get_roi_size(df)
    tot_num_rows = _get_total_number_rows(df, min_track_size)

    # initialize output data as a numpy recarray (pytables friendly format)
    trajectories_df = np.recarray(tot_num_rows, dtype=[('frame_number', np.int32),
                                                       ('worm_index_joined', np.int32),
                                                       ('plate_worm_id', np.int32),
                                                       ('skeleton_id', np.int32),
                                                       ('coord_x', np.float32),
                                                       ('coord_y', np.float32),
                                                       ('threshold', np.float32),
                                                       ('has_skeleton', np.uint8),
                                                       ('roi_size', np.float32),
                                                       ('area', np.float32),
                                                       ('timestamp_raw', np.float32),
                                                       ('timestamp_time', np.float32)])

    # store the maximum and minimum frame of each worm
    worms_frame_range = {}

    # smooth trajectories (reduce giggling from the CM to obtain a nicer video)
    # interpolate for possible missing frames in the trajectories
    curr_rows = 0
    for worm_index, worm_data in df.groupby('worm_index_joined'):
        worm_data = worm_data[['coord_x', 'coord_y', 'frame_number', 'threshold', 'area']]
        worm_data = worm_data.drop_duplicates(subset='frame_number')

        x = worm_data['coord_x'].values
        y = worm_data['coord_y'].values
        t = worm_data['frame_number'].values.astype(np.int)
        thresh = worm_data['threshold'].values
        area = worm_data['area'].values

        first_frame = np.min(t)
        last_frame = np.max(t)
        worms_frame_range[worm_index] = (first_frame, last_frame)

        tnew = np.arange(first_frame, last_frame + 1, dtype=np.int32)

        if len(tnew) <= min_track_size:
            continue

        #add a random shift in case there is a duplicated value (interp1 will produce a nan otherwise)
        delt = np.diff(t)
        if np.any(delt == 0):
        	#Ugly patch
            raise ValueError('Time frame duplicate?')


        # iterpolate missing points in the trajectory and smooth data using the
        # savitzky golay filter
        fx = interp1d(t, x)
        fy = interp1d(t, y)
        xnew = fx(tnew)
        ynew = fy(tnew)


        farea = interp1d(t, area)
        areanew = farea(tnew)

        fthresh = interp1d(t, thresh)
        threshnew = fthresh(tnew)

        if len(tnew) > displacement_smooth_win and displacement_smooth_win > 3:

            xnew = savgol_filter(xnew, displacement_smooth_win, 3)
            ynew = savgol_filter(ynew, displacement_smooth_win, 3)
            areanew = median_filter(areanew, displacement_smooth_win)

        # smooth the threshold (the worm intensity shouldn't change abruptly
        # along the trajectory)
        if len(tnew) > threshold_smooth_win:
            threshnew = median_filter(threshnew, threshold_smooth_win)

        #we use skeleton_id to add the data into the correct position in the trajectories_data
        new_total = curr_rows + xnew.size
        skeleton_id = np.arange(curr_rows, new_total, dtype=np.int32)
        curr_rows = new_total



        # store the indexes in the original plate_worms table
        plate_worm_id = np.full(tnew.size, -1, dtype=np.int32)

        plate_worm_id[t - first_frame] = worm_data.index

        trajectories_df['worm_index_joined'][skeleton_id] = worm_index
        trajectories_df['coord_x'][skeleton_id] = xnew
        trajectories_df['coord_y'][skeleton_id] = ynew

        frame_number = np.arange(first_frame, last_frame + 1, dtype=np.int32)
        trajectories_df['frame_number'][skeleton_id] = frame_number
        trajectories_df['timestamp_raw'][skeleton_id] = timestamp_raw[frame_number]
        trajectories_df['timestamp_time'][skeleton_id] = timestamp_time[frame_number]

        trajectories_df['threshold'][skeleton_id] = threshnew
        trajectories_df['plate_worm_id'][skeleton_id] = plate_worm_id
        trajectories_df['skeleton_id'][skeleton_id] = skeleton_id
        trajectories_df['has_skeleton'][skeleton_id] = False
        trajectories_df['roi_size'][skeleton_id] = roi_range[worm_index]

        trajectories_df['area'][skeleton_id] = areanew

    assert curr_rows == tot_num_rows
    trajectories_data = pd.DataFrame(trajectories_df)

    if roi_size > 0:
        trajectories_data['roi_size'] = roi_size

    return trajectories_data



def saveTrajData(trajectories_data, masked_image_file, skeletons_file):
    #save data into the skeletons file
    with tables.File(skeletons_file, "a") as ske_file_id:
        trajectories_data_f = ske_file_id.create_table(
            '/',
            'trajectories_data',
            obj=trajectories_data.to_records(index=False),
            filters=TABLE_FILTERS)

        plate_worms = ske_file_id.get_node('/plate_worms')
        if 'bgnd_param' in plate_worms._v_attrs:
            bgnd_param = plate_worms._v_attrs['bgnd_param']
        else:
            bgnd_param = bytes(json.dumps({}), 'utf-8') #default empty

        trajectories_data_f._v_attrs['bgnd_param'] = bgnd_param
        #read and the units information information
        fps, microns_per_pixel, is_light_background = \
        copy_unit_conversions(trajectories_data_f, masked_image_file)

        if not '/timestamp' in ske_file_id:
            read_and_save_timestamp(masked_image_file, skeletons_file)

        ske_file_id.flush()


def processTrajectoryData(
        skeletons_file,
        masked_image_file,
        trajectories_file,
        smoothed_traj_param,
        min_track_size=1,  # probably useless
        displacement_smooth_win=-1,
        threshold_smooth_win=-1,
        roi_size=-1,
        filter_model_name=''):
    '''
    Initialize the skeletons by creating the table `/trajectories_data`.
    This table is used by the GUI and by all the subsequent functions.
    filter_model_path - name of the pretrainned keras model to used to
                        filter worms from spurius blobs.
                        The file must be stored in the `tierpsy/aux` directory.
                        If the variable is empty this step will be ignored.
    '''

    smoothed_traj_param = ske_init_defaults(masked_image_file,
                                            **smoothed_traj_param)

    trajectories_data = getSmoothedTraj(trajectories_file,
                                        **smoothed_traj_param)
    if filter_model_name:
        trajectories_data = filterModelWorms(masked_image_file,
                                             trajectories_data,
                                             filter_model_name)

    saveTrajData(trajectories_data, masked_image_file, skeletons_file)


if __name__ == '__main__':
    import os
    from tierpsy.helper.params.models_path import get_model_filter_worms
    # root_dir = '/Users/ajaver/OneDrive - Imperial College London/'
    # root_dir += 'Local_Videos/test_messy/'
    root_dir = '/Users/lferiani/Hackathon/multiwell_tierpsy/'
    root_dir += '3_TRAJ_JOIN/MaskedVideos/20191205'

    ff = 'syngenta_screen_run1_bluelight_20191205_151104.22956805/'
    ff += 'metadata.hdf5'

    model_path = get_model_filter_worms('pytorch_default')

    masked_image_file = os.path.join(root_dir, ff)
    skeletons_file = masked_image_file.replace('.hdf5', '_skeletons.hdf5')
    skeletons_file = skeletons_file.replace('MaskedVideos', 'Results')
    trajectories_file = masked_image_file.replace('.hdf5', '_skeletons.hdf5')
    trajectories_file = trajectories_file.replace('MaskedVideos', 'Results')

    # restore backed up
    import shutil
    shutil.copy(skeletons_file.replace('.hdf5', '.bak'), skeletons_file)

    processTrajectoryData(skeletons_file, masked_image_file, trajectories_file,
                          smoothed_traj_param={},
                          filter_model_name=model_path)
