# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:33:34 2015

@author: ajaver
"""

import numpy as np
import pandas as pd
import tables


def correctTrajectories(trajectories_file, is_single_worm, join_traj_param):
    if is_single_worm:
        correctSingleWormCase(trajectories_file)
    else:
        joinTrajectories(trajectories_file, **join_traj_param)

    with tables.File(trajectories_file, "r+") as traj_fid:
        traj_fid.get_node('/plate_worms')._v_attrs['has_finished'] = 2
        traj_fid.flush()


def _validRowsByArea(plate_worms):
    # here I am assuming that most of the time the largest area in the frame is a worm. Therefore a very large area is likely to be
    # noise
    groupsbyframe = plate_worms.groupby('frame_number')
    maxAreaPerFrame = groupsbyframe.agg({'area': 'max'})
    med_area = np.median(maxAreaPerFrame)
    mad_area = np.median(np.abs(maxAreaPerFrame - med_area))
    min_area = med_area - mad_area * 6
    max_area = med_area + mad_area * 6

    groupByIndex = plate_worms.groupby('worm_index_blob')

    median_area_by_index = groupByIndex.agg({'area': np.median})

    good = ((median_area_by_index > min_area) & (
        median_area_by_index < max_area)).values
    valid_ind = median_area_by_index[good].index

    plate_worms_f = plate_worms[plate_worms['worm_index_blob'].isin(valid_ind)]

    # median location, it is likely the worm spend more time here since the
    # stage movements tries to get it in the centre of the frame
    CMx_med = plate_worms_f['coord_x'].median()
    CMy_med = plate_worms_f['coord_y'].median()
    L_med = plate_worms_f['box_length'].median()

    # let's use a threshold of movement of at most a quarter of the worm size,
    # otherwise we discard frame.
    L_th = L_med / 4

    # now if there are still a lot of valid blobs we decide by choising the
    # closest blob
    valid_rows = []
    tot_frames = plate_worms['frame_number'].max() + 1

    def get_valid_indexes(frame_number, prev_row):
        try:
            current_group_f = groupbyframe_f.get_group(frame_number)
        except KeyError:
            # there are not valid index in the current group
            prev_row = -1
            return prev_row

        # pick the closest blob if there are more than one blob to pick
        if not isinstance(prev_row, int):
            delX = current_group_f['coord_x'] - prev_row['coord_x']
            delY = current_group_f['coord_y'] - prev_row['coord_y']
        else:
            delX = current_group_f['coord_x'] - CMx_med
            delY = current_group_f['coord_y'] - CMy_med

        R = np.sqrt(delX * delX + delY * delY)
        good_ind = np.argmin(R)
        if R[good_ind] < L_th:
            prev_row = current_group_f.loc[good_ind]
            valid_rows.append(good_ind)
        else:
            prev_row = -1

        return prev_row

    # group by frame
    groupbyframe_f = plate_worms_f.groupby('frame_number')

    prev_row = -1
    first_frame = tot_frames
    for frame_number in range(tot_frames):
        prev_row = get_valid_indexes(frame_number, prev_row)
        if not isinstance(prev_row, int) and first_frame > frame_number:
            first_frame = frame_number

    # if the first_frame is larger than zero it means that it might have lost some data in from the begining
    # let's try to search again from opposite direction
    if frame_number > 0 and len(valid_rows) > 0:
        prev_row = plate_worms_f.loc[np.min(valid_rows)]
        for frame_number in range(frame_number, -1, -1):
            prev_row = get_valid_indexes(frame_number, prev_row)

    #valid_rows = list(set(valid_rows))

    return valid_rows


def correctSingleWormCase(trajectories_file):
    '''
    Only keep the object with the largest area when cosider the case of individual worms.
    '''
    with pd.HDFStore(trajectories_file, 'r') as traj_fid:
        plate_worms = traj_fid['/plate_worms']

    # emtpy table nothing to do here
    if len(plate_worms) == 0:
        return

    valid_rows = _validRowsByArea(plate_worms)

    # np.array(1, dtype=np.int32)
    plate_worms['worm_index_joined'] = np.array(-1, dtype=np.int32)
    plate_worms.loc[valid_rows, 'worm_index_joined'] = 1

    with tables.File(trajectories_file, "r+") as traj_fid:
        table_filters = tables.Filters(
            complevel=5,
            complib='zlib',
            shuffle=True,
            fletcher32=True)
        newT = traj_fid.create_table('/', 'plate_worms_t',
                                     obj=plate_worms.to_records(index=False),
                                     filters=table_filters)
        newT._v_attrs['has_finished'] = 2
        traj_fid.remove_node('/', 'plate_worms')
        newT.rename('plate_worms')


def _findNextTraj(df, area_ratio_lim, min_track_size, max_time_gap):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames
    min_track_size -- minimum tracksize accepted
    max_time_gap -- time gap between joined trajectories
    '''

    df = df[['worm_index_blob', 'frame_number',
             'coord_x', 'coord_y', 'area', 'box_length']]
    # select the first and last frame_number for each separate trajectory
    tracks_data = df[['worm_index_blob', 'frame_number']]
    tracks_data = tracks_data.groupby('worm_index_blob')
    tracks_data = tracks_data.aggregate(
        {'frame_number': [np.argmin, np.argmax, 'count']})

    # filter data only to include trajectories larger than min_track_size
    tracks_data = tracks_data[
        tracks_data['frame_number']['count'] >= min_track_size]
    valid_indexes = tracks_data.index

    # select the corresponding first and last rows of each trajectory
    first_rows = df.ix[tracks_data['frame_number']['argmin'].values]
    last_rows = df.ix[tracks_data['frame_number']['argmax'].values]
    # let's use the particle id as index instead of the row number
    last_rows.index = tracks_data['frame_number'].index
    first_rows.index = tracks_data['frame_number'].index

    #%% look for trajectories that could be join together in a small time gap
    join_frames = []
    for curr_index in valid_indexes:
        # the possible connected trajectories must have started after the end of the current trajectories,
        # within a timegap given by max_time_gap
        possible_rows = first_rows[
            (first_rows['frame_number'] > last_rows['frame_number'][curr_index]) & (
                first_rows['frame_number'] < last_rows['frame_number'][curr_index] +
                max_time_gap)]

        # the area change must be smaller than the one given by area_ratio_lim
        # it is better to use the last point change of area because we are
        # considered changes near that occur near time
        areaR = last_rows['area'][curr_index] / possible_rows['area']
        possible_rows = possible_rows[
            (areaR > area_ratio_lim[0]) & (
                areaR < area_ratio_lim[1])]

        # not valid rows left
        if len(possible_rows) == 0:
            continue

        R = np.sqrt((possible_rows['coord_x'] -
                     last_rows['coord_x'][curr_index]) ** 2 +
                    (possible_rows['coord_y'] -
                     last_rows['coord_x'][curr_index]) ** 2)

        indmin = np.argmin(R)
        # only join trajectories that move at most one worm body
        if R[indmin] <= last_rows['box_length'][curr_index]:
            #print(curr_index, indmin)
            join_frames.append((indmin, curr_index))

    relations_dict = dict(join_frames)

    return relations_dict, valid_indexes


def _joinDict2Index(worm_index_blob, relations_dict, valid_indexes):
    worm_index_joined = np.full_like(worm_index_blob, -1)

    for ind in valid_indexes:
        # seach in the dictionary for the first index in the joined trajectory
        # group
        ind_joined = ind
        while ind_joined in relations_dict:
            ind_joined = relations_dict[ind_joined]

        # replace the previous index for the root index
        worm_index_joined[worm_index_blob == ind] = ind_joined

    return worm_index_joined


def joinTrajectories(trajectories_file, min_track_size=50,
                     max_time_gap=100, area_ratio_lim=(0.67, 1.5)):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames
    min_track_size -- minimum tracksize accepted
    max_time_gap -- time gap between joined trajectories
    '''

    # check the previous step finished correctly
    with tables.open_file(trajectories_file, mode='r') as fid:
        traj_table = fid.get_node('/plate_worms')
        assert traj_table._v_attrs['has_finished'] >= 1

    #%% get the first and last rows for each trajectory. Pandas is easier of manipulate than tables.
    with pd.HDFStore(trajectories_file, 'r') as fid:
        df = fid['plate_worms'][['worm_index_blob', 'frame_number',
                                 'coord_x', 'coord_y', 'area', 'box_length']]

    relations_dict, valid_indexes = _findNextTraj(
        df, area_ratio_lim, min_track_size, max_time_gap)

    #%%
    # update worm_index_joined field
    with tables.open_file(trajectories_file, mode='r+') as fid:
        plate_worms = fid.get_node('/plate_worms')

        # read the worm_index_blob column, this is the index order that have to
        # be conserved in the worm_index_joined column
        worm_index_blob = plate_worms.col('worm_index_blob')
        worm_index_joined = _joinDict2Index(
            worm_index_blob, relations_dict, valid_indexes)

        # add the result the column worm_index_joined
        plate_worms.modify_column(
            colname='worm_index_joined',
            column=worm_index_joined)

        # flag the join data as finished
        plate_worms._v_attrs['has_finished'] = 2
        fid.flush()
