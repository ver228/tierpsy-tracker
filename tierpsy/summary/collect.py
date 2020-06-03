#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: avelinojaver
"""
from functools import partial
import os
import glob
import datetime
import tables
import pandas as pd
import numpy as np

from tierpsy.helper.misc import TimeCounter, print_flush
from tierpsy.summary.process_ow import ow_plate_summary, \
    ow_trajectories_summary, ow_plate_summary_augmented
from tierpsy.summary.process_tierpsy import tierpsy_plate_summary, \
    tierpsy_trajectories_summary, tierpsy_plate_summary_augmented
from tierpsy.summary.helper import \
    get_featsum_headers, get_fnamesum_headers, shorten_feature_names
from tierpsy.summary.parsers import \
    time_windows_parser, filter_args_parser, select_parser

feature_files_ext = {'openworm' : ('_features.hdf5', '_feat_manual.hdf5'),
                     'tierpsy' : ('_featuresN.hdf5', '_featuresN.hdf5')
                     }

valid_feature_types = list(feature_files_ext.keys())
valid_summary_types = ['plate', 'trajectory', 'plate_augmented']

feat_df_id_cols = \
    ['file_id', 'i_fold', 'worm_index', 'n_skeletons', 'well_name', 'is_good_well']

def check_in_list(x, list_of_x, x_name):
    if not x in list_of_x:
        raise ValueError(
            '{} invalid {}. Valid options {}.'.format(x, x_name, list_of_x)
            )

def get_summary_func(
        feature_type, summary_type,
        time_windows_ints, time_units,
        selected_feat,
        dorsal_side_known, filter_params,
        is_manual_index, **fold_args
        ):
    """
    Chooses the function used for the extraction of feature summaries based on
    the input from the GUI
    """
    if feature_type == 'tierpsy':
        if summary_type == 'plate':
            func = partial(
                tierpsy_plate_summary,
                time_windows=time_windows_ints, time_units=time_units,
                only_abs_ventral = not dorsal_side_known,
                selected_feat = selected_feat,
                is_manual_index=is_manual_index,
                filter_params = filter_params
                )
        elif summary_type == 'trajectory':
            func = partial(
                tierpsy_trajectories_summary,
                time_windows=time_windows_ints, time_units=time_units,
                only_abs_ventral = not dorsal_side_known,
                selected_feat = selected_feat,
                is_manual_index=is_manual_index,
                filter_params = filter_params
                )
        elif summary_type == 'plate_augmented':
            func = partial(
                tierpsy_plate_summary_augmented,
                time_windows=time_windows_ints, time_units=time_units,
                only_abs_ventral = not dorsal_side_known,
                selected_feat = selected_feat,
                is_manual_index=is_manual_index,
                filter_params = filter_params,
                **fold_args
                )

    elif feature_type == 'openworm':
        if summary_type == 'plate':
            func = ow_plate_summary
        elif summary_type == 'trajectory':
            func = ow_trajectories_summary
        elif summary_type == 'plate_augmented':
            func = partial(ow_plate_summary_augmented, **fold_args)
    return func


def sort_columns(df, selected_feat):
    """
    Sorts the columns of the feat summaries dataframe to make sure that each
    line written in the features summaries file contains the same features with
    the same order. If a feature has not been calculated for a specific file,
    then a nan column is added in its place.
    """

    not_existing_cols = [col for col in selected_feat if col not in df.columns]

    if len(not_existing_cols) > 0:
        for col in not_existing_cols:
            df[col] = np.nan

    df = df[[x for x in feat_df_id_cols if x in df.columns] + selected_feat]

    return df

def make_df_filenames(fnames):
    """
    EM : Create dataframe with filename summaries and time window info for
    every time window
    """
    dd = tuple(zip(*enumerate(sorted(fnames))))
    df_files = pd.DataFrame({'file_id' : dd[0], 'filename' : dd[1]})
    df_files['is_good'] = False
    return df_files


def calculate_summaries(
        root_dir, feature_type, summary_type, is_manual_index,
        abbreviate_features, dorsal_side_known,
        time_windows='0:end', time_units=None,
        select_feat='all', keywords_include='', keywords_exclude='',
        _is_debug = False, append_to_file=None, **kwargs
        ):
    """
    Gets input from the GUI, calls the function that chooses the type of
    summary and runs the summary calculation for each file in the root_dir.
    """
    filter_args = {k:kwargs[k] for k in kwargs.keys() if 'filter' in k}
    fold_args = {k:kwargs[k] for k in kwargs.keys() if 'filter' not in k}

    #check the options are valid
    check_in_list(feature_type, valid_feature_types, 'feature_type')
    check_in_list(summary_type, valid_summary_types, 'summary_type')

    # EM : convert time windows to list of integers in frame number units
    time_windows_ints = time_windows_parser(time_windows)
    filter_params = filter_args_parser(filter_args)
    # EM: get lists of strings (in a tuple) defining the feature selection
    # from keywords_in,
    # keywords_ex and select_feat.
    selected_feat = select_parser(
        feature_type, keywords_include, keywords_exclude, select_feat, dorsal_side_known)

    #get summary function
    # INPUT time windows time units here
    summary_func = get_summary_func(
        feature_type, summary_type,
        time_windows_ints, time_units,
        selected_feat,
        dorsal_side_known, filter_params,
        is_manual_index, **fold_args)

    # get basenames of summary files
    save_base_name = 'summary_{}_{}'.format(feature_type, summary_type)
    if is_manual_index:
        save_base_name += '_manual'
    save_base_name += '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    #get extension of results file
    possible_ext = feature_files_ext[feature_type]
    ext = possible_ext[1] if is_manual_index else possible_ext[0]

    fnames = glob.glob(os.path.join(root_dir, '**', '*' + ext), recursive=True)
    if not fnames:
        print_flush('No valid files found. Nothing to do here.')
        return None,None

    # EM :Make df_files dataframe with filenames and file ids
    df_files = make_df_filenames(fnames)

    # EM : Create features_summaries and filenames_summaries files
    #      and write headers
    fnames_files = []
    featsum_files = []
    for iwin in range(len(time_windows_ints)):
        # EM : Create features_summaries and filenames_summaries files
        if select_feat != 'all':
            win_save_base_name = save_base_name.replace(
                'tierpsy',select_feat+'_tierpsy')
        else:
            win_save_base_name = save_base_name

        if not (len(time_windows_ints)==1 and time_windows_ints[0]==[0,-1]):
            win_save_base_name = win_save_base_name+'_window_{}'.format(iwin)

        f1 = os.path.join(
            root_dir, 'filenames_{}.csv'.format(win_save_base_name))
        f2 = os.path.join(
            root_dir,'features_{}.csv'.format(win_save_base_name))

        fnamesum_headers = get_fnamesum_headers(
            f2, feature_type, summary_type, iwin, time_windows_ints[iwin],
            time_units, len(time_windows_ints), select_feat, filter_params,
            fold_args, df_files.columns.to_list())
        featsum_headers = get_featsum_headers(f1)

        with open(f1,'w') as fid:
            fid.write(fnamesum_headers)

        with open(f2,'w') as fid:
            fid.write(featsum_headers)

        fnames_files.append(f1)
        featsum_files.append(f2)

    progress_timer = TimeCounter('')
    def _displayProgress(n):
        args = (n + 1, len(df_files), progress_timer.get_time_str())
        dd = "Extracting features summary. "
        dd += "File {} of {} done. Total time: {}".format(*args)
        print_flush(dd)

    _displayProgress(-1)

    # EM : Extract feature summaries from all the files for all time windows.
    is_featnames_written = [False for i in range(len(time_windows_ints))]

    for ifile,row in df_files.iterrows():
        fname = row['filename']
        file_id = row['file_id']

        summaries_per_win = summary_func(fname)

        for iwin,df in enumerate(summaries_per_win):

            f1 = fnames_files[iwin]
            f2 = featsum_files[iwin]

            try:
                df.insert(0, 'file_id', file_id)
                df = sort_columns(df, selected_feat)
            except (AttributeError, IOError, KeyError,
                    tables.exceptions.HDF5ExtError,
                    tables.exceptions.NoSuchNodeError):
                continue
            else:
                # Get the filename summary line
                filenames = row.copy()
                if not df.empty:
                    filenames['is_good'] = True
                # Store the filename summary line
                with open(f1,'a') as fid:
                    fid.write(','.join([str(x)
                                        for x in filenames.values])+"\n")

                if not df.empty:
                    # Abbreviate names
                    if abbreviate_features:
                        df = shorten_feature_names(df)

                    # Store line(s) of features summaries for the given file
                    # and given window
                    with open(f2,'a') as fid:
                        if not is_featnames_written[iwin]:
                            df.to_csv(fid, header=True, index=False)
                            is_featnames_written[iwin] = True
                        else:
                            df.to_csv(fid, header=False, index=False)


        _displayProgress(ifile)

    out = '****************************'
    out += '\nFINISHED. Created Files:'
    for f1,f2 in zip(fnames_files,featsum_files):
        out += '\n-> {}\n-> {}'.format(f1,f2)

    print_flush(out)


    return df_files

if __name__ == '__main__':

    root_dir = \
        '/Users/em812/Data/Tierpsy_GUI/test_results_multiwell/Syngenta/Results'
        # '/Users/em812/Data/Tierpsy_GUI/test_results_2'
        #'/Users/em812/Data/Tierpsy_GUI/test_results_multiwell/20190808_subset'

    is_manual_index = False
    feature_type = 'tierpsy'
    # feature_type = 'openworm'
    # summary_type = 'plate_augmented'
    summary_type = 'plate'
    #summary_type = 'trajectory'

# Luigi
#    root_dir = \
#        '/Users/lferiani/Desktop/Data_FOVsplitter/evgeny/Results/20190808_subset'
#    is_manual_index = False
#    feature_type = 'tierpsy'
#    #feature_type = 'openworm'
#    #summary_type = 'plate_augmented'
##    summary_type = 'plate'
#    summary_type = 'trajectory'

    # fold_args = dict(
    #              n_folds = 2,
    #              frac_worms_to_keep = 0.8,
    #              time_sample_seconds = 10*60
    #              )
    kwargs = {
        'filter_time_min': '100',
        'filter_travel_min': '1400',
        'filter_time_units': 'frame_numbers',
        'filter_distance_units': 'pixels',
        'filter_length_min': '50',
        'filter_length_max': '200',
        'filter_width_min': '2',
        'filter_width_max': '15'
        }

    time_windows = '0:100+200:300+350:400, 150:200' #'0:end:1000' #'0:end' #
    time_units = 'seconds'
    select_feat = 'tierpsy_16' #'tierpsy_2k'
    keywords_include = ''
    keywords_exclude = 'blob' #'curvature,velocity,norm,abs'
    abbreviate_features = False
    dorsal_side_known = False

    df_files = calculate_summaries(
        root_dir, feature_type, summary_type, is_manual_index,
        abbreviate_features, dorsal_side_known,
        time_windows=time_windows, time_units=time_units,
        select_feat=select_feat, keywords_include=keywords_include,
        keywords_exclude=keywords_exclude,
        _is_debug = False, **kwargs)
        #**fold_args)

    # Luigi
#    df_files, all_summaries = calculate_summaries(
#         root_dir, feature_type, summary_type, is_manual_index,
#         time_windows, time_units, **fold_args)
