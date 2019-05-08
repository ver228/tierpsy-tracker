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

from tierpsy.helper.misc import TimeCounter, print_flush
from tierpsy.summary.process_ow import ow_plate_summary, ow_trajectories_summary, ow_plate_summary_augmented
from tierpsy.summary.process_tierpsy import tierpsy_plate_summary, tierpsy_trajectories_summary, tierpsy_plate_summary_augmented
from tierpsy import AUX_FILES_DIR

feature_files_ext = {'openworm' : ('_features.hdf5', '_feat_manual.hdf5'), 
                     'tierpsy' : ('_featuresN.hdf5', '_featuresN.hdf5')
                     }
FEAT_SET_DIR = os.path.join(AUX_FILES_DIR,'feat_sets')
feature_sets_filenames = {'tierpsy_8' : 'tierpsy_8.csv', 'tierpsy_16' : 'tierpsy_16.csv', 
                          'tierpsy_256' : 'tierpsy_256.csv', 
                          'tierpsy_2k' : 'top2k_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
                          }
valid_feature_types = list(feature_files_ext.keys())
valid_summary_types = ['plate', 'trajectory', 'plate_augmented']
valid_time_windows_connector = ':'
valid_time_windows_separator = ','
time_windows_format_explain = 'Each time window must be defined by the start time and the end time connected by \':\' (start_time:end_time). Different windows must be separated by {}.'.format(valid_time_windows_separator) 

def check_in_list(x, list_of_x, x_name):
    if not x in list_of_x:
        raise ValueError('{} invalid {}. Valid options {}.'.format(x, x_name, list_of_x))
    

def get_summary_func(feature_type, summary_type, time_windows_ints, time_units, is_manual_index, **fold_args):
    """
    Chooses the function used for the extraction of feature summaries based on the input from the GUI
    """
    if feature_type == 'tierpsy': 
        if summary_type == 'plate':
            func = partial(tierpsy_plate_summary, time_windows=time_windows_ints, time_units=time_units, is_manual_index=is_manual_index)
        elif summary_type == 'trajectory':
            func = partial(tierpsy_trajectories_summary, time_windows=time_windows_ints, time_units=time_units, is_manual_index=is_manual_index)
        elif summary_type == 'plate_augmented':
            func = partial(tierpsy_plate_summary_augmented, time_windows=time_windows_ints, time_units=time_units, is_manual_index=is_manual_index, **fold_args)
        
    elif feature_type == 'openworm':
        if summary_type == 'plate':
            func = ow_plate_summary
        elif summary_type == 'trajectory':
            func = ow_trajectories_summary
        elif summary_type == 'plate_augmented':
            func = partial(ow_plate_summary_augmented, **fold_args)
    return func

def time_windows_parser(time_windows):
    """
    EM : Converts the string input from the GUI to a list object of integers.
    Asserts that for each time window start_time<=end_time
    """
    if not time_windows.replace(' ',''):
        windows = [[0,-1]]
        return windows
    if valid_time_windows_connector not in time_windows:
        ValueError(time_windows_format_explain)
        return
    
    # Remove spaces and replace end with -1
    windows = time_windows.replace(' ','').replace('end','-1')
    # Split at ; to separate time windows, then split each non-empty time window at :
    windows = [x.split(':') for x in windows.split(valid_time_windows_separator) if x]
    # Convert to integers
    try:
        windows = [[int(x) for x in wdw] for wdw in windows]
    except ValueError:
        print_flush('Time windows input could not be converted to list of integers.'+time_windows_format_explain)
        raise
    else:
        for iwin,window in enumerate(windows):
            if window[1]!=-1:
                assert window[0]<=window[1], "The end time of time window {}/{} is smaller than the start time.".format(iwin+1,len(windows))
        return windows

def keywords_parser(keywords):
    """
    EM : Converts the string input from the GUI to a list object of strings.
    """
    # Remove spaces
    kwrds = keywords.replace(' ','')
    # Split at ',' to separate time windows, then keep non-empty words
    kwrds = [x for x in kwrds.split(',') if x]
    
    if kwrds:
        return kwrds
    else:
        return None

def feat_set_parser(feat_set):
    """
    EM : gets the full path of the file containing the selected feature set.
    """    
    if feat_set != 'all':
        feat_set_file = os.path.join(FEAT_SET_DIR,feature_sets_filenames[feat_set])
        selected_feat = pd.read_csv(feat_set_file, header=None, index_col=None)
        selected_feat = selected_feat.values.flatten().tolist()
    else:
        selected_feat = None
    return selected_feat

def make_df_filenames(fnames,time_windows_ints):
    """
    EM : Create dataframe with filename summaries and time window info for every time window
    """
    dd = tuple(zip(*enumerate(sorted(fnames))))
    df_files = [pd.DataFrame({'file_id' : dd[0], 'file_name' : dd[1]}) for x in range(len(time_windows_ints))]
    for iwin in range(len(time_windows_ints)): 
        df_files[iwin]['is_good'] = False
        df_files[iwin]['window_id'] = iwin
        df_files[iwin]['start_time'] = time_windows_ints[iwin][0]
        df_files[iwin]['end_time'] = time_windows_ints[iwin][1]    
    return df_files

def select_features(win_summaries,keywords_in,keywords_ex,selected_feat):
    if not win_summaries.empty:
        if selected_feat is not None:
            win_summaries = win_summaries[selected_feat]
        if keywords_in is not None:
            filter_col = [x for x in win_summaries.columns if any(key in x for key in keywords_in)]
            win_summaries = win_summaries[filter_col]
        if keywords_ex is not None:
            filter_col = [x for x in win_summaries.columns if any(key in x for key in keywords_ex)]
            win_summaries = win_summaries[win_summaries.columns.drop(filter_col)]
    return win_summaries
    
def calculate_summaries(root_dir, feature_type, summary_type, is_manual_index, time_windows, time_units, 
                        feat_set, keywords_include, keywords_exclude, _is_debug = False, **fold_args):
    """
    Gets input from the GUI, calls the function that chooses the type of summary 
    and runs the summary calculation for each file in the root_dir.
    """
    save_base_name = 'summary_{}_{}'.format(feature_type, summary_type)
    if is_manual_index:
        save_base_name += '_manual'
    save_base_name += '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    #check the options are valid
    check_in_list(feature_type, valid_feature_types, 'feature_type')
    check_in_list(summary_type, valid_summary_types, 'summary_type')
    
    # EM : convert time windows to list of integers in frame number units
    time_windows_ints = time_windows_parser(time_windows)

    # EM : get list of keywords to include and to exclude
    keywords_in = keywords_parser(keywords_include)
    keywords_ex = keywords_parser(keywords_exclude)
    
    # EM : get full path to feature set file
    selected_feat = feat_set_parser(feat_set)
    
    #get summary function
    # INPUT time windows time units here
    summary_func = get_summary_func(feature_type, summary_type, time_windows_ints, time_units, is_manual_index, **fold_args)
    
    #get extension of results file
    possible_ext = feature_files_ext[feature_type]
    ext = possible_ext[1] if is_manual_index else possible_ext[0]
    
    fnames = glob.glob(os.path.join(root_dir, '**', '*' + ext), recursive=True)
    if not fnames:
        print_flush('No valid files found. Nothing to do here.')
        return
    
    # EM :Make df_files list with one features_summaries dataframe per time window
    df_files = make_df_filenames(fnames,time_windows_ints)
    
    progress_timer = TimeCounter('')
    def _displayProgress(n):
        args = (n + 1, len(df_files[0]), progress_timer.get_time_str())
        dd = "Extracting features summary. File {} of {} done. Total time: {}".format(*args)
        print_flush(dd)
    
    _displayProgress(-1)
    
    # EM :Make all_summaries list with one element per time window. Each element contains 
    # the extracted feature summaries from all the files for the given time window.
    all_summaries = [[] for x in range(len(time_windows_ints))]
    for ifile, row in df_files[0].iterrows():
        fname = row['file_name']
        
        df_list = summary_func(fname)
        for iwin,df in enumerate(df_list):
            try:
                df.insert(0, 'file_id', ifile)             
                all_summaries[iwin].append(df)
            except (AttributeError, IOError, KeyError, tables.exceptions.HDF5ExtError, tables.exceptions.NoSuchNodeError):
                continue
            else:
                if not df.empty:
                    df_files[iwin].loc[ifile, 'is_good'] = True
        _displayProgress(ifile)
    
    # EM : Concatenate summaries for each window into one dataframe and select features
    for iwin in range(len(time_windows_ints)):
        all_summaries[iwin] = pd.concat(all_summaries[iwin], ignore_index=True, sort=False)
        all_summaries[iwin] = select_features(all_summaries[iwin],keywords_in,keywords_ex,selected_feat)
        
    # EM : Save results
        f1 = os.path.join(root_dir, 'filenames_{}_window_{}.csv'.format(save_base_name,iwin))
        df_files[iwin].to_csv(f1, index=False)
        
        f2 = os.path.join(root_dir,'features_{}_window_{}.csv'.format(save_base_name,iwin))
        all_summaries[iwin].to_csv(f2, index=False)
    
    out = '****************************'
    out += '\nFINISHED. Created Files:\n-> {}\n-> {}'.format(f1,f2)
    
    print_flush(out)
    
    
    return df_files, all_summaries

if __name__ == '__main__':
    root_dir = '/Users/em812/Documents/OneDrive - Imperial College London/Eleni/Tierpsy_GUI/test_results_2'
    is_manual_index = False
#    feature_type = 'tierpsy'
    feature_type = 'openworm'
    summary_type = 'plate_augmented'
#    summary_type = 'plate'
    #summary_type = 'trajectory'
    
    fold_args = dict(
                 n_folds = 2, 
                 frac_worms_to_keep = 0.8,
                 time_sample_seconds = 10*60
                 )
    
    time_windows = '0:end' #'0:end,100000:101000'
    time_units = 'frame numbers'
    feat_set = 'all' #'tierpsy_2k'
    keywords_include = ''
    keywords_exclude = '' #'curvature,velocity,norm,abs'
    
    df_files, all_summaries = calculate_summaries(root_dir, feature_type, summary_type, is_manual_index, time_windows, time_units, feat_set, keywords_include, keywords_exclude, **fold_args)
        