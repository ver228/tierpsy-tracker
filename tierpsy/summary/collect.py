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

feature_files_ext = {'openworm' : ('_features.hdf5', '_feat_manual.hdf5'), 
                     'tierpsy' : ('_featuresN.hdf5', '_featuresN.hdf5')
                     }
valid_feature_types = list(feature_files_ext.keys())
valid_summary_types = ['plate', 'trajectory', 'plate_augmented']

def check_in_list(x, list_of_x, x_name):
    if not x in list_of_x:
        raise ValueError('{} invalid {}. Valid options {}.'.format(x, x_name, list_of_x))
    

def get_summary_func(feature_type, summary_type, is_manual_index, **fold_args):
    if feature_type == 'tierpsy': 
        if summary_type == 'plate':
            func = partial(tierpsy_plate_summary, is_manual_index=is_manual_index)
        elif summary_type == 'trajectory':
            func = partial(tierpsy_trajectories_summary, is_manual_index=is_manual_index)
        elif summary_type == 'plate_augmented':
            func = partial(tierpsy_plate_summary_augmented, is_manual_index=is_manual_index, **fold_args)
        
    elif feature_type == 'openworm':
        if summary_type == 'plate':
            func = ow_plate_summary
        elif summary_type == 'trajectory':
            func = ow_trajectories_summary
        elif summary_type == 'plate_augmented':
            func = partial(ow_plate_summary_augmented, **fold_args)
    return func


def calculate_summaries(root_dir, feature_type, summary_type, is_manual_index, _is_debug = False, **fold_args):
    save_base_name = 'summary_{}_{}'.format(feature_type, summary_type)
    if is_manual_index:
        save_base_name += '_manual'
    save_base_name += '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    #check the options are valid
    check_in_list(feature_type, valid_feature_types, 'feature_type')
    check_in_list(summary_type, valid_summary_types, 'summary_type')
    
    summary_func = get_summary_func(feature_type, summary_type, is_manual_index, **fold_args)
    
    possible_ext = feature_files_ext[feature_type]
    ext = possible_ext[1] if is_manual_index else possible_ext[0]
    
    fnames = glob.glob(os.path.join(root_dir, '**', '*' + ext), recursive=True)
    if not fnames:
        print_flush('Not valid files found. Nothing to do here.')
        return
    
    dd = tuple(zip(*enumerate(sorted(fnames))))
    df_files = pd.DataFrame({'file_id' : dd[0], 'file_name' : dd[1]})
    df_files['is_good'] = False
    
    progress_timer = TimeCounter('')
    def _displayProgress(n):
        args = (n + 1, len(df_files), progress_timer.get_time_str())
        dd = "Extracting features summary. File {} of {} done. Total time: {}".format(*args)
        print_flush(dd)
    
    _displayProgress(-1)
    all_summaries = []
    for ifile, row in df_files.iterrows():
        fname = row['file_name']
        try:
            df = summary_func(fname)
            df.insert(0, 'file_id', ifile)             
            all_summaries.append(df)
        except (AttributeError, IOError, KeyError, tables.exceptions.HDF5ExtError, tables.exceptions.NoSuchNodeError):
            continue
        
        df_files.loc[ifile, 'is_good'] = True
        _displayProgress(ifile)
        
    all_summaries = pd.concat(all_summaries, ignore_index=True, sort=False)
    
    
    f1 = os.path.join(root_dir, 'filenames_{}.csv'.format(save_base_name))
    df_files.to_csv(f1, index=False)
    
    f2 = os.path.join(root_dir,'features_{}.csv'.format(save_base_name))
    all_summaries.to_csv(f2, index=False)
    
    out = '****************************'
    out += '\nFINISHED. Created Files:\n-> {}\n-> {}'.format(f1,f2)
    
    print_flush(out)
    
    
    return df_files, all_summaries

if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples'
    is_manual_index = False
    #feature_type = 'tierpsy'
    feature_type = 'openworm'
    #summary_type = 'plate_augmented'
    summary_type = 'plate'
    
    fold_args = dict(
                 n_folds = 2, 
                 frac_worms_to_keep = 0.8,
                 time_sample_seconds = 10*60
                 )
    
    df_files, all_summaries = calculate_summaries(root_dir, feature_type, summary_type, is_manual_index, **fold_args)
        