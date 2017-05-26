# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:59:08 2016

@author: ajaver
"""
import os
import errno
import sys
import fnmatch
from tierpsy.helper.misc import RESERVED_EXT
from tierpsy.helper.params.tracker_param import valid_options

def find_valid_files(root_dir, pattern_include='*', pattern_exclude=''):
    '''
    Recursively find the files in root_dir that 
    match pattern_include and do not match pattern_exclude .
    '''

    #input validation
    invalid_ext = ['*' + x for x in RESERVED_EXT]
    if not pattern_exclude:
        pattern_exclude = []
    elif not isinstance(pattern_exclude, (list, tuple)):
        pattern_exclude = [pattern_exclude]
    pattern_exclude += invalid_ext
    
    #real processing
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root_dir)
    # if there is only a string (only one pattern) let's make it a list to be
    # able to reuse the code
    if not isinstance(pattern_include, (list, tuple)):
        pattern_include = [pattern_include]
    if not isinstance(pattern_exclude, (list, tuple)):
        pattern_exclude = [pattern_exclude]

    valid_files = []
    for dpath, dnames, fnames in os.walk(root_dir):
        for fname in fnames:
            good_patterns = any(fnmatch.fnmatch(fname, dd)
                                for dd in pattern_include)
            bad_patterns = any(fnmatch.fnmatch(fname, dd)
                               for dd in pattern_exclude)
            if good_patterns and not bad_patterns:
                fullfilename = os.path.abspath(os.path.join(dpath, fname))
                assert os.path.exists(fullfilename)
                valid_files.append(fullfilename)

    return valid_files

def create_script(base_cmd, args, argkws):
    '''
    Produce string as command line arguments in the form:

    base_cmd arg1 ... --argkw_key1 argkw_val1 ...
    '''
    cmd = base_cmd + args
    for key, dat in argkws.items():
        if isinstance(dat, bool):
            if dat:
                cmd.append('--' + key)
        elif isinstance(dat, (list, tuple)):
            cmd += ['--'+key] + list(dat)
        else:
            cmd += ['--' + key, str(dat)]
    return cmd

def get_real_script_path(fullfile):
    '''get the path name that works with pyinstaller binaries'''
    try:
        base_name = os.path.splitext(os.path.basename(fullfile))[0]
        # use this directory if it is a one-file produced by pyinstaller
        script_cmd = [os.path.join(sys._MEIPASS, base_name)]
        if os.name == 'nt':
            script_cmd[0] += '.exe'
        return script_cmd
    
    except AttributeError:
        return [sys.executable, os.path.realpath(fullfile)]


def get_dflt_sequence(analysis_type, add_manual_feats=False):
    assert analysis_type in valid_options['analysis_type']
    if analysis_type == 'SINGLE_WORM_SHAFER':
        analysis_checkpoints = ['COMPRESS',
                                'COMPRESS_ADD_DATA',
                                'VID_SUBSAMPLE',
                                'TRAJ_CREATE',
                                'TRAJ_JOIN',
                                'SKE_INIT',
                                'BLOB_FEATS',
                                'SKE_CREATE',
                                'SKE_FILT',
                                'SKE_ORIENT',
                                'STAGE_ALIGMENT',
                                'CONTOUR_ORIENT', #orientation must occur before the intensity map calculation.
                                'INT_PROFILE',
                                'INT_SKE_ORIENT',
                                'FEAT_CREATE',
                                'WCON_EXPORT'
                                ]
    else:
        #default sequence "WORM"
        analysis_checkpoints = ['COMPRESS',
                                'VID_SUBSAMPLE',
                                'TRAJ_CREATE',
                                'TRAJ_JOIN',
                                'SKE_INIT',
                                'BLOB_FEATS',
                                'SKE_CREATE',
                                'SKE_FILT',
                                'SKE_ORIENT',
                                'INT_PROFILE',
                                'INT_SKE_ORIENT',
                                'FEAT_CREATE'
                                ]
    if add_manual_feats:
        analysis_checkpoints.append('FEAT_MANUAL_CREATE') 

    return analysis_checkpoints


def remove_border_checkpoints(list_of_points, last_valid, index):
    assert (index == 0) or (index == -1) #decide if start removing from the begining or the end
    if last_valid:
        if not last_valid in list_of_points:
            raise ValueError("Point {} is not valid.".format(last_valid))
        #move points until 
        while list_of_points and \
        list_of_points[index] != last_valid:
            list_of_points.pop(index)
    
    return list_of_points
