#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:19:07 2017

@author: ajaver
"""

import glob
import os
import tables

from tierpsy.helper.misc import RESERVED_EXT
from tierpsy.helper.params import set_unit_conversions, read_unit_conversions
from tierpsy import DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from tierpsy.helper.params import TrackerParams

#script to correct a previous bug in how the expected_fps, microns_per_pixel are saved.

params = TrackerParams(os.path.join(DFLT_PARAMS_PATH, '_TEST.json'))
expected_fps = params.p_dict['expected_fps']
microns_per_pixel = params.p_dict['microns_per_pixel']

#main_dir = '/Volumes/behavgenom_archive$/Adam/screening'
#fnames = glob.glob(os.path.join(main_dir, '**', '*.hdf5'), recursive=True)

dname = '/Volumes/behavgenom_archive$/Ida/test_3/**/*.hdf5'
fnames = glob.glob(dname, recursive=True)

masked_files = [x for x in fnames if not any(x.endswith(ext) for ext in RESERVED_EXT)]
skeletons_files = [x for x in fnames if x.endswith('_skeletons.hdf5')]


def change_attrs(fname, field_name):
    print(os.path.basename(fname))
    read_unit_conversions(fname)
    with tables.File(fname, 'r+') as fid:
        group_to_save = fid.get_node(field_name)
        set_unit_conversions(group_to_save, 
                             expected_fps=expected_fps, 
                             microns_per_pixel=microns_per_pixel)
        
    read_unit_conversions(fname)


for skeletons_file in masked_files:
    change_attrs(skeletons_file, '/mask')
 