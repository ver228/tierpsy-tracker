#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:51:07 2019

@author: lferiani
"""


import glob
import os
import tables

from tierpsy.helper.misc import RESERVED_EXT
from tierpsy.helper.params import set_unit_conversions, read_unit_conversions
from tierpsy import DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from tierpsy.helper.params import TrackerParams

#script to correct a previous bug in how the expected_fps, microns_per_pixel are saved.
# actually let's first check if files have gone bad!

#%%

params = TrackerParams(os.path.join(DFLT_PARAMS_PATH, '_AEX_RIG.json'))
expected_fps = params.p_dict['expected_fps']
microns_per_pixel = params.p_dict['microns_per_pixel']

#%%
#main_dir = '/Volumes/behavgenom_archive$/Adam/screening'
#fnames = glob.glob(os.path.join(main_dir, '**', '*.hdf5'), recursive=True)

#dname = '/Volumes/behavgenom_archive$/Ida/test_3/**/*.hdf5'
#dname = '/Volumes/behavgenom_archive$/Ida/LoopBio_rig/180222_blue_light/3/**/*.hdf5'
#dname = '/Volumes/behavgenom$/Bertie/singleplatequiescence/**/*.hdf5'
#fnames = glob.glob(dname, recursive=True)
#
#masked_files = [x for x in fnames if not any(x.endswith(ext) for ext in RESERVED_EXT)]
#skeletons_files = [x for x in fnames if x.endswith('_skeletons.hdf5')]

mv_dname = '/Volumes/behavgenom$/Bertie/singleplatequiescence/MaskedVideos/**/*.hdf5'
fnames = glob.glob(mv_dname, recursive=True)
masked_files = [x for x in fnames if not any(x.endswith(ext) for ext in RESERVED_EXT)]

r_dname = '/Volumes/behavgenom$/Bertie/singleplatequiescence/Results/**/*.hdf5'
r_fnames = glob.glob(r_dname, recursive=True)
skeletons_files = [x for x in r_fnames if x.endswith('_skeletons.hdf5')]


#%% check inconsistencies
print('MaskedVideos without skeletons:')
for f in masked_files:
    foo = f.replace('MaskedVideos','Results')
    foo = foo.replace('.hdf5','_skeletons.hdf5')
    if foo not in skeletons_files:
        print(f)
        
print('skeletons without MaskedVideos:')
for f in skeletons_files:
    foo = f.replace('Results','MaskedVideos')
    foo = foo.replace('_skeletons.hdf5','.hdf5')
    if foo not in masked_files:
        print(f)
        


#%%

def check_attrs(fname):
    fps_out, microns_per_pixel_out, is_light_background = read_unit_conversions(fname)
    if fps_out != (25.0, 25.0, 'seconds') or \
    microns_per_pixel_out != (10.0, 'micrometers'):
        print('Fix %s' % os.path.basename(fname))
    return

for i,fname in enumerate(masked_files):
    if i<900:
        continue
    if i%100==0:
        print(i)
    try:
        check_attrs(fname)
    except:
        print('Failed to check %s' % fname)

for i,fname in enumerate(skeletons_files):
    if i%100==0:
        print(i)
    try:
        check_attrs(fname)
    except:
        print('Failed to check %s' % fname)


#%%

def change_attrs(fname, field_name):
    print(os.path.basename(fname))
    read_unit_conversions(fname)
    with tables.File(fname, 'r+') as fid:
        group_to_save = fid.get_node(field_name)
        set_unit_conversions(group_to_save, 
                             expected_fps=expected_fps, 
                             microns_per_pixel=microns_per_pixel)
        
    read_unit_conversions(fname)


#for fname in masked_files:
#    change_attrs(fname, '/mask')
#for fname in skeletons_files:
#    change_attrs(fname, '/trajectories_data')