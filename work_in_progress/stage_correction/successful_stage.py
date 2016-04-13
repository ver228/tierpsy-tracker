# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:02:51 2016

@author: ajaver
"""
import glob
import os
import h5py
import numpy as np
import matplotlib.pylab as plt

#main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_1/MaskedVideos/'
#files = glob.glob(os.path.join(main_dir, '*.hdf5' ))
#files = sorted(files)

mask_list_files = ['/Users/ajaver/Documents/GitHub/Multiworm_Tracking/single_worm_db/masks_agar_1', \
                   '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/single_worm_db/masks_agar_2']

files = []
for file_list in mask_list_files:
    with open(file_list, 'r') as fid:
        these_files = fid.read().split('\n')
        these_files = [x for x in these_files if x]
        files += these_files


files_results = {}
for mask_id, masked_image_file in enumerate(files):
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    feat_file = masked_image_file.replace('MaskedVideos', 'Features')[:-5] + '_features.mat'
    
    if mask_id % 100 == 0:
        print(mask_id, len(files))

    try:
        with h5py.File(skeletons_file, 'r') as fid:
            has_finished = fid['/stage_movement'].attrs['has_finished'][0]
    except (KeyError, OSError):
        continue
    
    if not has_finished in files_results:
        files_results[has_finished] = []

    files_results[has_finished].append(masked_image_file)
#%%
for x in sorted(files_results):
    print('FLAG: %i -> N = %i' % (x, len(files_results[x])))

#%%
with open('failed_files.txt', 'w') as fid:
    for file in files_results[82]:
        fid.write(file + '\n')
