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

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_goa/MaskedVideos/'
files = glob.glob(os.path.join(main_dir, '*.hdf5' ))
files = sorted(files)


good_files = []
bad_files = []
no_stage_vec = []
for mask_id, masked_image_file in enumerate(files):
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    feat_file = masked_image_file.replace('MaskedVideos', 'Features')[:-5] + '_features.mat'

    with h5py.File(skeletons_file, 'r') as fid:
        if not '/stage_movement/stage_vec' in fid:
            no_stage_vec.append(masked_image_file)
            continue
        
        stage_vec = fid['/stage_movement/stage_vec'][:]
        if np.all(np.isnan(stage_vec)):
            bad_files.append(masked_image_file)
            plt.figure()
            plt.plot(fid['/stage_movement/frame_diffs'], 'r')
        else:
            good_files.append(masked_image_file)
            
            plt.figure()
            plt.plot(fid['/stage_movement/frame_diffs'], 'b')


print('GOOD', len(good_files))
print('BAD', len(bad_files))