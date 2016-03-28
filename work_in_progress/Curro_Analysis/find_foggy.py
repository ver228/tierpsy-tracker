# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:23:09 2016

@author: ajaver
"""

import glob
import os
import h5py
import matplotlib.pylab as plt
import numpy as np

def imshow(img):
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.grid('off')

main_dir = '/Volumes/behavgenom$/GeckoVideo/Curro/MaskedVideos/exp1'
file_names = [ff for ff in os.listdir(main_dir) if ff.endswith('hdf5')]
file_names = sorted(file_names)



vid_avg_int = {}
for ii, fname in enumerate(file_names[120:],):
    print(fname)
    
    full_name = os.path.join(main_dir, fname);    
    with h5py.File(full_name) as fid:
        #full_frames = fid['/full_data'][:]
        first_full = fid['/full_data'][0]
        last_full = fid['/full_data'][-1]
        
    
    #vid_avg_int[fname] = np.percentile(full_frames, [10, 25, 50, 75, 90], axis=(1,2))
    
    plt.figure(num=None, figsize=(16, 12), dpi=80)
    plt.subplot(1,2,1)
    plt.title([ii, fname])
    imshow(first_full)
    #imshow(np.min(full_frames,axis=0))
    plt.subplot(1,2,2)
        
    #imshow(np.max(full_frames,axis=0))
    imshow(last_full)
    