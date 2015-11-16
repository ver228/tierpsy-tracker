# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:39:56 2015

@author: ajaver
"""


import h5py
import numpy as np
import matplotlib.pylab as plt

filename = '/Users/ajaver/Google Drive/MWTracker_Example/Results/Capture_Ch1_18062015_140908_intensities.hdf5'


worm_index = 1
with h5py.File(filename, 'r') as fid:
    worm_indexes = fid['/worm_index'][:]
    valid = np.where(worm_indexes == worm_index)[0]
    #should be order by worm_index block
    first = np.min(valid);
    last = np.max(valid);
    
    frame_numbers = fid['/frame_number'][first:last+1]
    worm_int = fid['/straighten_worm_intensity'][first:last+1]

#data is saved in float16 to save space, but further analysis require to use higher precision to avoid overflow
worm_int = worm_int.astype(np.float64)


plt.figure()
plt.imshow(worm_int[0], interpolation='none', cmap='gray')
    
plt.figure()
plt.imshow(np.nanmean(worm_int[0:100], axis=0), interpolation='none', cmap='gray')

