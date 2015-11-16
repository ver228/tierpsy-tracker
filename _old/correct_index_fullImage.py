# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""
import h5py
import matplotlib.pylab as plt
import numpy as np
from math import floor
import time

#maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch1_16022015_174636.hdf5';
maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch4_18022015_230213.hdf5';


mask_fid = h5py.File(maskFile, 'a');

full_dataset = mask_fid["/full_data"]

expected_size = floor(mask_fid["/mask"].shape[0]/float(full_dataset.attrs['save_interval']) + 1);

if expected_size<full_dataset.shape[0]:
    #much faster than #full_dataset[0::full_dataset.attrs['save_interval'],:,:]
    
    N = int(floor(full_dataset.shape[0]/full_dataset.attrs['save_interval']))+1
    d1 = np.zeros((N,full_dataset.shape[1],full_dataset.shape[2]),dtype = full_dataset.dtype);
    for ii, kk in enumerate(range(0, full_dataset.shape[0], full_dataset.attrs['save_interval'])):
        print ii, kk
        full_dataset[ii,:,:] = full_dataset[kk,:,:];

    full_dataset.resize(expected_size, axis=0); 
    print full_dataset.shape
else:
    print full_dataset.shape