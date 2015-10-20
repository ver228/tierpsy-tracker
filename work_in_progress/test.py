# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:42:02 2015

@author: ajaver
"""
import h5py
import numpy as np
import time

fid = h5py.File('/Volumes/behavgenom$/GeckoVideo/MaskedVideos/DiffMedia_20150619/Capture_Ch5_19062015_193839.hdf5' , 'r')

valid_groups = []
def geth5name(name, dat):
    dat = fid[name]
    if isinstance(dat, h5py.Dataset) and len(dat.shape) == 3 and dat.dtype == np.uint8:
        valid_groups.append('/' + name)

tic = time.time()
fid.visititems(geth5name)
print(valid_groups)
print(time.time() - tic)