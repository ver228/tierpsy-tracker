# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:43:32 2015

@author: ajaver
"""
import h5py
import cv2
import numpy as np
from math import floor
from skimage.io._plugins import freeimage_plugin as fi

def save_full_frames(mask_fid, tiff_file, reduce_fractor = 8):
    expected_size = int(floor(mask_fid["/mask"].shape[0]/float(mask_fid["/full_data"].attrs['save_interval']) + 1));

    im_size = tuple(np.array(mask_fid["/full_data"].shape[1:])/reduce_fractor)
    
    I_worms = np.zeros((expected_size, im_size[0],im_size[1]), dtype = np.uint8)
    
    if np.all(mask_fid["/full_data"][1,:,:]==0):
        save_interval = mask_fid["/full_data"].attrs['save_interval']
        for frame in range(expected_size):
            
            I_worms[frame, :,:] = cv2.resize(mask_fid["/full_data"][frame*save_interval,:,:], im_size);
        
    else:
        for frame in range(expected_size):
            I_worms[frame, :,:] = cv2.resize(mask_fid["/full_data"][frame,:,:], im_size);
        
        
    fi.write_multipage(I_worms, tiff_file, fi.IO_FLAGS.TIFF_LZW)


#masked_image_file = rootDir + 'Capture_Ch3_26022015_161436.hdf5'
#masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150218/CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#root_dir = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150218/';
#base_file = 'CaptureTest_90pc_Ch%i_18022015_230213'

#root_dir = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150216/';
#base_file = 'CaptureTest_90pc_Ch%i_16022015_174636'
root_dir = '/Volumes/ajaver$/GeckoVideo/Compressed/'
base_file = 'CaptureTest_90pc_Ch1%i_20022015_183607'

for ch_ind in range(1,2):
    channel_file = (base_file % ch_ind)
    masked_image_file = root_dir + channel_file + '.hdf5'
    tiff_file = root_dir + channel_file + '_full.tiff'
    
    mask_fid = h5py.File(masked_image_file, "r");
    
    save_full_frames(mask_fid, tiff_file, reduce_fractor = 8)
