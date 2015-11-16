# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:43:32 2015

@author: ajaver
"""
import h5py

root_dir = '/Users/ajaver/Downloads/wetransfer-2af646/'
base_file = 'CaptureTest_90pc_Ch3_21022015_205929'

masked_image_file = root_dir + base_file + '.hdf5'
tiff_file = root_dir + base_file + '_mask_deflate.tiff'
    
mask_fid = h5py.File(masked_image_file, "r");
    
#from skimage.io._plugins import freeimage_plugin as fi
#fi.write_multipage(mask_fid["/mask"], tiff_file, fi.IO_FLAGS.TIFF_DEFLATE)

import tifffile
tifffile.imsave(tiff_file, mask_fid["/mask"], compress=4) 