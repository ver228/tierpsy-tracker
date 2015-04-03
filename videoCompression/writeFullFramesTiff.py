# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:24:56 2015

@author: ajaver
"""
import numpy as np
import cv2
import os
import h5py

from parallelProcHelper import sendQueueOrPrint


def writeFullFramesTiff(masked_image_file, tiff_file = -1, reduce_fractor = 8, base_name = '', status_queue = ''):
    '''
    write scale down the saved full frames, and put them into a tiff stack.
    requires either install skimage with the freeimage library plugin, or the tifffile module.
    '''
    #if no tiff_file is given, the save name will be derived from masked_image_file   
    if tiff_file == -1:
        tiff_file = os.path.splitext(masked_image_file)[0] + '_full.tiff';
    
    
    mask_fid = h5py.File(masked_image_file, "r");

    #determine the expected size of the full_data size with respect to the 
    #number of frames in full_data and the same interval (I only do that 
    #to correct for a bug in previously saved files)
    expected_size = int(np.floor(mask_fid["/mask"].shape[0]/float(mask_fid["/full_data"].attrs['save_interval']) + 1));
    if expected_size > mask_fid["/full_data"].shape[0]: 
        expected_size = mask_fid["/full_data"].shape[0]
    
    #initialized reduced array  
    im_size = tuple(np.array(mask_fid["/full_data"].shape)[1:]/reduce_fractor)
    reduce_factor = (im_size[-1], im_size[-2])
    I_worms = np.zeros((expected_size, im_size[0],im_size[1]), dtype = np.uint8)
    
    sendQueueOrPrint(status_queue, 'Reading for data the tiff file...', base_name);
    
    for frame_number in range(expected_size):
        I_worms[frame_number, :,:] = cv2.resize(mask_fid["/full_data"][frame_number,:,:], reduce_factor);
    
    sendQueueOrPrint(status_queue, 'Writing tiff file...', base_name);
    
    try: 
        #Requires the installation of freeimage library. 
        #On mac is trivial using brew (brew install freeimage), 
        #but i am not sure how to do it on windows
        from skimage.io._plugins import freeimage_plugin as fi
        fi.write_multipage(I_worms, tiff_file, fi.IO_FLAGS.TIFF_LZW) #the best way I found to write lzw compression on python
    
    except:
        import tifffile #pip install tifffile
        
        #For some reason gzip compression appears as an inverted image in 
        #preview (both windows and mac), but it is read correctly in ImageJ
        tifffile.imsave(tiff_file, I_worms, compress=4) 
    
    sendQueueOrPrint(status_queue, 'Tiff file done.', base_name);