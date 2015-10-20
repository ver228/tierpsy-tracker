# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:24:56 2015

@author: ajaver
"""
import numpy as np
import cv2
import os
import tables
import tifffile

def writeFullFramesTiff(masked_image_file, tiff_file = '', reduce_fractor = 8, base_name = ''):
    '''
    write scale down the saved full frames, and put them into a tiff stack.
    requires either install skimage with the freeimage library plugin, or the tifffile module.
    '''
    #if no tiff_file is given, the save name will be derived from masked_image_file   
    if not tiff_file:
        tiff_file = os.path.splitext(masked_image_file)[0] + '_full.tiff';
    
    with tables.File(masked_image_file, "r") as mask_fid:
        full_data = mask_fid.get_node("/full_data")
        #determine the expected size of the full_data size with respect to the
        #number of frames in full_data and the same interval (I only do that 
        #to correct for a bug in previously saved files)
        expected_size = int(np.floor(mask_fid.get_node("/mask").shape[0]/full_data._f_getattr('save_interval')) + 1);
        if expected_size > full_data.shape[0]:
            expected_size = full_data.shape[0]
        
        #initialized reduced array  
        im_size = tuple(np.array(full_data.shape)[1:]//reduce_fractor)
        reduce_factor = (im_size[-1], im_size[-2])
        I_worms = np.zeros((expected_size, im_size[0],im_size[1]), dtype = np.uint8)
        
        print(base_name + ' Reading for data the tiff file...');
        
        for frame_number in range(expected_size):
            I_worms[frame_number, :,:] = cv2.resize(full_data[frame_number,:,:], reduce_factor);
    print(base_name + ' Writing tiff file...');

    #For some reason gzip compression appears as an inverted image in 
    #preview (both windows and mac), but it is read correctly in ImageJ
    tifffile.imsave(tiff_file, I_worms)#, compress=5, photometric = 'minisblack') 
    print(base_name + ' Tiff file done.');

if __name__ == '__main__':
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/Capture_Ch1_11052015_195105.hdf5'
    writeFullFramesTiff(masked_image_file)