# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:59:25 2015

@author: ajaver
"""

import time
import datetime
import os
import sys
import h5py
from ..compressVideos.compressVideo import compressVideo
from ..compressVideos.getAdditionalData import storeAdditionalDataSW, getAdditionalFiles

from ..helperFunctions.tracker_param import tracker_param

def print_flush(fstr):
    print(fstr)
    sys.stdout.flush()

def compressVideoWorkerL(video_file, mask_dir, param_file = '', is_single_worm = False): 
    
    if mask_dir[-1] != os.sep: mask_dir += os.sep

    #get function parameters
    param = tracker_param(param_file)
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = os.path.join(mask_dir, base_name + '.hdf5')
    
    print(masked_image_file)
    try:
        #try to get the has_finished flag, if this fails the file is likely to be corrupted so we can start over again
        with h5py.File(masked_image_file, "r") as mask_fid:
            has_finished = mask_fid['/mask'].attrs['has_finished']
    except:
        has_finished = 0 

    if has_finished:
        print_flush('File alread exists: %s. If you want to calculate the mask again delete the existing file.' % masked_image_file)
        return
    else:
        initial_time = time.time();

        #check if the video file exists
        assert os.path.exists(video_file)
        
        #This function will throw an exception if the additional files do not exists
        if is_single_worm: getAdditionalFiles(video_file)
        
        #create the mask path if it does not exist
        if not os.path.exists(mask_dir): os.makedirs(mask_dir)
        
        compressVideo(video_file, masked_image_file, **param.compress_vid_param)
        
        #get the store the additional data for the single worm case. It needs that the 
        #masked_image_file has been created.
        if is_single_worm: storeAdditionalDataSW(video_file, masked_image_file)
    
    time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
    progress_str = 'Processing Done. Total time = %s' % time_str
    print_flush(base_name + ' ' + progress_str)
    
    return masked_image_file
    
