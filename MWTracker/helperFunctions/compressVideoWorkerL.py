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
from ..compressVideos.getAdditionalData import storeAdditionalDataSW
#from ..compressVideos.writeDownsampledVideo import writeDownsampledVideo
#from ..compressVideos.writeFullFramesTiff import writeFullFramesTiff

from ..helperFunctions.tracker_param import tracker_param

def print_flush(fstr):
    print(fstr)
    sys.stdout.flush()

def compressVideoWorkerL(video_file, mask_dir, param_file = '', is_single_worm = False): 
    #get function parameters
    param = tracker_param(param_file)
    
    #check if the video file exists
    assert os.path.exists(video_file)
    
    if mask_dir[-1] != os.sep:
        mask_dir += os.sep
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = mask_dir + base_name + '.hdf5'
    
    print(masked_image_file)
    try:
        with h5py.File(masked_image_file, "r") as mask_fid:
            has_finished = mask_fid['/mask'].attrs['has_finished']
    except:
        has_finished = 0 

    if has_finished == 2:
        print_flush('File alread exists: %s. If you want to calculate the mask again delete the existing file.' % masked_image_file)
        return
    
    initial_time = time.time();
    
    if has_finished < 1:    
        compressVideo(video_file, masked_image_file, **param.compress_vid_param)
    
    if has_finished < 2:
        #store the additional information for the case of single worm
        if is_single_worm: 
            storeAdditionalDataSW(video_file, masked_image_file)

    time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
    progress_str = 'Processing Done. Total time = %s' % time_str
    print_flush(base_name + ' ' + progress_str)
    
    return masked_image_file
    
