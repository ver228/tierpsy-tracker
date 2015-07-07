# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:59:25 2015

@author: ajaver
"""

import time
import datetime
import os

from ..compressVideos.compressVideo import compressVideo
from ..compressVideos.writeDownsampledVideo import writeDownsampledVideo
from ..compressVideos.writeFullFramesTiff import writeFullFramesTiff

from ..helperFunctions.tracker_param import tracker_param

def compressVideoWorker(video_file, mask_files_dir, param_file = ''):
    
    #get function parameters
    param = tracker_param(param_file)
    
    #check if the video file exists
    assert os.path.exists(video_file)
    
    if mask_files_dir[-1] != os.sep:
        mask_files_dir += os.sep
    if not os.path.exists(mask_files_dir):
        os.makedirs(mask_files_dir)
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = mask_files_dir + base_name + '.hdf5'
    
    #tiff_file_name: used as flag to check if a previous run of the program finished succesfully
    tiff_file_name =  mask_files_dir + base_name + '_full.tiff' 
    
    if not os.path.exists(tiff_file_name):
        initial_time = time.time();
        compressVideo(video_file, masked_image_file, **param.compress_vid_param)
        writeDownsampledVideo(masked_image_file, base_name = base_name);
        writeFullFramesTiff(masked_image_file, base_name = base_name);
            
        time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
        progress_str = 'Processing Done. Total time = %s' % time_str
        print(base_name + ' ' + progress_str)
    else:
        print('File alread exists: %s' % masked_image_file)
        print('If you want to calculate the mask again delete the existing file.')
    
    return masked_image_file
    
