# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:59:25 2015

@author: ajaver
"""

import time
import datetime
import os
import sys

sys.path.append('../compressVideos/')
from compressVideo import compressVideo
from writeDownsampledVideo import writeDownsampledVideo
from writeFullFramesTiff import writeFullFramesTiff

def getCompressVidWorker(video_file, mask_files_dir, 
                         FPS = 25, expected_frames = 15000, has_timestamp = True):
                             
    if mask_files_dir[-1] != os.sep:
        mask_files_dir += os.sep
    if not os.path.exists(mask_files_dir):
        os.makedirs(mask_files_dir)
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    masked_image_file = mask_files_dir + base_name + '.hdf5'
    if not os.path.exists(masked_image_file):
        initial_time = time.time();
        
        compressVideo(video_file, masked_image_file, buffer_size = FPS, \
        save_full_interval = 5000//FPS, base_name = base_name, useVideoCapture = False, 
        has_timestamp=True, expected_frames = 15000)
        
        writeDownsampledVideo(masked_image_file, base_name = base_name);
        writeFullFramesTiff(masked_image_file, base_name = base_name);
            
        time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
        progress_str = 'Processing Done. Total time = %s' % time_str;
        print(base_name + ' ' + progress_str)
    else:
        print('File alread exists: %s' % masked_image_file)
        print('If you want to calculate the mask again delete the existing file.')
    
    return masked_image_file
    
if __name__ == "__main__":
    video_file = sys.argv[1]
    mask_files_dir = sys.argv[2]

    getCompressVidWorker(video_file, mask_files_dir, has_timestamp = True, 
                         FPS = 25, expected_frames = 15000)