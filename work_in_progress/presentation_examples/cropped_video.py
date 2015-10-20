# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:49:27 2015

@author: ajaver
"""

import cv2
import matplotlib.pylab as plt
import tables

import sys


sys.path.append('../../helperFunctions/')
sys.path.append('../../trackWorms/')
from getDrawTrajectories import writeVideoffmpeg

#base_name = 'Capture_Ch3_12052015_194303'
#mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
#results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    

base_name = 'Capture_Ch1_11052015_195105'
mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/'
results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'    

masked_image_file = mask_dir + base_name + '.hdf5'
trajectories_file = results_dir + base_name + '_trajectories.hdf5'
skeletons_file = results_dir + base_name + '_skeletons.hdf5'


with tables.File(masked_image_file, 'r')  as mask_fid:
    video_id = writeVideoffmpeg('cropped_video.avi',256,256);
    mask_group = mask_fid.get_node('/mask')
    for frame in range(mask_group.shape[0]):
        print(frame)
        Icrop = mask_group[frame,306:562, 706:962]*2
        video_id.write(Icrop)

    video_id.release()