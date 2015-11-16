# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:55:55 2015

@author: ajaver
"""

import cv2
import matplotlib.pylab as plt
import numpy as np
import os
import functools
import json

import sys
sys.path.append('..')
from MWTracker.compressVideos.compressVideo import getROIMask


#from MWTracker.compressVideos.readVideoffmpeg import readVideoffmpeg

from skimage.viewer import ImageViewer
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.widgets import Slider



#file path
#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_25062015_190805.mjpg'
video_file = '/Users/ajaver/Desktop/Pratheeban_videos/RawData/15_07_03_2hrL1_Ch1_03072015_162628.mjpg'
assert os.path.exists(video_file)


mask_param = {'min_area': 100, 'max_area': 5000,  'thresh_block_size':61, 'thresh_C':15}
buffer_size = 25

vid = cv2.VideoCapture(video_file);

im_width= vid.get(cv2.CAP_PROP_FRAME_WIDTH)
im_height= vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


Ibuff = np.zeros((buffer_size, im_height, im_width), dtype = np.uint8)
for ii in range(buffer_size):    
    ret, image = vid.read() #get video frame, stop program when no frame is retrive (end of file)
    if ret == 0:
        break
    Ibuff[ii] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
Imin = np.min(Ibuff, axis=0)

#ret, image = vid.read()
#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def maskedImage(image, thresh_C, min_area, max_area, thresh_block_size):
    mask_param['min_area'] = min_area
    mask_param['max_area'] = max_area
    mask_param['thresh_block_size'] = thresh_block_size
    mask_param['thresh_C'] = thresh_C

    mask = getROIMask(image,  min_area=min_area, max_area=max_area, thresh_block_size=thresh_block_size, thresh_C=thresh_C, has_timestamp=False)
    return image*mask

plt.figure()
plt.imshow(Imin, cmap='gray', interpolation='none')

masked_plugin = Plugin(image_filter=maskedImage)
masked_plugin += Slider('min_area', 0, 1000, value = mask_param['min_area'], value_type = 'int', update_on='release')
masked_plugin += Slider('max_area', 500, 10000, value = mask_param['max_area'], value_type = 'int', update_on='release')
masked_plugin += Slider('thresh_C', -100, 100, value = mask_param['thresh_C'], value_type = 'int', update_on='release')
masked_plugin += Slider('thresh_block_size', 0, 200, value = mask_param['thresh_block_size'], value_type = 'int', update_on='release')

viewer = ImageViewer(Imin);
viewer += masked_plugin
viewer.show()

json_file = video_file.rpartition('.')[0] + '.json'

with open(json_file, 'w') as fid:
    json.dump(mask_param, fid)
