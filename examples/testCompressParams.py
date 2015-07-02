# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:55:55 2015

@author: ajaver
"""

import cv2
import numpy as np

import sys
sys.path.append('..')
from MWTracker.compressVideo.compressVideo import getROIMask

from skimage.viewer import ImageViewer
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.widgets import Slider

#file path
video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/L1_early/15_06_18/15_06_18_video3.mjpg'
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

mask_param = {'min_area': 100, 'max_area': 5000, 'has_timestamp': True, 
'thresh_block_size':61, 'thresh_C':70}

def maskedImage(image, thresh_C, min_area, max_area, thresh_block_size):
    mask = getROIMask(image,  min_area=min_area, max_area=max_area, thresh_block_size=thresh_block_size, thresh_C=thresh_C, has_timestamp=False)
    return image*mask


masked_plugin = Plugin(image_filter=maskedImage)
masked_plugin += Slider('min_area', 0, 1000, value = 100, value_type = 'int', update_on='release')
masked_plugin += Slider('max_area', 500, 10000, value=5000, value_type = 'int', update_on='release')
masked_plugin += Slider('thresh_C', -100, 100, value=15, value_type = 'int', update_on='release')
masked_plugin += Slider('thresh_block_size', 0, 200, value=61, value_type = 'int', update_on='release')


viewer = ImageViewer(Ibuff[0]);
viewer += masked_plugin
viewer.show()
