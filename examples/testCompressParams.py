# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:55:55 2015

@author: ajaver
"""

import cv2
import numpy as np
import os

import sys
sys.path.append('..')
from MWTracker.compressVideos.compressVideo import getROIMask
from MWTracker.compressVideos.readVideoffmpeg import readVideoffmpeg

from skimage.viewer import ImageViewer
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.widgets import Slider

#file path
video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_25062015_190805.mjpg'

#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_29062015_083206.mjpg'
#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_28062015_202201.mjpg'

#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_27062015_220331.mjpg'
#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_27062015_180008.mjpg'

#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_07_01/15_07_01_earlyL1_Ch1_02072015_075924.mjpg'
#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_07_01/15_07_01_earlyL1_Ch1_02072015_055743.mjpg'

#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_30/15_06_30_L1_Ch1_30062015_180744.mjpg'

#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_07_03/15_07_03_midL3_Ch1_03072015_113312.mjpg'

#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_07_01/15_07_01_earlyL1_Ch1_01072015_113354.mjpg'


#video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_07_03/15_07_03_midL3_Ch1_03072015_144228.mjpg'



assert os.path.exists(video_file)

mask_param = {'min_area': 100, 'max_area': 5000, 'has_timestamp': True, 
'thresh_block_size':61, 'thresh_C':15}

mask_param = {'min_area': 100, 'max_area': 10000, 'has_timestamp': True, 
'thresh_block_size':61, 'thresh_C':55}

#mask_param = {'min_area': 100, 'max_area': 10000, 'has_timestamp': True, 
#'thresh_block_size':85, 'thresh_C':15}


def maskedImage(image, thresh_C, min_area, max_area, thresh_block_size):
    mask = getROIMask(image,  min_area=min_area, max_area=max_area, thresh_block_size=thresh_block_size, thresh_C=thresh_C, has_timestamp=False)
    return image*mask



masked_plugin = Plugin(image_filter=maskedImage)
masked_plugin += Slider('min_area', 0, 1000, value = mask_param['min_area'], value_type = 'int', update_on='release')
masked_plugin += Slider('max_area', 500, 10000, value = mask_param['max_area'], value_type = 'int', update_on='release')
masked_plugin += Slider('thresh_C', -100, 100, value = mask_param['thresh_C'], value_type = 'int', update_on='release')
masked_plugin += Slider('thresh_block_size', 0, 200, value = mask_param['thresh_block_size'], value_type = 'int', update_on='release')



vid = cv2.VideoCapture(video_file);

#im_width= vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#im_height= vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

#vid = readVideoffmpeg(video_file)


ret, image = vid.read()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#Ibuff = np.zeros((buffer_size, im_height, im_width), dtype = np.uint8)
#for ii in range(buffer_size):    
#    ret, image = vid.read() #get video frame, stop program when no frame is retrive (end of file)
#    if ret == 0:
#        break
#    Ibuff[ii] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

import matplotlib.pylab as plt

#plt.figure()
plt.imshow(image, cmap='gray', interpolation='none')

#plt.figure()
viewer = ImageViewer(image);
viewer += masked_plugin
viewer.show()
