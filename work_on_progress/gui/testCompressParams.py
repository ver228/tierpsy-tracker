# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:55:55 2015

@author: ajaver
"""

import cv2
import numpy as np
import os
import functools

import sys
sys.path.append('..')
from MWTracker.compressVideos.compressVideo import getROIMask


#from MWTracker.compressVideos.readVideoffmpeg import readVideoffmpeg

#from skimage.viewer import ImageViewer
#from skimage.viewer.plugins.base import Plugin
#from skimage.viewer.widgets import Slider

#file path
video_file = '/Volumes/behavgenom$/Pratheeban/Worm_Videos/15_06_25/15_06_25_L1toadult_Ch1_25062015_190805.mjpg'
assert os.path.exists(video_file)

buffer_size = 25
mask_param = {'min_area': 100, 'max_area': 5000, 'has_timestamp': True, 
'thresh_block_size':61, 'thresh_C':15}

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

resize_factor = max(im_width,im_height)/512
resize_dim = (int(im_width//resize_factor), int(im_height//resize_factor))


def updateBar(value, key):
    mask_param[key] = value
    updateMask()

def updateMask():
    mask = getROIMask(Imin, **mask_param)
    mask = mask*Ibuff[0]
    mask = cv2.resize(mask, resize_dim)
    
    cv2.imshow('Mask', mask)

import ui_main
from PyQt4 import QtCore, QtGui
from PIL import Image


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    win_main = ui_main.QtGui.QWidget()
    uimain = ui_main.Ui_win_main()
    uimain.setupUi(win_main)
    
    
#cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

#cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
#cv2.createTrackbar('thresh_C', 'Mask', mask_param['thresh_C'], 100, functools.partial(updateBar, key='thresh_C'))
#
#
#
#im_ori = cv2.resize(Ibuff[0], resize_dim)
#
#cv2.imshow('Original', im_ori)


#
#k = cv2.waitKey(0)
#if k == ord('q') or k == 27:
#    break;

#cv2.destroyAllWindows()



#
#mask_param = {'min_area': 100, 'max_area': 10000, 'has_timestamp': True, 
#'thresh_block_size':61, 'thresh_C':55}

#mask_param = {'min_area': 100, 'max_area': 10000, 'has_timestamp': True, 
#'thresh_block_size':85, 'thresh_C':15}


#def maskedImage(image, thresh_C, min_area, max_area, thresh_block_size):
#    mask = getROIMask(image,  min_area=min_area, max_area=max_area, thresh_block_size=thresh_block_size, thresh_C=thresh_C, has_timestamp=False)
#    return image*mask



#masked_plugin = Plugin(image_filter=maskedImage)
#masked_plugin += Slider('min_area', 0, 1000, value = mask_param['min_area'], value_type = 'int', update_on='release')
#masked_plugin += Slider('max_area', 500, 10000, value = mask_param['max_area'], value_type = 'int', update_on='release')
#masked_plugin += Slider('thresh_C', -100, 100, value = mask_param['thresh_C'], value_type = 'int', update_on='release')
#masked_plugin += Slider('thresh_block_size', 0, 200, value = mask_param['thresh_block_size'], value_type = 'int', update_on='release')
#


#vid = cv2.VideoCapture(video_file);

#im_width= vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#im_height= vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

#vid = readVideoffmpeg(video_file)



#import matplotlib.pylab as plt
##%%
#
#fig = plt.figure()
#ax1 = plt.subplot(121)
#plt.imshow(image, cmap='gray', interpolation='none')
#
#ax2 = plt.subplot(121)
#
#viewer = ImageViewer(image);
#viewer += masked_plugin
#viewer.show()
