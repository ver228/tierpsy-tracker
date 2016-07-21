# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 20:31:40 2016

@author: ajaver
"""

import cv2
import matplotlib.pylab as plt
from scipy.ndimage.filters import median_filter

img_name = '/Users/ajaver/Desktop/Videos/tiffs/recording 8.1_X1.tif'

thresh_block_size = 61
thresh_C = -15

img = cv2.imread(img_name, 0)
img = median_filter(img, 5)

mask = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh_block_size, thresh_C)
            
plt.imshow(mask, interpolation='none', cmap='gray')
