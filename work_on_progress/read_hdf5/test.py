# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:32:57 2015

@author: ajaver
"""

import tables
import matplotlib.pylab as plt
import numpy as np
import cv2

filename = '/Users/ajaver/Desktop/test/CSTCTest_Ch1_02102015_122155.hdf5'

#with tables.FILE(filename, 'r') as ff:
ff =  tables.File(filename, 'r')
masks = ff.get_node('/mask') 

full_img = masks[0,:,:]



curr_img = masks[100,:,:]
mask_bw = curr_img == 0
curr_img[mask_bw] = np.median(full_img)#full_img[mask_bw]

mask = cv2.adaptiveThreshold(curr_img.copy(), 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 15)

plt.imshow(mask)


