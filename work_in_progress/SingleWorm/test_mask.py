# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:27:28 2015

@author: ajaver
"""
import cv2
import os
import sys
import matplotlib.pylab as plt
import numpy as np
import tables

sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
from MWTracker.trackWorms.getSkeletonsTables import getSmoothTrajectories
from MWTracker.compressVideos.compressVideo import getROIMask

#videos_file = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Videos/135 CB4852 on food L_2011_03_09__15_51_36___1___8.avi'
videos_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Videos/npr-2 (ok419)IV on food R_2010_01_25__15_29_03___4___10.avi'

assert os.path.exists(videos_file)

full_file = videos_file[:-4] + '_full.hdf5'

if not os.path.exists(full_file):
    vid = cv2.VideoCapture(videos_file)
    images = np.zeros((26999, 480, 640), np.uint8)
    tot = 0
    while 1:
        ret, img = vid.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
        images[tot] = img
        tot += 1
        print(tot)
    
    with tables.File(full_file, 'w') as full_fid:
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        full_fid.create_carray('/', 'full', obj = images, filters=table_filters)

#%%


DEFAULT_MASK_PARAM = {'min_area':100, 'max_area':5000, 
'has_timestamp':False, 'thresh_block_size':61, 'thresh_C':15, 
'dilation_size': 1}

ind = 7261
with tables.File(full_file, 'r') as full_fid:
    img = full_fid.get_node('/', 'full')[ind]
    
#plt.figure()
#plt.imshow(getROIMask(img, **DEFAULT_MASK_PARAM))


IM_LIMX = img.shape[0]-2
IM_LIMY = img.shape[1]-2
min_area = DEFAULT_MASK_PARAM['min_area']
max_area = DEFAULT_MASK_PARAM['max_area']
thresh_block_size = DEFAULT_MASK_PARAM['thresh_block_size']

N = 15
area_bw = np.zeros(N+1) 

for thresh_C in range(N, N+1):
    mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh_block_size, thresh_C)    
    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    goodIndex = []
    for ii, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if not np.any(contour ==1) and not np.any(contour[:,:,0] ==  IM_LIMY)\
            and not np.any(contour[:,:,1] == IM_LIMX) and hierarchy[0][ii][2] == -1:
    
                goodIndex.append(ii)
    
    mask = np.zeros(img.shape, dtype=img.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 1, cv2.FILLED)
    #DEFAULT_MASK_PARAM['thresh_C'] = ii
    #dd = getROIMask(img, **DEFAULT_MASK_PARAM)
    area_bw[thresh_C] = np.sum(mask)
    
plt.figure()
plt.imshow(mask)
