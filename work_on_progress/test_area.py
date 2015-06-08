# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:11:59 2015

@author: ajaver
"""
import cv2
import numpy as np
import matplotlib.pylab as plt

import sys
import os

sys.path.append('../../movement_validation')
from movement_validation import user_config, NormalizedWorm

# Set up the necessary file paths for file loading
#----------------------
base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

matlab_generated_file_path = os.path.join(
    base_path,'example_video_feature_file.mat')

data_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

# OPENWORM
#----------------------
# Load the normalized worm from file
nw = NormalizedWorm.from_schafer_file_factory(data_file_path)

#Since the first and the last points of vulva and non_vulva contour are equal
#This add two redundant points. The last one is useful to shift the data (close the contour)
#in using the shoelace formula. The one in the middle will not be included in the 
#calculation since it correspond to two equal consecutive points, whos contribution
#will be cancel out. z = x(n-1)*y(n) - x(n)*y(n-1) if x(n) = x(n-1) and y(n) = y(n-1), z = 0
contour = np.concatenate((nw.vulva_contour, nw.non_vulva_contour[::-1,:,:])) 

#this area is signed. It is positive if the contour is going clockwise, and negative if it is counter-clockwise
signed_area = np.sum(contour[:-1,0,:]*contour[1:,1,:]-\
contour[1:,0,:]*contour[:-1,1,:], axis=0)/2

#get the absolute value and obtain the real area
area_shoelace = np.abs(signed_area)


#now let's calculate the area by creating a binary image with a worm image
tot_cnt = nw.vulva_contour.shape[2]
area_filled = np.full(tot_cnt, np.nan);
for ii in range(tot_cnt):
    worm_cnt = contour[:,:,ii];
    
    #stop if the contour is not valid
    if np.any(np.isnan(worm_cnt)):
        continue
    
    #the minimum value of the contour will be the image upper-left corner
    corner = np.min(worm_cnt, axis=0)
    worm_cnt = worm_cnt-corner
    
    #after subracting the corner the maximum contour values will be the other corner
    im_size = np.max(worm_cnt, axis=0)[::-1] #the coordinate systems used by opencv is different from the numpy's one
    
    #TODO: it might be necessary to rescale the contour so the sampling is the same, in all cases
    
    im_dum = np.zeros(im_size)
    
    #draw contours
    cv2.drawContours(im_dum, [worm_cnt.astype(np.int32)], 0, 1, -1)
    area_filled[ii] = np.sum(im_dum)
    
    #example plot
    if ii == 4624:
        plt.figure()
        plt.imshow(im_dum)
        plt.plot(worm_cnt[:,0], worm_cnt[:,1], 'g')
        plt.savefig('Filled_area_example.png')

plt.figure()
plt.plot(area_shoelace, label ='Shoelace method')
plt.plot(area_filled, label = 'Drawing the polygon')
plt.legend()
plt.ylabel('Area')
plt.ylabel('Frame Number')
plt.savefig('Shoelace_vs_Filled_area.png')

#%%
plt.plot(area_shoelace, area_filled, '.')