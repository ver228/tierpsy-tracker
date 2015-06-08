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
from movement_validation import WormFeatures

# Set up the necessary file paths for file loading
#----------------------
base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)

matlab_generated_file_path = os.path.join(
    base_path,'example_video_feature_file.mat')

data_file_path = os.path.join(base_path,"example_video_norm_worm.mat")

# Load the normalized worm from file
nw = NormalizedWorm.from_schafer_file_factory(data_file_path)
#load and calculate data for comparison
matlab_worm_features = WormFeatures.from_disk(matlab_generated_file_path)

#Since the first and the last points of vulva and non_vulva contour are equal
contour = np.concatenate((nw.vulva_contour, nw.non_vulva_contour[-1::-1,:,:])) 

#opencv does not like float64, this actually make sense for image data where 
#we do not require a large precition in the decimal part. This could save quite a lot of space
contour = contour.astype(np.float32)
tot = contour.shape[-1]

#--------------------
#let's calculate the eccentricity from the contour moments
eccentricity = np.full(tot, np.nan);
orientation = np.full(tot, np.nan);
for ii in range(tot):
    worm_cnt = contour[:,:,ii];
    if ~np.any(np.isnan(worm_cnt)):
        moments  = cv2.moments(worm_cnt)
        
        a1 = (moments['mu20']+moments['mu02'])/2
        a2 = np.sqrt(4*moments['mu11']**2+(moments['mu20']-moments['mu02'])**2)/2
        
        minor_axis = a1-a2
        major_axis = a1+a2
        
        eccentricity[ii] = np.sqrt(1-minor_axis/major_axis)
        orientation[ii] = (180 / np.pi)*np.arctan2(2*moments['mu11'], (moments['mu20']-moments['mu02']))/2
#%%
#--------------------
#now let's calculate the eccentricity by creating a binary image with a worm image
eccentricity_s = np.full(tot, np.nan);
orientation_s = np.full(tot, np.nan);
for ii in range(tot):
    worm_cnt = contour[:,:,ii];
    
    #stop if the contour is not valid
    if np.any(np.isnan(worm_cnt)):
        continue
    
    #the minimum value of the contour will be the image upper-left corner
    corner = np.min(worm_cnt, axis=0)
    worm_cnt = worm_cnt-corner
    
    #after subracting the corner the maximum contour values will be the other corner
    im_size = np.max(worm_cnt, axis=0)[::-1] #the coordinate systems used by opencv is different from the numpy's one
    
    im_dum = np.zeros(im_size);
    
    #draw contours
    cv2.drawContours(im_dum, [worm_cnt.astype(np.int32)], 0, 1, -1);
    
    #get contour points and substract the center of mass
    y,x = np.where(im_dum)
    x = x-np.mean(x)
    y = y-np.mean(y)
    
    #same code as in h__calculateSingleValues in postures_features.py
    N = float(len(x))
    # Calculate normalized second central moments for the region.
    uxx = np.sum(x * x) / N
    uyy = np.sum(y * y) / N
    uxy = np.sum(x * y) / N
    
    # Calculate major axis length, minor axis length, and eccentricity.
    common = np.sqrt((uxx - uyy) ** 2 + 4 * (uxy ** 2))
    majorAxisLength = 2 * np.sqrt(2) * np.sqrt(uxx + uyy + common)
    minorAxisLength = 2 * np.sqrt(2) * np.sqrt(uxx + uyy - common)
    eccentricity_s[ii] = 2 * np.sqrt((majorAxisLength / 2) ** 2 - (minorAxisLength / 2) ** 2) / majorAxisLength
    
    # Calculate orientation.
    if (uyy > uxx):
        num = uyy - uxx + np.sqrt((uyy - uxx) ** 2 + 4 * uxy ** 2)
        den = 2 * uxy
    else:
        num = 2 * uxy
        den = uxx - uyy + np.sqrt((uxx - uyy) ** 2 + 4 * uxy ** 2)
    
    orientation_s[ii] = (180 / np.pi) * np.arctan(num / den) 

        
#%%
plt.figure()
plt.plot(eccentricity, eccentricity_s, '.')
plt.xlabel('Eccentricity (contour opencv moments)')
plt.ylabel('Eccentricity (filled contour)')
plt.savefig('Eccentricity contour moment vs full image.png')
#%%
import matplotlib.pylab as plt
plt.figure()
plt.plot(matlab_worm_features.posture.eccentricity, eccentricity, '.')
plt.xlabel('Eccentricity (MATLAB file)')
plt.ylabel('Eccentricity (contour opencv moments)')
plt.savefig('Eccentricity contour moment vs MATLAB method.png')