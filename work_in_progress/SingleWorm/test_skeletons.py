# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:57:30 2015

@author: ajaver
"""
import matplotlib.pylab as plt
import h5py
import pandas as pd
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from scipy.signal import medfilt
import time
import os


file_mask = ''

file_mask = '/Users/ajaver/Desktop/Videos/03-03-11/MaskedVideos/03-03-11/N2 swimming_2011_03_03__16_36___3___10.hdf5'
file_skel = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
file_traj = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_trajectories.hdf5')
assert(os.path.exists(file_mask))
assert(os.path.exists(file_traj))
assert(os.path.exists(file_skel))


#file_mask = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/MaskedVideos/135 CB4852 on food L_2011_03_09__15_51_36___1___8.hdf5'
#file_skel = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/135 CB4852 on food L_2011_03_09__15_51_36___1___8_skeletons.hdf5'
#file_traj = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/135 CB4852 on food L_2011_03_09__15_51_36___1___8_trajectories.hdf5'

with pd.HDFStore(file_skel, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

with pd.HDFStore(file_traj, 'r') as fid:
    plate_worms = fid['/plate_worms']

current_frame = 5000
with h5py.File(file_mask, 'r') as fid:
    worm_img = fid['/mask'][current_frame]

current_row = trajectories_data[trajectories_data['frame_number'] == current_frame]
current_row_p = plate_worms[plate_worms['frame_number'] == current_frame]

#win_size = 

def plot_hist(worm_img):
    pix_valid = worm_img[worm_img!=0]
    
    otsu_thresh = threshold_otsu(pix_valid)
    
    pix_hist = np.bincount(pix_valid)  
    #pix_hist = medfilt(pix_hist, 5)
    pix_hist = np.convolve(pix_hist, np.ones(5), 'same')    
    
    
    
    xx = np.arange(otsu_thresh, pix_hist.size)
    
    cumhist = np.cumsum(pix_hist)
    hist_ratio = pix_hist[xx]/cumhist[xx]
    
    thresh = np.where(pix_hist[xx]/cumhist[xx]>0.020)[0][0] + otsu_thresh
    thresh2 = np.where((hist_ratio[3:]-hist_ratio[:-3])>0.02)[0][0] + otsu_thresh
    
    plt.plot(hist_ratio[3:]-hist_ratio[:-3])
    
    print(otsu_thresh, thresh, thresh2)
    return thresh2

threshold = current_row['threshold'].values


plt.figure()
thresh2 = plot_hist(worm_img)

N = 5
#make the worm more uniform. This is important to get smoother contours.
#worm_img = cv2.medianBlur(worm_img, N);

plot_hist(worm_img)

#smooth mask by morphological closing
#worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
#%%
worm_mask = ((worm_img < thresh2) & (worm_img!=0)).astype(np.uint8)

plt.figure()
plt.imshow(worm_mask, interpolation = 'none', cmap = 'gray')
#%%
plt.figure()
plt.imshow(worm_img, interpolation = 'none', cmap = 'gray')

#%%
#plt.figure()
#plt.plot(trajectories_data['threshold'])
#%%
#tic = time.time()
#for kk in range(1000):
#    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#    cv2.morphologyEx(worm_img, cv2.MORPH_CLOSE, strel)
#print('erosion', time.time()-tic)
#
#tic = time.time()
#for kk in range(1000):
#    cv2.medianBlur(worm_img, cv2.MORPH_CLOSE, 3)
#print('erosion', time.time()-tic)
