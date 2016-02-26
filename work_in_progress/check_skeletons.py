# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:10:20 2016

@author: ajaver
"""
import h5py
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.trackWorms.getSkeletonsTables import trajectories2Skeletons, getWormMask, getWormROI 


masked_image_file = '/Users/ajaver/Tmp/MaskedVideos/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/Andre/15-03-11/431 JU298 on food R_2011_03_15__16_17___3___11.hdf5'
skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
trajectories_file = skeletons_file.replace('_skeletons', '_trajectories')

#trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
#create_single_movies = False, resampling_N = 49, min_mask_area = 50, smoothed_traj_param = {})
#%%
#from MWTracker.helperFunctions.timeCounterStr import timeCounterStr
#from MWTracker.trackWorms.segWormPython.mainSegworm import getSkeleton
with pd.HDFStore(skeletons_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']
#%%

with h5py.File(masked_image_file, 'r') as fid:
    img = fid['/mask'][0]

row_data = trajectories_data.loc[0]

CMx = row_data['coord_x']
CMy = row_data['coord_y']
roi_size = row_data['roi_size']

roi_center = int(roi_size)//2
roi_range = np.array([-roi_center, roi_center])

#obtain bounding box from the trajectories
range_x = round(CMx) + roi_range
range_y = round(CMy) + roi_range
#%%
if range_x[0]<0: range_x[0] = 0#range_x -= 
if range_y[0]<0: range_y[0] = 0#range_y -= range_y[0]
#%%
if range_x[1]>img.shape[1]: range_x[1] = img.shape[1]#range_x += img.shape[1]-range_x[1]-1
if range_y[1]>img.shape[0]: range_y[1] = img.shape[0]#range_y += img.shape[0]-range_y[1]-1
#%%
worm_img = img[range_y[0]:range_y[1], range_x[0]:range_x[1]]

roi_corner = np.array([range_x[0]-1, range_y[0]-1])

#worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], row_data['roi_size'])
worm_mask = getWormMask(worm_img, row_data['threshold'])

plt.figure()
plt.imshow(worm_img, interpolation='none', cmap='gray')
plt.grid('off')