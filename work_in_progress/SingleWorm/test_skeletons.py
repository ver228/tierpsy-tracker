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

from MWTracker.trackWorms.getSkeletonsTables import getWormMask, binaryMask2Contour
from MWTracker.trackWorms.segWormPython.cleanWorm import cleanWorm

#file_mask = '/Users/ajaver/Desktop/Videos/03-03-11/MaskedVideos/03-03-11/N2 swimming_2011_03_03__16_36___3___10.hdf5'
#file_mask = '/Volumes/behavgenom_archive$/MaskedVideos/nas207-3/Data/from pc207-15/laura/09-07-10/3/egl-17 (e1313)X on food R_2010_07_09__11_43_13___2___4.hdf5'
#file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_1/MaskedVideos/431 JU298 on food L_2011_03_17__12_02_58___2___3.hdf5'
file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/798 JU258 on food L_2011_03_22__16_26_58___1___12.hdf5'

file_skel = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
file_traj = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_trajectories.hdf5')
assert(os.path.exists(file_mask))
assert(os.path.exists(file_traj))
assert(os.path.exists(file_skel))


with pd.HDFStore(file_skel, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

#with pd.HDFStore(file_traj, 'r') as fid:
#    plate_worms = fid['/plate_worms']

current_frame = 261
with h5py.File(file_mask, 'r') as fid:
    worm_img = fid['/mask'][current_frame]

row_data = trajectories_data[trajectories_data['frame_number'] == current_frame]
row_data = row_data.iloc[0]

worm_mask = getWormMask(worm_img, row_data['threshold'])

min_mask_area = row_data['area']/2
worm_cnt, _ = binaryMask2Contour(worm_mask, roi_center_x = row_data['coord_y'], roi_center_y = row_data['coord_x'], min_mask_area = min_mask_area)


worm_cnt_n = cleanWorm(worm_cnt, 49)
worm_mask_n = np.zeros_like(worm_mask)
cv2.drawContours(worm_mask_n, [worm_cnt_n.astype(np.int32)], 0, 1, -1)


plt.figure()
plt.imshow(worm_mask, interpolation = 'none', cmap = 'gray')
plt.grid('off')

plt.figure()
plt.imshow(worm_mask_n, interpolation = 'none', cmap = 'gray')
plt.grid('off')


