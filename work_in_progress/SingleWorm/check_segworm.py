# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:38:33 2016

@author: ajaver
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d
        
from scipy.signal import savgol_filter

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline

import time
import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')

from MWTracker.trackWorms.getSkeletonsTables import getWormMask, getSkeleton
from MWTracker.trackWorms.segWormPython.mainSegworm import binaryMask2Contour, contour2Skeleton



file_skel = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/135 CB4852 on food L_2011_03_09__15_51_36___1___8_skeletons.hdf5'
file_mask = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/MaskedVideos/135 CB4852 on food L_2011_03_09__15_51_36___1___8.hdf5'


with pd.HDFStore(file_skel, 'r') as fid:
    trajectories_data = fid['/trajectories_data']
    
    
n_sample = 49

current_frame = 250



time1 = 0
time2 = 0
with h5py.File(file_mask, 'r') as fid:
    
    for skeleton_id, row_data in trajectories_data.iterrows():

        frame_number = row_data['frame_number']
        threshold = row_data['threshold']
        
        #if frame_number < current_frame: continue        
        if frame_number > current_frame: break
        print(frame_number)
        
        worm_img = fid['/mask'][frame_number]
        worm_mask = getWormMask(worm_img, threshold)
        
        tic = time.time()        
        
        contour = binaryMask2Contour(worm_mask, min_mask_area=50)
        skeleton, cnt_side1, cnt_side2, cnt_widths, err_msg = \
        contour2Skeleton(contour)
        
        time1 += time.time() - tic
        print(time.time() - tic)
        
        tic = time.time()      
        output = getSkeleton(worm_mask, np.zeros(0), n_sample, 50)
        time2 += time.time() - tic
        print(time.time() - tic)
        
print(time1, time2)
        #plt.figure()
        #plt.plot(skeleton[:,1], skeleton[:,0])
        #plt.plot(smooth_skel[:,1], smooth_skel[:,0], 'o-')
        
        #plt.figure()
        #plt.plot(skeleton[:,1], skeleton[:,0])
        #plt.plot(output[0][:,1], output[0][:,0], 's-')
        
        