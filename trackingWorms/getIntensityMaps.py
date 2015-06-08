# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:15:48 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import tables
from scipy.interpolate import RectBivariateSpline
import sys
import os
#import matplotlib.pylab as plt

from getSkeletonsTables import getWormROI


sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr

sys.path.append('../segworm_python/')
from curvspace import curvspace

def angleSmoothed(x, y, window_size):
    #given a series of x and y coordinates over time, calculates the angle
    #between each tangent vector over a given window making up the skeleton
    #and the x-axis.
    #arrays to build up and export
    dX = x[:-window_size] - x[window_size:];
    dY = y[:-window_size] - y[window_size:];
    
    #calculate angles
    skel_angles = np.arctan2(dY, dX)
    
    
    #%repeat final angle to make array the same length as skelX and skelY
    skel_angles = np.lib.pad(skel_angles, (window_size//2, window_size//2), 'edge')
    return skel_angles;

def getStraightenWormInt(worm_img, skeleton, half_width = -1, cnt_widths  = np.zeros(0), width_resampling = 7, ang_smooth_win = 6, length_resampling = 49):
    
    #if np.all(np.isnan(skeleton)):
    #    buff = np.empty((skeleton.shape[0], width_resampling))
    #    buff.fill(np.nan)
    #    return buff
    assert half_width>0 or cnt_widths.size>0
    assert not np.any(np.isnan(skeleton))
    
    if ang_smooth_win%2 == 1:
        ang_smooth_win += 1; 
    
    if skeleton.shape[0] != length_resampling:
        skeleton, _ = curvspace(np.ascontiguousarray(skeleton), length_resampling)
    
    skelX = skeleton[:,0];
    skelY = skeleton[:,1];
    
    assert np.max(skelX) < worm_img.shape[0]
    assert np.max(skelY) < worm_img.shape[1]
    assert np.min(skelY) >= 0
    assert np.min(skelY) >= 0
    
    #calculate smoothed angles
    skel_angles = angleSmoothed(skelX, skelY, ang_smooth_win)
    
    #%get the perpendicular angles to define line scans (orientation doesn't
    #%matter here so subtracting pi/2 should always work)
    perp_angles = skel_angles - np.pi/2;
    
    #%for each skeleton point get the coordinates for two line scans: one in the
    #%positive direction along perpAngles and one in the negative direction (use
    #%two that both start on skeleton so that the intensities are the same in
    #%the line scan)
    
    #resample the points along the worm width
    if half_width <= 0:
        half_width = (np.median(cnt_widths[10:-10])/2.) #add half a pixel to get part of the contour
    r_ind = np.linspace(-half_width, half_width, width_resampling)
    
    #create the grid of points to be interpolated (make use of numpy implicit broadcasting Nx1 + 1xM = NxM)
    grid_x = skelX + r_ind[:, np.newaxis]*np.cos(perp_angles);
    grid_y = skelY + r_ind[:, np.newaxis]*np.sin(perp_angles);
    
    
    f = RectBivariateSpline(np.arange(worm_img.shape[0]), np.arange(worm_img.shape[1]), worm_img)
    return f.ev(grid_y, grid_x) #return interpolated intensity map


base_name = 'Capture_Ch3_12052015_194303'
mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    

#root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
#base_name = 'Capture_Ch1_11052015_195105'

masked_image_file = mask_dir + base_name + '.hdf5'
trajectories_file = results_dir + base_name + '_trajectories.hdf5'
skeletons_file = results_dir + base_name + '_skeletons.hdf5'
intensities_file = results_dir + base_name + '_intensities.hdf5'

#%%
base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

#MAKE VIDEOStho
roi_size = 128
width_resampling = 13
length_resampling = 121


with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
    #data to extract the ROI
    trajectories_df = ske_file_id['/trajectories_data']

    #get the first and last frame of each worm_index
    indexes_data = trajectories_df[['worm_index_joined', 'skeleton_id']]
    rows_indexes = indexes_data.groupby('worm_index_joined').agg([min, max])['skeleton_id']
    del indexes_data

#def getIntensitiesMap(masked_image_file, skeletons_file, intensities_file, roi_size = 128):
with tables.File(masked_image_file, 'r')  as mask_fid, \
     tables.File(skeletons_file, 'r') as ske_file_id, \
     tables.File(intensities_file, "w") as int_file_id:
    
    #pointer to the compressed videos
    mask_dataset = mask_fid.get_node("/mask")
    
    #obtain a fix length for the worm half width
    half_widths = {}
    mid_cnt = ske_file_id.get_node('/contour_width').shape[1]/2
    for worm_index, row_range in rows_indexes.iterrows():
        contour_width = ske_file_id.get_node('/contour_width')[row_range['min']:row_range['max']+1, :]
        half_widths[worm_index] = np.nanmedian(contour_width[:, mid_cnt])/2 + 0.5
    
    #pointer to skeletons
    skel_tab = ske_file_id.get_node('/skeleton')
    
    #get first and last frame for each worm
    worms_frame_range = trajectories_df.groupby('worm_index_joined').agg({'frame_number': [min, max]})['frame_number']
    tot_rows = len(trajectories_df)
    
    #create array to save the intensities
    filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)
    worm_int_tab = int_file_id.create_carray("/", "straighten_worm_intensity", \
                               tables.Float16Atom(dflt=np.nan), \
                               (tot_rows, length_resampling, width_resampling), \
                                chunkshape = (1, length_resampling, width_resampling),\
                                filters = filters);
    
    progressTime = timeCounterStr('Obtaining intensity maps.');
    for frame, frame_data in trajectories_df.groupby('frame_number'):
        img = mask_dataset[frame,:,:]
        for segworm_id, row_data in frame_data.iterrows():
            worm_index = row_data['worm_index_joined']
            worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], roi_size)
            
            skeleton = skel_tab[segworm_id,:,:]-roi_corner
            
            if not np.any(np.isnan(skeleton)): 
                straighten_worm = getStraightenWormInt(worm_img, skeleton, half_width = half_widths[worm_index], \
                width_resampling = width_resampling, length_resampling = length_resampling)
                worm_int_tab[segworm_id, :, :]  = straighten_worm.T
                
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);   
    
    mask_fid.close()
    int_file_id.close()
#%%

 
#plt.plot(grid_x, grid_y, '.')

#plt.figure()
#plt.imshow(straighten_img, cmap='gray', interpolation= 'none')    

#plt.figure()
#tt = worm_img[np.round(grid_y).astype(np.int), np.round(grid_x).astype(np.int)]
#plt.imshow(tt, cmap='gray', interpolation= 'none')    

#%%


    #if frame % 500 == 0:
    #    progress_str = progressTime.getStr(frame)
    #    print(base_name + ' ' + progress_str);

#for kk in [41, 42]:
#    plt.figure()
#    for frame in range(kk*1000, (kk+1)*1000, 100):
#        xx = np.nanmean(np.nanmean(worm_intensity[frame:(frame+ 1),:,2:5], axis=0),axis=1)
#        plt.plot(xx)
#
#mid_intensity = np.mean(worm_intensity[:,:,2:5], axis=2);
##%%
#
#for kk in range(0,51, 5):
#    plt.figure()
#    plt.plot(median_filter(mid_intensity[:, kk],1000))



#%%
#import tifffile
#N = 25.
#windows = np.ones(N)/N
#worm_intensity2 = np.zeros((worm_intensity.shape[0], worm_intensity.shape[2], worm_intensity.shape[1]))
#for ii in range(worm_intensity.shape[1]):
#    for jj in range(worm_intensity.shape[2]):
#        worm_intensity2[:,jj,ii] = np.convolve(worm_intensity[:,ii,jj], windows, 'same')
#
#tifffile.imsave('/Users/ajaver/Desktop/Gecko_compressed/image.tiff', worm_intensity2.astype(np.uint8), compress=4) 