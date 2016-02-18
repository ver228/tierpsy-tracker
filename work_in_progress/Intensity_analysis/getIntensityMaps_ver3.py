# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:15:48 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import tables
from scipy.interpolate import RectBivariateSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os
import matplotlib.pylab as plt

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.featuresAnalysis.obtainFeaturesHelper import WLAB, smoothCurve, calWormAngles


from MWTracker.trackWorms.getSkeletonsTables import getWormROI
from MWTracker.helperFunctions.timeCounterStr import timeCounterStr
from MWTracker.trackWorms.segWormPython.cythonFiles.curvspace import curvspace

def smoothSkeletons(skeleton, length_resampling = 131, smooth_win = 11, pol_degree = 3):
    xx = savgol_filter(skeleton[:,0], smooth_win, pol_degree)
    yy = savgol_filter(skeleton[:,1], smooth_win, pol_degree)
    
    ii = np.arange(xx.size)
    ii_new = np.linspace(0, xx.size-1, length_resampling)
    
    fx = interp1d(ii, xx)
    fy = interp1d(ii, yy)
    
    xx_new = fx(ii_new)
    yy_new = fy(ii_new)
    
    skel_new = np.vstack((xx_new, yy_new)).T
    return skel_new

#%%
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


def getStraightenWormInt(worm_img, skeleton, half_width, width_resampling):
    '''
        Code to straighten the worm worms.
        worm_image - image containing the worm
        skeleton - worm skeleton
        half_width - half width of the worm, if it is -1 it would try to calculated from cnt_widths
        cnt_widths - contour widths used in case the half width is not given
        width_resampling - number of data points used in the intensity map along the worm width
        length_resampling - number of data points used in the intensity map along the worm length
        ang_smooth_win - window used to calculate the skeleton angles. 
            A small value will introduce noise, therefore obtaining bad perpendicular segments.
            A large value will over smooth the skeleton, therefore not capturing the correct shape.
        
    '''
    assert half_width>0 or cnt_widths.size>0
    assert not np.any(np.isnan(skeleton))
    
    dX = np.diff(skeleton[:,0])
    dY = np.diff(skeleton[:,1])
    
    skel_angles = np.arctan2(dY, dX)
    skel_angles = np.hstack((skel_angles[0], skel_angles))
    
    #%get the perpendicular angles to define line scans (orientation doesn't
    #%matter here so subtracting pi/2 should always work)
    perp_angles = skel_angles - np.pi/2;
    
    #%for each skeleton point get the coordinates for two line scans: one in the
    #%positive direction along perpAngles and one in the negative direction (use
    #%two that both start on skeleton so that the intensities are the same in
    #%the line scan)
    
    r_ind = np.linspace(-half_width, half_width, width_resampling)
    
    #create the grid of points to be interpolated (make use of numpy implicit broadcasting Nx1 + 1xM = NxM)
    grid_x = skeleton[:,0] + r_ind[:, np.newaxis]*np.cos(perp_angles);
    grid_y = skeleton[:,1] + r_ind[:, np.newaxis]*np.sin(perp_angles);
    
    f = RectBivariateSpline(np.arange(worm_img.shape[0]), np.arange(worm_img.shape[1]), worm_img)
    straighten_worm =  f.ev(grid_y, grid_x) #return interpolated intensity map
    
    return straighten_worm, grid_x, grid_y #return interpolated intensity map

def getIntensityMaps(masked_image_file, skeletons_file, intensities_file, 
                     width_resampling = 15, length_resampling = 131, min_num_skel = 100,
                     smooth_win = 11, pol_degree = 3, width_percentage = 0.5):
    
    if length_resampling % 2 == 0: length_resampling += 1
    if width_resampling % 2 == 0: width_resampling += 1    
    #if width_average_win % 2 == 0: width_average_win += 1
    #assert width_average_win <= width_resampling
    assert smooth_win > pol_degree
    assert min_num_skel > 0
    
    #mid_w = width_resampling//2
    #win_w = width_average_win//2
    #width_win_ind = (mid_w-win_w, mid_w + win_w + 1) #add plus one to use the correct numpy indexing
    
    table_filters = tables.Filters(complevel=5, complib='zlib', 
                                   shuffle=True, fletcher32=True)


    #might be better to a complete table with the int_map_id in the skeletons_file, and maybe a copy in intensities_file
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        if 'auto_label' in trajectories_data:
            good = trajectories_data['auto_label'] == WLAB['GOOD_SKE'];
            trajectories_data = trajectories_data[good]
    
            N = trajectories_data.groupby('worm_index_joined').agg({'has_skeleton':np.nansum})
            N = N[N>min_num_skel].dropna()
    
            good = trajectories_data['worm_index_joined'].isin(N.index)
            trajectories_data = trajectories_data.loc[good]
        else:
            trajectories_data = trajectories_data[trajectories_data['has_skeleton']==1]
        
        #trajectories_data = trajectories_data[trajectories_data['frame_number']<500]
       
       
       
    trajectories_data['int_map_id'] = np.arange(len(trajectories_data))
    
    tot_rows = len(trajectories_data)
        
    #let's save this data into the intensities file
    with tables.File(intensities_file, 'w') as fid:
        fid.create_table('/', 'trajectories_data', \
            obj = trajectories_data.to_records(index=False), filters=table_filters)

    with tables.File(masked_image_file, 'r')  as mask_fid, \
         tables.File(skeletons_file, 'r') as ske_file_id, \
         tables.File(intensities_file, "r+") as int_file_id:
        
        #pointer to the compressed videos
        mask_dataset = mask_fid.get_node("/mask")
        
        #pointer to skeletons
        skel_tab = ske_file_id.get_node('/skeleton')
        skel_width_tab = ske_file_id.get_node('/width_midbody')
        

        #create array to save the intensities
        filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)
        
        worm_int_avg_tab = int_file_id.create_carray("/", "straighten_worm_intensity_median", \
                                   tables.Float32Atom(dflt=np.nan), \
                                   (tot_rows, length_resampling), \
                                    chunkshape = (1, length_resampling),\
                                    filters = filters);
        worm_int_std_tab = int_file_id.create_carray("/", "worm_intensity_std", \
                                   tables.Float32Atom(dflt=np.nan), \
                                   (tot_rows,),\
                                    filters = filters);
                                    
        progressTime = timeCounterStr('Obtaining intensity maps.');
        
        for frame, frame_data in trajectories_data.groupby('frame_number'):
            img = mask_dataset[frame,:,:]
            for ii, row_data in frame_data.iterrows():
                skeleton_id = int(row_data['skeleton_id'])
                worm_index = int(row_data['worm_index_joined'])
                int_map_id = int(row_data['int_map_id'])
                
                worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], row_data['roi_size'])
                
                skeleton = skel_tab[skeleton_id,:,:]-roi_corner
                half_width = (width_percentage*skel_width_tab[skeleton_id])/2
                
                assert not np.isnan(skeleton[0,0])
                
                skel_smooth = smoothSkeletons(skeleton, length_resampling = length_resampling, smooth_win = smooth_win, pol_degree = pol_degree)
                straighten_worm,grid_x, grid_y = getStraightenWormInt(worm_img, skel_smooth, half_width=half_width, width_resampling=width_resampling)
                
                #if you use the mean it is better to do not use float16
                #int_avg = np.median(straighten_worm[width_win_ind[0]:width_win_ind[1],:], axis = 0)
                int_avg = np.median(straighten_worm, axis = 0)
                worm_int_avg_tab[int_map_id] = int_avg
                worm_int_std_tab[int_map_id] = np.std(straighten_worm)
                
            if frame % 500 == 0:
                progress_str = progressTime.getStr(frame)
                print('' + ' ' + progress_str);

if __name__ == '__main__':
    #base directory
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch2_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch3_17112015_205616.hdf5'
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch6_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
    
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')
    #parameters
    
    dd = np.asarray([131, 15, 7])#*2+1    
    argkws = {'width_resampling':dd[1], 'length_resampling':dd[0], 'min_num_skel':100,
                     'smooth_win':dd[2], 'pol_degree':3}
    getIntensityMaps(masked_image_file, skeletons_file, intensities_file, **argkws)
    
    
    
    


