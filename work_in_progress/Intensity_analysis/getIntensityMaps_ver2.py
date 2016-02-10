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


def getStraightenWormInt(worm_img, skeleton, half_width):
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
    
    #if ang_smooth_win%2 == 1: ang_smooth_win += 1; 
    
    #assert np.max(skelX) < worm_img.shape[0]
    #assert np.max(skelY) < worm_img.shape[1]
    #assert np.min(skelY) >= 0
    #assert np.min(skelY) >= 0
    
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
    
    
    mid_c = width_resampling//2 #middle index
    #print(grid_x.shape)
    
    #check if the grid orientation is correct    
    #%% x1y2 + x2y3 + x3y1 - x2y1 - x3y2 - x1y3
    def triangSignedArea(xx,yy): 
        return xx[0]*yy[1] + xx[1]*yy[2] + xx[2]*yy[0] - yy[0]*xx[1] - yy[1]*xx[2] - yy[2]*xx[0]
    
    A = []
    xx = (grid_x[-1,0], grid_x[0,0], grid_x[mid_c,1])
    yy = (grid_y[-1,0], grid_y[0,0], grid_y[mid_c,1])
    
    A.append(triangSignedArea(xx,yy))
    
    for ii in range(1,grid_x.shape[1]):
        xx = (grid_x[0,ii], grid_x[-1,ii], grid_x[mid_c,ii-1])
        yy = (grid_y[0,ii], grid_y[-1,ii], grid_y[mid_c,ii-1])
        
        A.append(triangSignedArea(xx,yy))
    
    assert (all(a<0 for a in A))
    #signed_area = np.sum(contour[:, :-1,0]*contour[:, 1:,1]-contour[:, 1:,0]*contour[:, :-1,1], axis=1)
    
    
    
    
    f = RectBivariateSpline(np.arange(worm_img.shape[0]), np.arange(worm_img.shape[1]), worm_img)
    straighten_worm =  f.ev(grid_y, grid_x) #return interpolated intensity map
    
    return straighten_worm, grid_x, grid_y #return interpolated intensity map


if __name__ == '__main__':
    #base directory
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')
    
    #MAKE VIDEOStho
    roi_size = 128
    width_resampling = 15
    length_resampling = 131
    skel_n_points = 49

    #parameters
    smooth_win = 11
    pol_degree = 3
    
    #minimal fraction of skeletonized frames in a trajectory to considere valid
    #bad_seg_thresh = 0.5
    min_num_skel = 500

    table_filters = tables.Filters(complevel=5, complib='zlib', 
                                   shuffle=True, fletcher32=True)


    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']

        good = trajectories_data['auto_label'] == WLAB['GOOD_SKE'];
        trajectories_data = trajectories_data[good]

        N = trajectories_data.groupby('worm_index_joined').agg({'has_skeleton':np.nansum})
        N = N[N>min_num_skel].dropna()

        good = trajectories_data['worm_index_joined'].isin(N.index)
        trajectories_data = trajectories_data.loc[good]
    
    trajectories_data['int_map_id'] = np.arange(len(trajectories_data))
        
    
    #test, only with a single index and less time
    #trajectories_data = trajectories_data[trajectories_data['worm_index_joined'] == 2]
    #trajectories_data = trajectories_data[trajectories_data['frame_number'] < 10]
    
    
    tot_rows = len(trajectories_data)
    
    
    #let's save this data into the intensities file
    with tables.File(intensities_file, 'w') as fid:
        fid.create_table('/', 'trajectories_data', \
            obj = trajectories_data.to_records(index=False), filters=table_filters)

    #def getIntensitiesMap(masked_image_file, skeletons_file, intensities_file, roi_size = 128):
    with tables.File(masked_image_file, 'r')  as mask_fid, \
         tables.File(skeletons_file, 'r') as ske_file_id, \
         tables.File(intensities_file, "r+") as int_file_id:
        
        #pointer to the compressed videos
        mask_dataset = mask_fid.get_node("/mask")
        
        #pointer to skeletons
        skel_tab = ske_file_id.get_node('/skeleton')
        skel_width_tab = ske_file_id.get_node('/width_midbody')
        

        #get first and last frame for each worm
        
        #create array to save the intensities
        filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)
        
        skel_smooth_tab = int_file_id.create_carray("/", "skeletons_smoothed", \
                                   tables.Float16Atom(dflt=np.nan), \
                                   (tot_rows, length_resampling, 2), \
                                    chunkshape = (1, length_resampling, 2),\
                                    filters = filters);
        
        worm_int_tab = int_file_id.create_carray("/", "straighten_worm_intensity", \
                                   tables.Float16Atom(dflt=np.nan), \
                                   (tot_rows, length_resampling,width_resampling), \
                                    chunkshape = (1, length_resampling,width_resampling),\
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
                half_width = skel_width_tab[skeleton_id]/2
                
                assert not np.isnan(skeleton[0,0])
                
                skel_smooth = smoothSkeletons(skeleton, length_resampling = length_resampling, smooth_win = smooth_win, pol_degree = pol_degree)
                straighten_worm,grid_x, grid_y = getStraightenWormInt(worm_img, skel_smooth, half_width)
                
                #straighten_worm = getStraightenWormInt(worm_img, skeleton, half_width = half_widths[worm_index], \
                #width_resampling = width_resampling, length_resampling = length_resampling)
                skel_smooth_tab[int_map_id] = skel_smooth             
                worm_int_tab[int_map_id]  = straighten_worm.T

            if frame % 500 == 0:
                progress_str = progressTime.getStr(frame)
                print('' + ' ' + progress_str);   
#%%
with tables.File(intensities_file, "r+") as int_file_id:
    straighten_worm = int_file_id.get_node('/straighten_worm_intensity')[0]
    plt.imshow(straighten_worm, interpolation='none', cmap='gray')
    plt.grid('off')
