#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""
import pandas as pd
import numpy as np
import matplotlib.path as mplPath
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from .velocities import _h_segment_position

food_columns = ['orientation_food_edge', 'dist_from_food_edge']

#%%
def _is_valid_cnt(x):
    return x is not None and \
           x.size >= 2 and \
           x.ndim ==2 and \
           x.shape[1] == 2

def _h_smooth_cnt(food_cnt, resampling_N = 1000, smooth_window=None, _is_debug=False):
    if smooth_window is None:
        smooth_window = resampling_N//20
    
    if not _is_valid_cnt(food_cnt):
        #invalid contour arrays
        return food_cnt
        
    smooth_window = smooth_window if smooth_window%2 == 1 else smooth_window+1
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(food_cnt[:, 0])
    dy = np.diff(food_cnt[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)
    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]
    fx = interp1d(lengths, food_cnt[:, 0])
    fy = interp1d(lengths, food_cnt[:, 1])
    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)
    
    rx = fx(subLengths)
    ry = fy(subLengths)
    
    pol_degree = 3
    rx = savgol_filter(rx, smooth_window, pol_degree, mode='wrap')
    ry = savgol_filter(ry, smooth_window, pol_degree, mode='wrap')
    
    food_cnt_s = np.stack((rx, ry), axis=1)
    
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(food_cnt[:, 0], food_cnt[:, 1], '.-')
        plt.plot(food_cnt_s[:, 0], food_cnt_s[:, 1], '.-')
        plt.axis('equal')
        plt.title('smoothed contour')
    
    return food_cnt_s

#%%
def _h_get_unit_vec(x):
    return x/np.linalg.norm(x, axis=1)[:, np.newaxis]
#%%
def get_cnt_feats(skeletons, 
                  food_cnt,
                  is_smooth_cnt = True,
                  _is_debug = False):
    
    if is_smooth_cnt:
        food_cnt = _h_smooth_cnt(food_cnt)
    #%%
    worm_coords, orientation_v = _h_segment_position(skeletons, partition = 'body')
    
    rr = np.linalg.norm(worm_coords[:, None, :] - food_cnt[None, ...], axis=2)
    cnt_ind = np.argmin(rr, axis=1)
    dist_from_cnt = np.array([x[i] for i,x in zip(cnt_ind, rr)])
    bbPath = mplPath.Path(food_cnt)
    outside = ~bbPath.contains_points(worm_coords)
    dist_from_cnt[outside] = -dist_from_cnt[outside]
    worm_u = _h_get_unit_vec(orientation_v)
    #%%
    top = cnt_ind+1
    top[top>=food_cnt.shape[0]] -= food_cnt.shape[0] #fix any overflow index
    bot = cnt_ind-1 #it is not necessary to correct because we can use negative indexing
    
    #I am using the normal vector so the orientation can be calculated between -90 and 90
    #positive if the worm is pointing towards the food center and negative if it is looking out
    food_u =  _h_get_unit_vec(food_cnt[top]-food_cnt[bot])
    R = np.array([[0,1], [-1, 0]])
    food_u = (np.dot(R, food_u.T)).T
        
    dot_prod = np.sum(food_u*worm_u, axis=1) 
    
    with np.errstate(invalid='ignore'):
        orientation_food_cnt = 90-np.arccos(dot_prod)*180/np.pi
        
    #%%
    dd = np.array([orientation_food_cnt, dist_from_cnt]).T
    food_df = pd.DataFrame(dd, columns = food_columns)
    #%%
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure(figsize=(24,12))
        
        plt.subplot(2,2,2)
        plt.plot(orientation_food_cnt)
        plt.title('Orientation respect to the food contour')
        
        plt.subplot(2,2,4)
        plt.plot(dist_from_cnt)
        plt.title('Distance from the food contour')
        
        plt.subplot(1,2,1)
        plt.plot(food_cnt[:,0], food_cnt[:,1])
        plt.plot(worm_coords[:,0], worm_coords[:,1], '.')
        plt.plot(food_cnt[cnt_ind,0], food_cnt[cnt_ind,1], 'r.')
        plt.axis('equal')
        
      
    return food_df


