#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:59:23 2017

@author: ajaver
"""

import numpy as np
import warnings
import pandas as pd

from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from .helper import nanunwrap, DataPartition
from .postures import get_length

curvature_columns = [
        'curvature_head', 
        'curvature_hips', 
        'curvature_midbody',
        'curvature_neck', 
        'curvature_tail', 
        'curvature_mean_head',
        'curvature_mean_neck',
        'curvature_mean_midbody',
        'curvature_mean_hips',
        'curvature_mean_tail',
        'curvature_std_head', 
        'curvature_std_neck', 
        'curvature_std_midbody', 
        'curvature_std_hips', 
        'curvature_std_tail'
        ]

def _curvature_angles(skeletons, window_length = None, lengths=None):
    if window_length is None:
        window_length = 7

    points_window = int(round(window_length/2))
    
    def _tangent_angles(skels, points_window):
        '''this is a vectorize version to calculate the angles between segments
        segment_size points from each side of a center point.
        '''
        s_center = skels[:, points_window:-points_window, :] #center points
        s_left = skels[:, :-2*points_window, :] #left side points
        s_right = skels[:, 2*points_window:, :] #right side points
        
        d_left = s_left - s_center 
        d_right = s_center - s_right
        
        #arctan2 expects the y,x angle
        ang_l = np.arctan2(d_left[...,1], d_left[...,0])
        ang_r = np.arctan2(d_right[...,1], d_right[...,0])
        
        with warnings.catch_warnings():
            #I am unwraping in one dimension first
            warnings.simplefilter("ignore")
            ang = np.unwrap(ang_r-ang_l, axis=1);
        
        for ii in range(ang.shape[1]):
            ang[:, ii] = nanunwrap(ang[:, ii])
        return ang
    
    if lengths is None:
        #caculate the length if it is not given
        lengths = get_length(skeletons)
    
    #Number of segments is the number of vertices minus 1
    n_segments = skeletons.shape[1] -1 
    
    #This is the fraction of the length the angle is calculated on
    length_frac = 2*(points_window-1)/(n_segments-1)
    segment_length = length_frac*lengths
    segment_angles = _tangent_angles(skeletons, points_window)
    
    curvature = segment_angles/segment_length[:, None]
    
    return curvature


def _curvature_savgol(skeletons, window_length = None, length=None):
    '''
    Calculate the curvature using univariate splines. This method is slower and can fail
    badly if the fit does not work, so I am only using it as testing
    '''

    if window_length is None:
        window_length = 7

    def _fitted_curvature(skel):
        if np.any(np.isnan(skel)):
            return np.full(skel.shape[0], np.nan)
        
        x = skel[:, 0]
        y = skel[:, 1]

        x_d = savgol_filter(x, window_length=window_length, polyorder=3, deriv=1)
        y_d = savgol_filter(y, window_length=window_length, polyorder=3, deriv=1)
        x_dd = savgol_filter(x, window_length=window_length, polyorder=3, deriv=2)
        y_dd = savgol_filter(y, window_length=window_length, polyorder=3, deriv=2)
        curvature = _curvature_fun(x_d, y_d, x_dd, y_dd)
        return  curvature

    
    curvatures_fit = np.array([_fitted_curvature(skel) for skel in skeletons])
    return curvatures_fit


def _curvature_spline(skeletons, points_window=None, length=None):
    '''
    Calculate the curvature using univariate splines. This method is slower and can fail
    badly if the fit does not work, so I am only using it as testing
    '''

    def _spline_curvature(skel):
        if np.any(np.isnan(skel)):
            return np.full(skel.shape[0], np.nan)
        
        x = skel[:, 0]
        y = skel[:, 1]
        n = np.arange(x.size)

        fx = UnivariateSpline(n, x, k=5)
        fy = UnivariateSpline(n, y, k=5)

        x_d = fx.derivative(1)(n)
        x_dd = fx.derivative(2)(n)
        y_d = fy.derivative(1)(n)
        y_dd = fy.derivative(2)(n)

        curvature = _curvature_fun(x_d, y_d, x_dd, y_dd)
        return  curvature

    
    curvatures_fit = np.array([_spline_curvature(skel) for skel in skeletons])
    return curvatures_fit

#%%
def _curvature_fun(x_d, y_d, x_dd, y_dd):
    return (x_d*y_dd - y_d*x_dd)/(x_d*x_d + y_d*y_d)**1.5

def _gradient_windowed(X, points_window, axis):
    '''
    Calculate the gradient using an arbitrary window. The larger window make 
    this procedure less noisy that the numpy native gradient.
    '''
    w_s = 2*points_window
    
    #I use slices to deal with arbritary dimenssions 
    #https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    n_axis_ini = max(0, axis)
    n_axis_fin = max(0, X.ndim-axis-1)
    
    right_slice = [slice(None, None, None)]*n_axis_ini + [slice(None, -w_s, None)]
    right_slice = tuple(right_slice)
    
    left_slice = [slice(None, None, None)]*n_axis_ini + [slice(w_s, None, None)]
    left_slice = tuple(left_slice)

    right_pad = [(0,0)]*n_axis_ini + [(w_s, 0)] + [(0,0)]*n_axis_fin
    left_pad = [(0,0)]*n_axis_ini + [(0, w_s)] + [(0,0)]*n_axis_fin
    
    right_side = np.pad(X[right_slice], right_pad, 'edge')
    left_side = np.pad(X[left_slice], left_pad, 'edge')
    
    ramp = np.full(X.shape[axis]-2*w_s, w_s*2)
    
    ramp = np.pad(ramp,  pad_width = (w_s, w_s),  mode='linear_ramp', end_values = w_s)
    #ramp = np.pad(ramp,  pad_width = (w_s, w_s),  mode='constant', constant_values = np.nan)
    ramp_slice = [None]*n_axis_ini + [slice(None, None, None)] + [None]*n_axis_fin
    ramp_slice = tuple(ramp_slice)

    grad = (left_side - right_side) / ramp[ramp_slice] #divide it by the time window
    
    return grad

def curvature_grad(curve, points_window=None, axis=1, is_nan_border=True):
    '''
    Calculate the curvature using the gradient using differences similar to numpy grad
    
    x1, x2, x3
    
    grad(x2) = (x3-x1)/2
    
    '''
    
    #The last element must be the coordinates
    assert curve.shape[-1] == 2
    assert axis != curve.ndim - 1    
    
    if points_window is None:
        points_window = 1
    
    if curve.shape[0] <= points_window*4:
        return np.full((curve.shape[0], curve.shape[1]), np.nan)
    
    d = _gradient_windowed(curve, points_window, axis=axis)
    dd = _gradient_windowed(d, points_window, axis=axis)
    
    gx = d[..., 0]
    gy = d[..., 1]
    ggx = dd[..., 0]
    ggy = dd[..., 1]
    
    curvature_r =  _curvature_fun(gx, gy, ggx, ggy)
    if is_nan_border:
        #I cannot really trust in the border gradient
        w_s = 4*points_window
        n_axis_ini = max(0, axis)
        right_slice = [slice(None, None, None)]*n_axis_ini + [slice(None, w_s, None)]
        right_slice = tuple(right_slice)

        left_slice = [slice(None, None, None)]*n_axis_ini + [slice(-w_s, None, None)]
        left_slice = tuple(left_slice)
        
        curvature_r[right_slice] = np.nan
        curvature_r[left_slice] = np.nan
    
    return curvature_r
#%%  

def get_curvature_features(skeletons, method = 'grad', points_window=None):
    curvature_funcs = {
            'angle' : _curvature_angles, 
            'spline' : _curvature_spline, 
            'savgol' : _curvature_savgol,
            'grad' : curvature_grad
            }
    
    
    assert method in curvature_funcs
    
    if method == 'angle':
        segments_ind_dflt = {
            'head' : 0,
            'neck' : 0.25,
            'midbody' : 0.5, 
            'hips' : 0.75,
            'tail' : 1.,
        }
    else:
        segments_ind_dflt = {
            'head' : 5/48,
            'neck' : 15/48,
            'midbody' : 24/48, 
            'hips' : 33/48,
            'tail' : 44/48,
        }
    
    curvatures = curvature_funcs[method](skeletons, points_window)
    max_angle_index = curvatures.shape[-1]-1
    segments_ind = {k:int(round(x*max_angle_index)) for k,x in segments_ind_dflt.items()}
    
    curv_dict = {'curvature_' + x :curvatures[:, ind] for x,ind in segments_ind.items()}
    
    #standard deviation of the curvature around the segments (seems to be usefull in classification)
    p_obj = DataPartition(list(segments_ind_dflt.keys()), n_segments = skeletons.shape[1])
    
    #i need to use nan because the curvature at the ends is not defined
    curv_std = p_obj.apply_partitions(curvatures, func=np.nanstd)
    for key, val in curv_std.items():
        curv_dict['curvature_std_' + key] = val
    
    #i need to use nan because the curvature at the ends is not defined
    curv_mean = p_obj.apply_partitions(curvatures, func=np.nanmean)
    for key, val in curv_mean.items():
        curv_dict['curvature_mean_' + key] = val
    
    data = pd.DataFrame.from_dict(curv_dict)
    
    return data

#%%

if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    R = 1
    
    ang = np.linspace(-np.pi, np.pi, 50)
    curve = np.array([np.cos(ang), np.sin(ang)]).T*R
    curvature = curvature_grad(curve, axis=0)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(curve[:, 0], curve[:, 1], '.-')
    plt.axis('equal')
    
    plt.subplot(1,2,2)
    plt.plot(curvature)
    
    k = 1/R
    plt.ylim(k - k/2, k + k/2)
    
