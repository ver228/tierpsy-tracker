# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:46:20 2015

@author: ajaver
"""

import cv2
import numpy as np
from scipy.interpolate import interp1d

from .linearSkeleton import linearSkeleton
from .getHeadTail import getHeadTail, rollHead2FirstIndex
from .cythonFiles.segWorm_cython import circComputeChainCodeLengths
from .cleanWorm import circSmooth, extremaPeaksCircDist

#wrappers around C functions
from .cythonFiles.circCurvature import circCurvature
from .cythonFiles.curvspace import curvspace

def contour2Skeleton(contour, resampling_N=50):
    #contour must be a Nx2 numpy array
    assert type(contour) == np.ndarray and contour.ndim == 2 and contour.shape[1] ==2
    
    if contour.dtype != np.double:
        contour = contour.astype(np.double)
    
    #% The worm is roughly divided into 24 segments of musculature (i.e., hinges
    #% that represent degrees of freedom) on each side. Therefore, 48 segments
    #% around a 2-D contour.
    #% Note: "In C. elegans the 95 rhomboid-shaped body wall muscle cells are
    #% arranged as staggered pairs in four longitudinal bundles located in four
    #% quadrants. Three of these bundles (DL, DR, VR) contain 24 cells each,
    #% whereas VL bundle contains 23 cells." - www.wormatlas.org
    ske_worm_segments = 24.;
    cnt_worm_segments = 2. * ske_worm_segments;

    #this line does not really seem to be useful
    #contour = cleanWorm(contour, cnt_worm_segments) 
    
    #% The contour is too small.
    if contour.shape[0] < cnt_worm_segments:
        err_msg =  'Contour is too small'
        return 4*[np.zeros(0)]+[err_msg]
    
    #make sure the contours are in the counter-clockwise direction
    #head tail indentification will not work otherwise
    #x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    signed_area = np.sum(contour[:-1,0]*contour[1:,1]-contour[1:,0]*contour[:-1,1])/2
    if signed_area>0:
        contour =  np.ascontiguousarray(contour[::-1,:])
    
    #make sure the array is C_continguous. Several functions required this.
    if not contour.flags['C_CONTIGUOUS']:
        contour = np.ascontiguousarray(contour)
    
    
    #% Compute the contour's local high/low-frequency curvature.
    #% Note: worm body muscles are arranged and innervated as staggered pairs.
    #% Therefore, 2 segments have one theoretical degree of freedom (i.e. one
    #% approximation of a hinge). In the head, muscles are innervated
    #% individually. Therefore, we sample the worm head's curvature at twice the
    #% frequency of its body.
    #% Note 2: we ignore Nyquist sampling theorem (sampling at twice the
    #% frequency) since the worm's cuticle constrains its mobility and practical
    #% degrees of freedom.

    cnt_chain_code_len = circComputeChainCodeLengths(contour);
    worm_seg_length = (cnt_chain_code_len[0] + cnt_chain_code_len[-1]) / cnt_worm_segments;
    
    edge_len_hi_freq = worm_seg_length;
    cnt_ang_hi_freq = circCurvature(contour, edge_len_hi_freq, cnt_chain_code_len);
    
    edge_len_low_freq = 2 * edge_len_hi_freq;
    cnt_ang_low_freq = circCurvature(contour, edge_len_low_freq, cnt_chain_code_len);
    
    #% Blur the contour's local high-frequency curvature.
    #% Note: on a small scale, noise causes contour imperfections that shift an
    #% angle from its correct location. Therefore, blurring angles by averaging
    #% them with their neighbors can localize them better.
    worm_seg_size = contour.shape[0] / cnt_worm_segments;
    blur_size_hi_freq = np.ceil(worm_seg_size / 2);
    cnt_ang_hi_freq = circSmooth(cnt_ang_hi_freq, blur_size_hi_freq)
        
    #% Compute the contour's local high/low-frequency curvature maxima.
    maxima_hi_freq, maxima_hi_freq_ind = \
    extremaPeaksCircDist(1, cnt_ang_hi_freq, edge_len_hi_freq, cnt_chain_code_len)
    
    maxima_low_freq, maxima_low_freq_ind = \
    extremaPeaksCircDist(1, cnt_ang_low_freq, edge_len_low_freq, cnt_chain_code_len)

    head_ind, tail_ind, err_msg = \
    getHeadTail(cnt_ang_low_freq, maxima_low_freq_ind, cnt_ang_hi_freq, maxima_hi_freq_ind, cnt_chain_code_len)
    
    if err_msg:
        return 4*[np.zeros(0)]+[err_msg]
    
    #change arrays so the head correspond to the first position
    head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind = \
    rollHead2FirstIndex(head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind)

    #% Compute the contour's local low-frequency curvature minima.
    minima_low_freq, minima_low_freq_ind = \
    extremaPeaksCircDist(-1, cnt_ang_low_freq, edge_len_low_freq, cnt_chain_code_len);
#%%
    #% Compute the worm's skeleton.
    skeleton, cnt_widths = linearSkeleton(head_ind, tail_ind, minima_low_freq, minima_low_freq_ind, \
        maxima_low_freq, maxima_low_freq_ind, contour.copy(), worm_seg_length, cnt_chain_code_len);

    #The head must be in position 0    
    assert head_ind == 0
    
    # Get the contour for each side.
    cnt_side1 = contour[:tail_ind+1, :].copy()
    cnt_side2 = np.vstack([contour[0,:], contour[:tail_ind-1:-1,:]])
    
    assert np.all(cnt_side1[0] == cnt_side2[0])
    assert np.all(cnt_side1[-1] == cnt_side2[-1])
    assert np.all(skeleton[-1] == cnt_side1[-1])
    assert np.all(skeleton[0] == np.round(cnt_side2[0]))
    
#%%    
    return skeleton, cnt_side1, cnt_side2, cnt_widths,  ''

def binaryMask2Contour(worm_mask, min_mask_area=50, roi_center_x = -1, roi_center_y = -1, pick_center = True):
    if roi_center_x < 1:
        roi_center_x = (worm_mask.shape[1]-1)/2.
    if roi_center_y < 1:
        roi_center_y = (worm_mask.shape[0]-1)/2.
    
    #select only one contour in the binary mask
    #get contour
    _,contour, _ = cv2.findContours(worm_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contour) == 1:
        contour = np.squeeze(contour[0])
    elif len(contour)>1:
    #clean mask if there is more than one contour
        #select the largest area  object
        cnt_areas = [cv2.contourArea(cnt) for cnt in contour]
        
        #filter only contours with areas larger than min_mask_area
        cnt_tuple = [(contour[ii], cnt_area) for ii, cnt_area in enumerate(cnt_areas) if cnt_area>min_mask_area]
        if not cnt_tuple:
            return np.zeros(0)
        contour, cnt_areas = zip(*cnt_tuple)
        
        if pick_center:
            #In the multiworm tracker the worm should be in the center of the ROI
            min_dist_center = np.inf;
            valid_ind = -1
            for ii, cnt in enumerate(contour):
                mm = cv2.moments(cnt)
                cm_x = mm['m10']/mm['m00']
                cm_y = mm['m01']/mm['m00']
                dist_center = (cm_x-roi_center_x)**2 + (cm_y-roi_center_y)**2
                if min_dist_center > dist_center:
                    min_dist_center = dist_center
                    valid_ind = ii
        else: 
            #select the largest area  object
            valid_ind = np.argmax(cnt_areas)
        
        #return the correct contour if there is a valid number
        contour = np.squeeze(contour[valid_ind])
    else:
        contour = np.zeros(0)
    return contour.astype(np.double)


def orientWorm(skeleton, prev_skeleton, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths):
    if skeleton.size == 0:
        return skeleton, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths
    
    #orient head tail with respect to hte previous worm
    if prev_skeleton.size > 0:
        dist2prev_head = np.sum((skeleton[0:3,:]-prev_skeleton[0:3,:])**2)
        dist2prev_tail = np.sum((skeleton[0:3,:]-prev_skeleton[-3:,:])**2)
        
        if dist2prev_head > dist2prev_tail: 
            #the skeleton is switched
            skeleton = skeleton[::-1,:]
            cnt_widths = cnt_widths[::-1]
            cnt_side1 = cnt_side1[::-1,:]
            cnt_side2 = cnt_side2[::-1,:]
        
    #make sure the contours are in the counter-clockwise direction
    #x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    contour = np.vstack((cnt_side1, cnt_side2[::-1,:])) 
    signed_area = np.sum(contour[:-1,0]*contour[1:,1]-contour[1:,0]*contour[:-1,1])/2
    if signed_area<0:
        cnt_side1, cnt_side2 = cnt_side2, cnt_side1
        cnt_side1_len, cnt_side2_len = cnt_side2_len, cnt_side1_len
    
    return skeleton, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths


def getSkeleton(worm_mask, prev_skeleton = np.zeros(0), resampling_N=50, min_mask_area = 50):
    contour = binaryMask2Contour(worm_mask, min_mask_area=50)
    if contour.size == 0:
        return 7*[np.zeros(0)]
    
    skeleton, cnt_side1, cnt_side2, cnt_widths, err_msg = \
    contour2Skeleton(contour, resampling_N)
    
    if skeleton.size == 0:
        return 7*[np.zeros(0)]
    
    #resample data
    skeleton, ske_len = curvspace(skeleton, resampling_N)
    cnt_side1, cnt_side1_len = curvspace(cnt_side1, resampling_N)
    cnt_side2, cnt_side2_len = curvspace(cnt_side2, resampling_N)
    
    f = interp1d(np.arange(cnt_widths.size), cnt_widths)
    x = np.linspace(0,cnt_widths.size-1, resampling_N)
    cnt_widths = f(x);
    
    #orient skeleton with respect to the previous skeleton
    skeleton, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths = \
    orientWorm(skeleton, prev_skeleton, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths)
    
    return skeleton, ske_len, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths
