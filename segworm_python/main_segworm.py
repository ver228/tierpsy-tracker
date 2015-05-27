# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:46:20 2015

@author: ajaver
"""

import h5py
import matplotlib.pylab as plt
import cv2
import numpy as np
import time

from cleanWorm import cleanWorm, circSmooth, extremaPeaksCircDist
from linearSkeleton import linearSkeleton
from segWorm_cython import circComputeChainCodeLengths
from getHeadTail import getHeadTail, rollHead2FirstIndex

#wrappers around C functions
from circCurvature import circCurvature 
from curvspace import curvspace

def contour2Skeleton(contour):
    
    if type(contour) != np.ndarray or contour.ndim != 2 or contour.shape[1] !=2:
        err_msg =  'Contour must be a Nx2 numpy array'
        return 5*[np.zeros(0)]+[err_msg]
    
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

    
    contour = cleanWorm(contour, cnt_worm_segments)
    #% The contour is too small.
    if contour.shape[0] < cnt_worm_segments:
        err_msg =  'Contour is too small'
        return 5*[np.zeros(0)]+[err_msg]
    
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
        return 5*[np.zeros(0)]+[err_msg]
    
    #change arrays so the head correspond to the first position
    head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind = \
    rollHead2FirstIndex(head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind)
    
    #% Compute the contour's local low-frequency curvature minima.
    minima_low_freq, minima_low_freq_ind = \
    extremaPeaksCircDist(-1, cnt_ang_low_freq, edge_len_low_freq, cnt_chain_code_len);

    #% Compute the worm's skeleton.
    skeleton, cnt_widths = linearSkeleton(head_ind, tail_ind, minima_low_freq, minima_low_freq_ind, \
        maxima_low_freq, maxima_low_freq_ind, contour, worm_seg_length, cnt_chain_code_len);

    return skeleton, cnt_widths, head_ind, tail_ind, contour, ''

def binaryMask2Contour(worm_mask, min_mask_area=50, roi_center_x = -1, roi_center_y = -1, pick_center = True):
    if roi_center_x < 1:
        roi_center_x = (worm_mask.shape[1]-1)/2.
    if roi_center_y < 1:
        roi_center_y = (worm_mask.shape[0]-1)/2.
    
    #select only one contour in the binary mask
    #get contour
    contour, _ = cv2.findContours(worm_mask.copy(), cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)
    if len(contour) == 1:
        contour = np.squeeze(contour[0])
    elif len(contour)>1:
    #clean mask if there is more than one contour
        #select the largest area  object
        cnt_areas = [cv2.contourArea(cnt) for cnt in contour]
        
        #filter only contours with areas larger than min_mask_area
        cnt_tuple = [(contour[ii], cnt_area) for ii, cnt_area in enumerate(cnt_areas) if cnt_area>min_mask_area]
        if not cnt_tuple:
            return np.zeros([])
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
        contour = np.zeros([])
    return contour.astype(np.double)


if __name__ == '__main__':
    worm_name = 'worm_1717.hdf5' #file where the binary masks are stored
    resampling_N = 50; #number of resampling points of the skeleton
    
    fid = h5py.File(worm_name, 'r');
    data_set = fid['/masks'][:] #ROI with worm binary mask
    worm_CMs = fid['CMs'][:] #center of mass of the ROI
    
    total_frames = data_set.shape[0]
    tic = time.time()
    
    all_skeletons = np.empty((total_frames, resampling_N, 2))
    all_skeletons.fill(np.nan)
    
    for frame in range(total_frames):
        print frame, total_frames
        worm_mask = data_set[frame,:,:];

        contour = binaryMask2Contour(worm_mask)
        skeleton, cnt_widths, head_ind, tail_ind, contour, err_msg = \
        contour2Skeleton(contour)
    
        if skeleton.size == 0:
            continue
        
        skeleton, ske_len = curvspace(skeleton, 50)
        
        all_skeletons[frame, :, :] = skeleton;
              
    print time.time() - tic   
        
    #%% Plot all skeletons
    #plot every jump frames. otherwise the skeletons overlap too much
    jump = 25 
    
    #add the ROI CMs to show the worm displacements, this value would be offshift by worm_mask.shape/2
    xx = (all_skeletons[::jump,:,0] + worm_CMs[::jump,0][:, np.newaxis]).T
    yy = (all_skeletons[::jump,:,1] + worm_CMs[::jump,1][:, np.newaxis]).T
            
    plt.figure()
    plt.plot(xx,yy)
    #%% Plot the results of the last  mask
    plt.figure()
    plt.imshow(worm_mask, cmap= 'gray', interpolation = 'none')
    plt.plot(contour[:,0], contour[:,1], '.-b')
    plt.plot(skeleton[:,0], skeleton[:,1], 'x-g')
    plt.plot(contour[head_ind,0], contour[head_ind,1], 'or')
    plt.plot(contour[tail_ind,0], contour[tail_ind,1], 'sc')

