# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:07:51 2015

@author: ajaver
"""
import pdb


import matplotlib.pylab as plt
import sys
import numpy as np
from math import ceil
sys.path.append('../segworm_python/')

from cleanWorm import cleanWorm, circSmooth, extremaPeaksCircDist
from linearSkeleton import linearSkeleton
from segWorm_cython import circComputeChainCodeLengths
from getHeadTail import getHeadTail, rollHead2FirstIndex
from circCurvature import circCurvature 
from curvspace import curvspace

from linearSkeleton_cython import chainCodeLength2Index, circOpposingNearestPoints, \
getHeadTailRegion, getInterBendSeeds, skeletonize, cleanSkeleton

from linearSkeleton import getMidBodySeed, getBendsSeeds, getSkeleton

from main_segworm import contour2Skeleton

def absDiff(a, b): 
    return a-b if a>b else b-a
    
if __name__ == '__main__':
    contour = np.load('contour_dum.npy')
    #pdb.set_trace()
    #a = contour2Skeleton(contour)


    
    
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
        raise
        #return 7*[np.zeros(0)]+[err_msg]
    
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
        raise
        #return 7*[np.zeros(0)]+[err_msg]
    
    #change arrays so the head correspond to the first position
    head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind = \
    rollHead2FirstIndex(head_ind, tail_ind, contour, cnt_chain_code_len, cnt_ang_low_freq, maxima_low_freq_ind)

    #% Compute the contour's local low-frequency curvature minima.
    minima_low_freq, minima_low_freq_ind = \
    extremaPeaksCircDist(-1, cnt_ang_low_freq, edge_len_low_freq, cnt_chain_code_len);
#%%
    #% Compute the worm's skeleton.
    #skeleton, cnt_widths = linearSkeleton(head_ind, tail_ind, minima_low_freq, minima_low_freq_ind, \
    #    maxima_low_freq, maxima_low_freq_ind, contour.copy(), worm_seg_length, cnt_chain_code_len);
    search_edge_size = cnt_chain_code_len[-1] / 8.;
    midbody_tuple =  getMidBodySeed(contour, cnt_chain_code_len, head_ind, tail_ind, search_edge_size)   
    
    #% Find the large minimal bends away from the head and tail.
    bend_ind = np.append(minima_low_freq_ind[minima_low_freq<-20], maxima_low_freq_ind[maxima_low_freq>20])
    bend_side1, bend_side2, midbody_ind = getBendsSeeds(contour, bend_ind, cnt_chain_code_len, \
    head_ind, tail_ind, midbody_tuple, worm_seg_length, search_edge_size)

    #get inter-bend seeds
    interbend_side1, interbend_side2 = getInterBendSeeds(bend_side1, bend_side2, contour, cnt_chain_code_len)
    #%%
    #get skeleton and contour cnt_widths
    skeleton, cnt_widths = getSkeleton(contour, head_ind, tail_ind, midbody_ind, bend_side1, bend_side2, interbend_side1, interbend_side2)
    assert (skeleton.size > 0) and (skeleton.ndim == 2)
    skeleton, cnt_widths = cleanSkeleton(skeleton, cnt_widths, worm_seg_length);
    
    
#    #cdef int 
#    FLAG_MAX = 2147483647 #max 32 integer. initialization
#    #cdef int 
#    maxSkeletonOverlap = (int)(ceil(2 * worm_seg_size));
#    #cdef int 
#    number_points = skeleton.shape[0]
#    buff_size = 2*number_points;
#    #cdef int 
#    last_index = number_points - 1
#    
#    #cdef np.ndarray[np.int_t, ndim=1] 
#    pSortC = np.lexsort((skeleton[:,1], skeleton[:,0])) 
#    #cdef np.ndarray[np.int_t, ndim=1] 
#    iSortC = np.argsort(pSortC)
#    
#    #cdef np.ndarray[np.int_t, ndim=1] 
#    keep = np.arange(number_points)
#    
#    #cdef int minI, maxI;
#    #cdef int 
#    s1I = 0; #// % the first index for the skeleton loop
#    #cdef int i, s2I, pI;
#    #cdef double dSkeleton[2];
#    dSkeleton = np.zeros(2)
#    while (s1I < last_index):
#        #//% Find small loops.
#        #// % Note: distal, looped sections are most likely touching;
#        #// % therefore, we don't remove these.
#        if (keep[s1I] != FLAG_MAX):
#            minI = s1I; #//% the minimum index for the loop
#            maxI = s1I; #//% the maximum index for the loop
#            
#            #//% Search backwards.
#            if (iSortC[s1I] > 0):
#                pI = iSortC[s1I] - 1; #//% the index for the sorted points
#                s2I = pSortC[pI]; #// % the second index for the skeleton loop
#                
#                dSkeleton[0] = absDiff(skeleton[s1I, 0], skeleton[s2I, 0]);
#                dSkeleton[1] = absDiff(skeleton[s1I, 1], skeleton[s2I, 1]);
#
#                while ((dSkeleton[0]<=1) or (dSkeleton[1]<=1)):
#                    if ((s2I > s1I) and (keep[s2I]!=FLAG_MAX) and (dSkeleton[0]<=1) and \
#                    (dSkeleton[1]<=1) and absDiff(s1I, s2I) < maxSkeletonOverlap):
#                        minI = min(minI, s2I);
#                        maxI = max(maxI, s2I);
#                    
#                    #// Advance the second index for the skeleton loop.
#                    pI = pI - 1;
#                    if(pI < 1):
#                        break;
#                    
#                    s2I = pSortC[pI];
#                    dSkeleton[0] = absDiff(skeleton[s1I, 0], skeleton[s2I, 0]);
#                    dSkeleton[1] = absDiff(skeleton[s1I, 1], skeleton[s2I, 1]);
#
#            #//% Search forwards.
#            if (iSortC[s1I]< last_index):
#                pI = iSortC[s1I] + 1; #//% the index for the sorted points
#                s2I = pSortC[pI]; #//% the second index for the skeleton loop
#                dSkeleton[0] = absDiff(skeleton[s1I, 0], skeleton[s2I, 0]);
#                dSkeleton[1] = absDiff(skeleton[s1I, 1], skeleton[s2I, 1]);
#
#                while ((dSkeleton[0]<=1) or (dSkeleton[1]<=1)):
#                    if ((s2I > s1I) and (keep[s2I]!=FLAG_MAX) \
#                    and (dSkeleton[0]<=1) and (dSkeleton[1]<=1) \
#                    and absDiff(s1I, s2I) < maxSkeletonOverlap):
#                        minI = min(minI, s2I);
#                        maxI = max(maxI, s2I);
#                    
#                    #// Advance the second index for the skeleton loop.
#                    pI = pI + 1;
#                    if (pI > last_index):
#                        break;
#                    
#                    s2I = pSortC[pI];
#                    dSkeleton[0] = absDiff(skeleton[s1I, 0], skeleton[s2I, 0]);
#                    dSkeleton[1] = absDiff(skeleton[s1I, 1], skeleton[s2I, 1]);
#
#
#            #//% Remove small loops.
#            if (minI < maxI):
#                #//% Remove the overlap.
#                if ((skeleton[minI,0] == skeleton[maxI,0]) and  \
#                (skeleton[minI,1] == skeleton[maxI,1])):
#                    for i in range(minI+1, maxI+1):
#                        keep[i] = FLAG_MAX;
#                        cnt_widths[minI] = min(cnt_widths[minI], cnt_widths[i]);
#                #//% Remove the loop.
#                else:
#                    if(minI < maxI - 1):
#                        for i in range(minI+1, maxI):
#                            keep[i] = FLAG_MAX;
#                            cnt_widths[minI] = min(cnt_widths[minI], cnt_widths[i]);
#                            cnt_widths[maxI] = min(cnt_widths[maxI], cnt_widths[i]);
#                        
#            #//% Advance the first index for the skeleton loop.
#            s1I = maxI if (s1I < maxI) else s1I + 1;
#        #//% Advance the first index for the skeleton loop.
#        else:
#            s1I = s1I + 1;
#    
#    
#    #cdef int 
#    newTotal = 0;
#    for i in range(number_points):
#        if (keep[i] != FLAG_MAX):
#            skeleton[newTotal, 0] = skeleton[i,0];
#            skeleton[newTotal, 1] = skeleton[i,1];
#            cnt_widths[newTotal] = cnt_widths[i];
#            newTotal+=1;
#
#    #//% The head and tail have no width.
#    cnt_widths[0] = 0;
#    cnt_widths[newTotal-1] = 0;
#    number_points = newTotal;
#    last_index = number_points-1
#    
#    #del iSortC, pSortC;
#    
#    #//% Heal the skeleton by interpolating missing points.
#    #cdef int 
#    
#    #cdef np.ndarray[np.float_t, ndim=2] 
#    cSkeleton = np.zeros((buff_size, 2), dtype = np.float); #//% pre-allocate memory
#    #cdef np.ndarray[np.float_t, ndim=1] 
#    cWidths = np.zeros(buff_size, dtype = np.float);
#            
#    #cdef int m
#    j = 0
#    #cdef float x,y, x1,x2, y1, y2, delY, delX, delW;
#    #cdef int points;
#    for i in range(last_index):
#        #//% Initialize the point differences.
#        y = absDiff(skeleton[i + 1, 0], skeleton[i, 0]);
#        x = abs(skeleton[i + 1, 1] - skeleton[i, 1]);
#        
#        #//% Add the point.
#        if ((y == 0 or y == 1) and (x == 0 or x == 1)):
#            cSkeleton[j,0] = skeleton[i,0];
#            cSkeleton[j,1] = skeleton[i,1];
#            
#            cWidths[j] = cnt_widths[i];
#            j +=1;
#        
#        #//% Interpolate the missing points.
#        else:
#            #points = <int>fmax(y, x);
#            points = int(max(y, x));
#            y1 = skeleton[i,0];
#            y2 = skeleton[i + 1,0];
#            delY = (y2-y1)/points;
#            x1 = skeleton[i,1];
#            x2 = skeleton[i + 1,1];
#            delX = (x2-x1)/points;
#            delW = (cnt_widths[i + 1] - cnt_widths[i])/points;
#            for m in range(points):
#                cSkeleton[j+m, 0] = round(y1 + m*delY);
#                cSkeleton[j+m, 1] = round(x1 + m*delX);
#                cWidths[j+m] = round(cnt_widths[i] + m*delW);
#            j += points;
#
#    
#    #//% Add the last point.
#    if ((cSkeleton[0,0] != skeleton[last_index,0]) or \
#    (cSkeleton[buff_size-1,1] != skeleton[last_index,1])):
#        cSkeleton[j,0] = skeleton[last_index,0];
#        cSkeleton[j,1] = skeleton[last_index,1];
#        cWidths[j] = cnt_widths[last_index];
#        j+=1;
#    
#    number_points = j;
#    
#    #//% Anti alias.
#    keep = np.arange(number_points)
#    #cdef int nextI
#    i = 0;
#    
#    while (i < number_points - 2):
#        #//% Smooth any stairs.
#        nextI = i + 2;
#        if ((absDiff(cSkeleton[i,0], cSkeleton[nextI,0])<=1) and (absDiff(cSkeleton[i,1], cSkeleton[nextI,1])<=1)):
#            keep[i + 1] = FLAG_MAX;
#            #//% Advance.
#            i = nextI;
#        #//% Advance.
#        else:
#            i+=1;
#    
#    newTotal = 0;
#    for i in range(number_points):
#        if (keep[i] != FLAG_MAX):
#            cSkeleton[newTotal,0] = cSkeleton[i,0];
#            cSkeleton[newTotal,1] = cSkeleton[i,1];
#            cWidths[newTotal] = cWidths[i];
#            newTotal+=1;
#        
#    
#    cSkeleton = cSkeleton[:newTotal, :]
#    cWidths = cWidths[:newTotal]    
#    
#    #% Clean up the rough skeleton.
#    #skeleton, cnt_widths = cleanSkeleton(skeleton, cnt_widths, worm_seg_length);
    
#    The head must be in position 0    
    assert head_ind == 0
    
    # Get the contour for each side.
    cnt_side1 = contour[:tail_ind+1, :].copy()
    cnt_side2 = np.vstack([contour[0,:], contour[:tail_ind-1:-1,:]])
    
    assert np.all(cnt_side1[0] == cnt_side2[0])
    assert np.all(cnt_side1[-1] == cnt_side2[-1])
    assert np.all(skeleton[-1] == cnt_side1[-1])
    assert np.all(skeleton[0] == cnt_side2[0])

    
