# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:56:10 2016

@author: ajaver
"""

cimport cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def getMaxMin(np.ndarray[np.float64_t, ndim=2] image):
    cdef double amax = image[0,0];
    cdef double amin = image[0,0];
    cdef double pix_val;
    cdef int i, j;
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pix_val = image[i,j]
            if pix_val > amax:
                amax = pix_val
            elif pix_val < amin:
                amin = pix_val
    
    return amin, amax
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def normalizeImage(image_ori):
    cdef np.ndarray[np.float64_t, ndim=2] image = image_ori.astype(np.float64) 
    
    cdef int i;
    cdef double amin, amax, factor;
    
    cdef np.ndarray[np.uint8_t, ndim=2] image_norm = \
        np.zeros(image_ori.shape, dtype=np.uint8);

    amin, amax = getMaxMin(image)  
    factor = 255/(amax-amin);
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_norm[i,j] = <char>((image[i,j]-amin)*factor);

    return image_norm, (amin, amax)    
        