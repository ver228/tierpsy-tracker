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
def get_max_min(np.ndarray[np.float64_t, ndim=2] image):
    cdef double amax = image[0];
    cdef double amin = image[0];
    cdef int i;
        
    for i in range(tot_pix):
        if image[i] > amax:
            amax = image[i]
        elif image[i] < amin:
            amin = image[i]
    
    return amin, amax
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def normalize_image(image_ori):
    np.ndarray[np.float64_t, ndim=2] image = image_ori.astype(np.float64) 
    
    cdef int i;
    cdef double amin, amax, factor;
    cdef int tot_pix = image.size;    
    
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] image_norm = \
        np.zeros(image.shape, dtype=np.uint8);

    amin, amax = get_max_min(image)  
    factor = 255/(amax-amin);
    for i in range(tot_pix):
        image_norm[i] = image[i]*factor;

    return image_norm        
        