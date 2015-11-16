# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:54:42 2015

@author: ajaver
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs

ctypedef np.float64_t DTYPE_t

cdef inline double imAbsDiff(unsigned char a, unsigned char b): 
    return <double>(a-b) if a>b else <double>(b-a) 
@cython.boundscheck(False)

def imageDifferenceMask(np.ndarray[np.uint8_t, ndim=2] f, np.ndarray[np.uint8_t, ndim=2] g):
    #f, g are to one dim vector
    
    cdef int n_row = f.shape[0];
    cdef int n_col = f.shape[1];
    cdef int i, k
    cdef double total;
    cdef double tot_pix = 0;
    
    for i in range(n_row):
        for j in range(n_col):
            if ( f[i,j] != 0 ) and ( g[i,j] != 0 ):
                total += imAbsDiff(f[i,j],g[i,j])
                tot_pix += 1;
        
    #print tot_pix
    if tot_pix > 0:
        return total/tot_pix;
    else:
        return -1;
    