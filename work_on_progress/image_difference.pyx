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

cdef inline int imAbsDiff(int a, int b): 
    return a-b if a>b else b-a 
@cython.boundscheck(False)

def image_difference(np.ndarray[np.int_t, ndim=2] f, np.ndarray[np.int_t, ndim=2] g):
    #f, g are to one dim vector
    
    cdef int n_row = f.shape[0];
    cdef int n_col = f.shape[1];
    cdef int i, j;
    cdef double total = 0;
    #cdef double tot_pix = 0;
    
    for i in range(n_row):
        for j in range(n_col):
            total += <double>imAbsDiff(f[i,j],g[i,j])
            
    #print tot_pix
    return total/<double>f.size;
    